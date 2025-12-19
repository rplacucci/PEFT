import os
import time
import argparse
import torch
import torch.nn as nn
import evaluate
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import GPT2TokenizerFast, DataCollatorWithPadding, logging, get_scheduler
from .utils import tokenize_fn
from .prefix import GPT2WithPrefixTuning

# accelerate launch -m prefix_tuning.tune

# Disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# Config argparser
parser = argparse.ArgumentParser(description="Fine-tune BERT on a GLUE task (adapters)")
parser.add_argument("--checkpoint", type=str, default="gpt2", help="Checkpoint of pre-trained BERT model")
parser.add_argument("--task_name", type=str, default="e2e_nlg", choices=["e2e_nlg", "web_nlg", "dart"], help="Name of table-to-text task to train on")
parser.add_argument("--prefix_len", type=int, default=10, help="Number of prefix tokens")
parser.add_argument("--prefix_hidden_dim", type=int, default=512, help="Size of hidden dimension in prefix encoder")
parser.add_argument("--lr", type=float, default=1e-4, help="Maximum learning rate for the optimizer")
parser.add_argument("--betas", nargs=2, type=float, default=(0.9, 0.999), help="Beta values for the optimizer")
parser.add_argument("--eps", type=float, default=1e-6, help="Constant to stabilize division in the optimizer update rule")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for the optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Size of batch to train with")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Initial fraction of training steps with linear LR warmup")
parser.add_argument("--n_epochs", type=int, default=4, help="Number of training epochs")
args = parser.parse_args()

checkpoint = args.checkpoint
task_name = args.task_name
prefix_len = args.prefix_len
prefix_hidden_dim = args.prefix_hidden_dim
lr = args.lr
betas = args.betas
eps = args.eps
weight_decay = args.weight_decay
batch_size = args.batch_size
warmup_ratio = args.warmup_ratio
n_epochs = args.n_epochs

# Config directories
log_dir = f"./logs/prefix_tuning/{task_name}"
os.makedirs(log_dir, exist_ok=True)

out_dir = f"./models/prefix_tuning/{task_name}"
run_id = f"{checkpoint}-{task_name}-prefix_len-{prefix_len}-prefix_hidden_dim-{prefix_hidden_dim}-batch_size-{batch_size}-lr-{lr:.0e}-n_epochs-{n_epochs}"
os.makedirs(out_dir, exist_ok=True)
model_save_path = os.path.join(out_dir, run_id)

# Explicitly set CUDA device
if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local_rank)

# Config distributed training with accelerate
accelerator = Accelerator(
    log_with=["tensorboard"],
    project_dir=log_dir,
    device_placement=True
)

accelerator.init_trackers(run_id)
world_size = accelerator.num_processes
accelerator.print(f"Initialized {accelerator.__class__.__name__} with {world_size} distributed processes")

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Config tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained(checkpoint)
added_tokens = tokenizer.add_special_tokens({"sep_token": "<|sep|>"})
vocab_size = tokenizer.vocab_size
accelerator.print(f"Loaded {tokenizer.__class__.__name__} with vocab size {vocab_size:,} (+{added_tokens} added_tokens)")

# Config model
model = GPT2WithPrefixTuning(checkpoint, prefix_len, prefix_hidden_dim)
model.resize_token_embeddings(len(tokenizer))
gpt2_params = sum(p.numel() for p in model.gpt2.parameters())
prefix_params = sum(p.numel() for p in model.prefix.parameters())
accelerator.print(f"Loaded {model.__class__.__name__} with {prefix_params:,} trainable parameters ({prefix_params/(gpt2_params + prefix_params):.2%})")

# Load dataset
if task_name == "e2e_nlg":
    nlg = load_dataset("tuetschek/e2e_nlg")
elif task_name == "web_nlg":
    nlg = load_dataset("GEM/web_nlg", "en")
elif task_name == "dart":
    nlg = load_dataset("Yale-LILY/dart")
else:
    raise ValueError(f"task_name '{task_name}' not compatible")

accelerator.print(f"Loaded {task_name} with {sum([len(nlg[split]) for split in list(nlg.keys())]):,} train/val/test sequences")

# Prepare dataset
dataset = nlg.map(
    tokenize_fn,
    batched=True,
    fn_kwargs={"task_name": task_name, "tokenizer": tokenizer},
    remove_columns=nlg['train'].column_names,
    load_from_cache_file=False
)

# Prepare dataloaders
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    dataset=dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    dataset=dataset["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Load evaluation metric
metrics = evaluate.combine(["sacrebleu", "meteor"])

# Config optimizer (do not apply weight decay to bias or LayerNorm weights)
no_decay = ["bias", "LayerNorm.weight"]
optim_groups = [
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad and any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, eps=eps)

# Prepare distributed learning
train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, valid_dataloader, model, optimizer
)

# Config LR scheduler
total_steps = n_epochs * len(train_dataloader)
warmup_steps = int(warmup_ratio * total_steps)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_training_steps=total_steps,
    num_warmup_steps=warmup_steps
)

# Begin training
step = 0
for epoch in range(n_epochs):
    model.train()
    for batch in train_dataloader:
        start = time.time()

        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)
        norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        max_len = batch["input_ids"].shape[1]
        tokens_per_sec = int(batch_size * max_len * world_size / elapsed) if elapsed > 0 else 0

        # Log to tensorboard
        step += 1
        loss = loss.detach().item()
        lr = scheduler.get_last_lr()[0]
        accelerator.log(
            {
                "loss": loss,
                "grad_norm": norm,
                "lr": lr,
            },
            step=step,
        )

        # Print to terminal
        accelerator.print(f"(train) epoch: {epoch:2d} | step: {step:4d} | loss: {loss:.4f} | lr: {lr:.4e} | norm: {norm:.4f} | tok/sec: {tokens_per_sec:,}")

    model.eval()
    for batch in valid_dataloader:
        labels = batch.pop("labels")

        with torch.no_grad():
            outputs = model.generate(
                **batch,
                num_beams=5,
                max_new_tokens=64,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode predictions
        prompt_len = batch["input_ids"].shape[1]
        gen_only = outputs[:, prompt_len:]
        preds = tokenizer.batch.decode(gen_only, skip_special_tokens=True)

        # Decode references
        labels[labels == -100] = tokenizer.pad_token_id
        refs = tokenizer.batch.decode(labels, skip_special_tokens=True)

        # Gather preds and refs across processes then add to metric
        preds = accelerator.gather_for_metrics(preds)
        refs = accelerator.gather_for_metrics(refs)

        metrics.add_batch(
            predictions=preds,
            references=[[r] for r in refs],
        )
    
    # Log to tensorboard
    scores = metrics.compute()
    accelerator.log(scores, step=epoch)
    
    # Print to terminal
    scores_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
    accelerator.print(f"(valid) epoch: {epoch:2d} | {scores_str}")

# Save final model
accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
unwrapped.save_pretrained(model_save_path)
accelerator.print(f"Model saved to {model_save_path}")

# Flush trackers
accelerator.end_training()
accelerator.print("Goodbye!")