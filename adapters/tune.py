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
from transformers import BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding, logging, get_scheduler
from .utils import tokenize_fn, postprocess_fn
from .adapter import Adapter, add_adapters

# accelerate launch -m adapters.tune 

# Disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# Config argparser
parser = argparse.ArgumentParser(description="Fine-tune BERT on a GLUE task (adapters)")
parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="Checkpoint of pre-trained BERT model")
parser.add_argument("--task_name", type=str, default="qqp", help="Name of the GLUE task to train on", choices=["mnli", "qqp", "qnli", "sst2", "cola", "stsb", "mrpc", "rte", "wnli"])
parser.add_argument("--bottleneck_size", type=int, default=64, help="Size of adapter bottleneck dimension")
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
bottleneck_size = args.bottleneck_size
lr = args.lr
betas = args.betas
eps = args.eps
weight_decay = args.weight_decay
batch_size = args.batch_size
warmup_ratio = args.warmup_ratio
n_epochs = args.n_epochs

# Config directories
log_dir = f"./logs/adapters/{task_name}"
os.makedirs(log_dir, exist_ok=True)

out_dir = f"./models/adapters/{task_name}"
os.makedirs(out_dir, exist_ok=True)
model_save_path = os.path.join(out_dir, f"{checkpoint}-glue-{task_name}-bottleneck_size-{bottleneck_size}-batch_size-{batch_size}-lr-{lr:.0e}-n_epochs-{n_epochs}")

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
accelerator.init_trackers(f"{checkpoint}-glue-{task_name}-bottleneck_size-{bottleneck_size}-batch_size-{batch_size}-lr-{lr:.0e}-n_epochs-{n_epochs}")
world_size = accelerator.num_processes
accelerator.print(f"Initialized {accelerator.__class__.__name__} with {world_size} distributed processes")

# Set seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Config tokenizer
tokenizer = BertTokenizer.from_pretrained(checkpoint)
vocab_size = tokenizer.vocab_size
accelerator.print(f"Loaded {tokenizer.__class__.__name__} with vocab size {vocab_size:,}")

# Config model
num_labels = 3 if task_name in ("mnli", "ax") else 1 if task_name == "stsb" else 2
model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
total_params = sum(p.numel() for p in model.bert.parameters() if p.requires_grad)
accelerator.print(f"Loaded {model.__class__.__name__} with {num_labels} labels and {total_params:,} encoder parameters")

# Add adapter
model = add_adapters(model, bottleneck_size)
accelerator.print(f"Added adapters with bottleneck size {bottleneck_size}")

# Freeze everything
for p in model.parameters():
    p.requires_grad = False

# Unfreeze adapters
adapter_trainable = 0
for name, module in model.named_modules():
    if isinstance(module, Adapter):
        for p in module.parameters():
            p.requires_grad = True
            adapter_trainable += p.numel()

# Unfreeze all LayerNorm parameters
ln_trainable = 0
for m in model.modules():
    if isinstance(m, nn.LayerNorm):
        for p in m.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                ln_trainable += p.numel()

# Unfreeze classification head
head_trainable = 0
for p in model.classifier.parameters():
    p.requires_grad = True
    head_trainable += p.numel()

# Print number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.print(f"Froze backbone. Model now has {trainable_params:,} trainable parameters ({trainable_params/total_params:.2%})")

# Load dataset
glue = load_dataset("glue", task_name)
valid_split = "validation_matched" if task_name == "mnli" else "validation"
accelerator.print(f"Loaded GLUE {task_name} with {sum([len(glue[split]) for split in ["train", valid_split]]):,} train/val sequences")

dataset = glue.map(tokenize_fn, batched=True, fn_kwargs={"task_name": task_name, "tokenizer": tokenizer})
dataset = postprocess_fn(dataset, task_name)

# Prepare dataloaders
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    dataset=dataset["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

valid_dataloader = DataLoader(
    dataset=dataset[valid_split],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Load evaluation metric
metric = evaluate.load("glue", task_name)

# Config optimizer (do not apply weight decay to bias or LayerNorm weights)
optim_groups = [
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad and "bias" not in n and "LayerNorm.weight" not in n
        ],
        "weight_decay": weight_decay,
    },
    {
        "params": [
            p for n, p in model.named_parameters()
            if p.requires_grad and ("bias" in n or "LayerNorm.weight" in n)
        ],
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
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        if task_name == "stsb":
            logits = logits.squeeze(-1)

        predictions = (
            logits if task_name == "stsb" else torch.argmax(logits, dim=-1)
        )

        # Gather preds and refs across processes then add to metric
        preds = accelerator.gather_for_metrics(predictions)
        refs  = accelerator.gather_for_metrics(batch["labels"])

        metric.add_batch(
            predictions=preds.tolist() if task_name == "stsb" else preds,
            references=refs,
        )
    
    # Log to tensorboard
    scores = metric.compute()
    accelerator.log(scores, step=epoch)
    
    # Print to terminal
    scores_str = " | ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
    accelerator.print(f"(valid) epoch: {epoch:2d} | {scores_str}")

# Save final model
accelerator.wait_for_everyone()
unwrapped = accelerator.unwrap_model(model)
trainable = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
state_dict = unwrapped.state_dict()
trainable_state_dict = {
    n: t
    for n, t in state_dict.items()
    if n in trainable
}
unwrapped.save_pretrained(
    model_save_path,
    state_dict=trainable_state_dict,
    save_function=accelerator.save,
    safe_serialization=True
)

accelerator.print(f"Model saved to {model_save_path}")

# Flush trackers
accelerator.end_training()
accelerator.print("Goodbye!")