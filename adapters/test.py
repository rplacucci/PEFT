import os
import time
import argparse
import pandas
import torch
import torch.nn as nn
import evaluate
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding, logging, get_scheduler
from .utils import tokenize_fn, postprocess_fn
from .adapter import BertWithAdapters

# accelerate launch -m adapters.test

# Disable warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

# Config argparser
parser = argparse.ArgumentParser(description="Fine-tune BERT on a GLUE task (adapters)")
parser.add_argument("--checkpoint", type=str, default="bert-base-uncased", help="Checkpoint of pre-trained BERT model")
parser.add_argument("--task_name", type=str, default="qqp", help="Name of the GLUE task to test on", choices=["mnli-m", "mnli-mm", "qqp", "qnli", "sst2", "cola", "stsb", "mrpc", "rte", "wnli", "ax"])
parser.add_argument("--batch_size", type=int, default=32, help="Size of batch to test with")
args = parser.parse_args()

checkpoint = args.checkpoint
task_name = args.task_name
batch_size = args.batch_size

# Config directories
out_dir = f"./outputs/adapters/submission-{checkpoint}"
os.makedirs(out_dir, exist_ok=True)

# Config accelerator
accelerator = Accelerator()
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

# Load model
model_path = f"./models/adapters/{checkpoint}-glue-{"mnli" if task_name in ("mnli-m", "mnli-mm", "ax") else task_name}"
model = BertWithAdapters.from_pretrained(save_dir=model_path)
total_params = sum(p.numel() for p in model.bert.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.print(f"Loaded {model.__class__.__name__} with {trainable_params:,} trainable parameters ({trainable_params/total_params:.2%})")

# Load dataset
glue = load_dataset(
    "glue", 
    "mnli" if task_name in ("mnli-m", "mnli-mm") else task_name
)
split = "test_matched" if task_name == "mnli-m" else "test_mismatched" if task_name == "mnli-mm" else "test"
accelerator.print(f"Loaded GLUE {task_name} with {len(glue[split])} test sequences")

dataset = glue.map(tokenize_fn, batched=True, fn_kwargs={"task_name": task_name, "tokenizer": tokenizer})
dataset = postprocess_fn(dataset, task_name, test=True)

# Prepare dataloaders
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

test_dataloader = DataLoader(
    dataset=dataset[split],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)

# Prepare distributed testing
test_dataloader, model = accelerator.prepare(test_dataloader, model)

# Begin testing
preds = []
model.eval()
with torch.no_grad():
    progress_bar = tqdm(test_dataloader, desc="Testing", disable=not accelerator.is_main_process)
    for batch in progress_bar:
        outputs = model(**batch)
        logits = outputs.logits

        if task_name == "stsb":
            logits = logits.squeeze(-1)

        predictions = (
            logits if task_name == "stsb" else torch.argmax(logits, dim=-1)
        )

        # Gather preds across processes
        gathered = accelerator.gather_for_metrics(predictions)
        if accelerator.is_main_process:
            preds.extend(gathered.cpu().tolist())

if accelerator.is_main_process:
    # Format submission according to task
    if task_name in ("ax", "mnli-m", "mnli-mm"):
        preds = [
            "entailment" if p == 0 else
            "neutral" if p == 1 else
            "contradiction"
            for p in preds
        ]

    if task_name in ("qnli", "rte"):
        preds = [
            "entailment" if p == 0 else
            "not_entailment"
            for p in preds
        ]

    if task_name == "stsb":
        preds = [min(max(p, 0), 5) for p in preds]
        preds = [f"{p:.3f}" for p in preds]

    print("Formatted results")

    # Save submission
    fname = {
        "cola": "CoLA.tsv",
        "mnli-m": "MNLI-m.tsv",
        "mnli-mm": "MNLI-mm.tsv",
        "mrpc": "MRPC.tsv",
        "qnli": "QNLI.tsv",
        "qqp": "QQP.tsv",
        "rte": "RTE.tsv",
        "sst2": "SST-2.tsv",
        "stsb": "STS-B.tsv",
        "wnli": "WNLI.tsv",
        "ax": "AX.tsv"
    }[task_name]

    out_path = os.path.join(out_dir, fname)
    df = pandas.DataFrame({
        'prediction': preds
    })
    df.to_csv(out_path, sep="\t", index=True, index_label="index")
    print(f"Saved {len(df)} rows to {out_path}")

accelerator.wait_for_everyone()
accelerator.end_training()
accelerator.print("Goodbye!")