def tokenize_fn(example, task_name, tokenizer):
    if task_name in ("ax", "mnli", "mnli-m", "mnli-mm"):
        return tokenizer(example["premise"], example["hypothesis"], truncation=True)
    elif task_name == "qqp":
        return tokenizer(example["question1"], example["question2"], truncation=True)
    elif task_name == "qnli":
        return tokenizer(example["question"], example["sentence"], truncation=True)
    elif task_name in ("sst2", "cola"):
        return tokenizer(example["sentence"], truncation=True)
    else:
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def postprocess_fn(dataset, task_name):
    # Remove unnecessary columns based on the task
    columns_to_remove = ["idx"]
    if task_name in ("ax", "mnli", "mnli-m", "mnli-mm"):
        columns_to_remove.extend(["premise", "hypothesis"])
    elif task_name == "qqp":
        columns_to_remove.extend(["question1", "question2"])
    elif task_name == "qnli":
        columns_to_remove.extend(["question", "sentence"])
    elif task_name in ("sst2", "cola"):
        columns_to_remove.extend(["sentence"])
    else:  # mrpc, rte, wnli, stsb
        columns_to_remove.extend(["sentence1", "sentence2"])
    
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")
    return dataset
