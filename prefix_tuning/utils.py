def tokenize_fn(example, task_name, tokenizer):
    if task_name == "e2e_nlg":
        sources = example["meaning_representation"]
        targets = example["human_reference"]
    elif task_name == "web_nlg":
        sources = example["input"]
        targets = example["target"]
    elif task_name == "dart":
        sources = example["tripleset"]
        targets = example["annotations"]
    else:
        raise NotImplementedError(f"task_name '{task_name}' not implemented")

    sep_token = tokenizer.sep_token
    sep_token_id = tokenizer.sep_token_id

    texts = [f"{src} {sep_token} {tgt}" for src, tgt in zip(sources, targets)]
    encoded = tokenizer(texts, truncation=True)

    labels = []
    for ids in encoded["input_ids"]:
        if sep_token_id in ids:
            sep_token_pos = ids.index(sep_token_id)
            label = [-100] * (sep_token_pos + 1) + ids[sep_token_pos + 1:]
        else:
            label = [-100] * len(ids)
        labels.append(label)

    encoded["labels"] = labels

    return encoded