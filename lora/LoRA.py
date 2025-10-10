import os
import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertConfig, BertForSequenceClassification
from safetensors.torch import save_file, load_file

class LoRA(nn.Module):
    def __init__(self, layer: nn.Linear, rank: int = 8, alpha: int = 18):
        super().__init__()
        assert isinstance(layer, nn.Linear)
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = (alpha / rank) if rank > 0 else 0.0

        # Parameters from pretrained layer
        self.weight = nn.Parameter(layer.weight.detach().clone())
        self.bias = nn.Parameter(layer.bias.detach().clone()) if layer.bias is not None else None

        # LoRA matrices
        if rank > 0:
            self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
            self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
            # Initialize parameter matrices
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B.weight)

        # Freeze base weights
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            return y + self.scaling * self.lora_B(self.lora_A(x))
        return y

    @torch.no_grad()
    def merge_lora(self):
        if self.rank > 0:
            delta = self.lora_B.weight @ self.lora_A.weight
            self.weight.add_(self.scaling * delta)
            self.lora_A.weight.zero_()
            self.lora_B.weight.zero_()

class BertWithLoRA(nn.Module):
    def __init__(
        self, checkpoint: str, 
        num_labels: int, rank: int, alpha: int,
        target_modules=(
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense"
        )
    ):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels
        )
        self.checkpoint = checkpoint
        self.rank = rank
        self.alpha = alpha
        self.target_modules = tuple(target_modules)

        # Keep for reproducible export
        self.model.config.lora_r = rank
        self.model.config.lora_alpha = alpha
        self.model.config.checkpoint = checkpoint
        self.model.config.target_modules = list(target_modules)
        self.model.config.architectures = ["BertForSequenceClassification"]

        self._inject_lora()
        self._freeze_params()

    @property
    def bert(self):
        return self.model.bert

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def save_pretrained(self, save_dir: str, filename: str = "model.safetensors"):
        os.makedirs(save_dir, exist_ok=True)
        self.model.config.save_pretrained(save_dir)

        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        sd = self.model.state_dict()
        trainable_sd = {n: t for n, t in sd.items() if n in trainable}
        save_file(trainable_sd, os.path.join(save_dir, filename))

    @classmethod
    def from_pretrained(cls, checkpoint: str | None = None, save_dir: str | None = None, filename: str = "model.safetensors"):
        if save_dir is None:
            raise ValueError("Provide save_dir containing config.json and the safetensors file.")
        
        cfg = BertConfig.from_pretrained(save_dir)
        if checkpoint is None:
            checkpoint = getattr(cfg, "checkpoint", "bert-base-uncased")

        wrapper = cls(
            checkpoint=checkpoint,
            num_labels=cfg.num_labels,
            rank=getattr(cfg, "lora_r", 8),
            alpha=getattr(cfg, "lora_alpha", 16),
            target_modules=getattr(
                cfg, 
                "target_modules", (
                    "attention.self.query",
                    "attention.self.key",
                    "attention.self.value",
                    "attention.output.dense"
                )
            )
        )
        
        trainable_sd = load_file(os.path.join(save_dir, filename), device="cpu")
        missing, unexpected = wrapper.model.load_state_dict(trainable_sd, strict=False)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in trainable checkpoint: {unexpected}")
        
        wrapper._freeze_params()
        return wrapper
    
    def _inject_lora(self):
        backbone = self.model.bert if hasattr(self.model, "bert") else self.model
        assert hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"), "Expected a BERT-like model with .encoder.layer"

        for layer in backbone.encoder.layer:
            for path in self.target_modules:
                parent, attr = self._resolve_parent_and_attr(layer, path)
                mod = getattr(parent, attr, None)
                if isinstance(mod, nn.Linear):
                    setattr(parent, attr, LoRA(mod, rank=self.rank, alpha=self.alpha))

    def _resolve_parent_and_attr(self, root, dotted: str):
        parts = dotted.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    def _freeze_params(self):
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze LoRA parameters
        for n, m in self.model.named_modules():
            if isinstance(m, LoRA):
                if hasattr(m, "lora_A") and m.lora_A is not None:
                    for p in m.lora_A.parameters():
                        p.requires_grad = True
                if hasattr(m, "lora_B") and m.lora_B is not None:
                    for p in m.lora_B.parameters():
                        p.requires_grad = True

        # Unfreeze classification head
        for p in self.model.classifier.parameters():
            p.requires_grad = True
    
    @torch.no_grad()
    def merge_lora_for_inference(self):
        backbone = self.model.bert if hasattr(self.model, "bert") else self.model
        for layer in backbone.encoder.layer:
            for path in self.target_modules:
                parent, attr = self._resolve_parent_and_attr(layer, path)
                mod = getattr(parent, attr, None)
                if isinstance(mod, LoRA):
                    mod.merge_lora()