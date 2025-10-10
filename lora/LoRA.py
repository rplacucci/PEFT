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
            self.matA = nn.Linear(self.in_features, self.rank, bias=False)
            self.matB = nn.Linear(self.rank, self.out_features, bias=False)
            # Initialize parameter matrices
            nn.init.normal_(self.matA.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.matB.weight)

        # Freeze base weights
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.rank > 0:
            return y + self.scaling * self.matB(self.matA(x))

    @torch.no_grad()
    def merge_lora(self):
        if self.rank > 0:
            delta = self.matB.weight @ self.matA.weight
            self.weight.add_(self.scaling * delta)
            self.matA.weight.zero_()
            self.matB.weight.zero_()