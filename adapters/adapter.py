import os
import types
import torch
import torch.nn as nn

from transformers import BertConfig, BertForSequenceClassification
from safetensors.torch import save_file, load_file

class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x, *args, **kwargs):
        return x + self.up(self.act(self.down(x)))

class BertWithAdapters(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int, bottleneck_size: int):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=num_labels
        )
        self.bottleneck_size = bottleneck_size
        self.checkpoint = checkpoint
        self.model.config.bottleneck_size = bottleneck_size
        self.model.config.checkpoint = checkpoint

        self._add_adapters()
        self._freeze_params()

    @property
    def bert(self):
        """Access to the BERT backbone for compatibility."""
        return self.model.bert

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def save_pretrained(self, save_dir: str, filename: str = "model.safetensors"):
        os.makedirs(save_dir, exist_ok=True)
        
        # Ensure checkpoint is saved to config and fix architecture
        self.model.config.checkpoint = self.checkpoint
        self.model.config.architectures = ["BertForSequenceClassification"]
        self.model.config.save_pretrained(save_dir)

        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        state_dict = self.model.state_dict()
        trainable_state_dict = {
            n: t for n, t in state_dict.items()
            if n in trainable
        }

        save_file(trainable_state_dict, os.path.join(save_dir, filename))

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
            bottleneck_size=getattr(cfg, "bottleneck_size", 64)
        )

        trainable_state_dict = load_file(os.path.join(save_dir, filename), device="cpu")
        missing, unexpected = wrapper.model.load_state_dict(trainable_state_dict, strict=False)

        if unexpected:
            raise RuntimeError(f"Unexpected keys in trainable checkpoint: {unexpected}")
        
        wrapper._freeze_params()
        return wrapper
    
    def _add_adapters(self):
        backbone = self.model.bert if hasattr(self.model, "bert") else self.model
        assert hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"), "Expected a BERT-like model with .encoder.layer"

        hidden_size = backbone.config.hidden_size

        for layer in backbone.encoder.layer:
            layer.attention.output.adapter = Adapter(hidden_size, self.bottleneck_size)
            layer.attention.output.adapter.to(layer.attention.output.dense.weight.device)

            layer.output.adapter = Adapter(hidden_size, self.bottleneck_size)
            layer.output.adapter.to(layer.output.dense.weight.device)

        def attn_forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.adapter(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states

        def ffn_forward(self, hidden_states, input_tensor):
            hidden_states = self.dense(hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.adapter(hidden_states)
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
            return hidden_states
        
        # Update forward methods for attention output and feed-forward output
        for layer in backbone.encoder.layer:
            layer.attention.output.forward = types.MethodType(attn_forward, layer.attention.output)
            layer.output.forward = types.MethodType(ffn_forward, layer.output)

    def _freeze_params(self):
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        # Unfreeze adapters
        for n, m in self.model.named_modules():
            if isinstance(m, Adapter):
                for p in m.parameters():
                    p.requires_grad = True

        # Unfreeze all LayerNorm parameters
        for m in self.model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True

        # Unfreeze classification head
        for p in self.model.classifier.parameters():
            p.requires_grad = True