import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class PrefixEncoder(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            head_dim: int,
            d_model: int,
            prefix_len: int = 10,
            hidden_dim: int = 512
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.prefix_len = prefix_len

        self.prefix_tokens = torch.arange(prefix_len).long()
        self.embedding = nn.Embedding(prefix_len, d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2 * n_layers * n_heads * head_dim)
        )

    def forward(self, batch_size: int, device: torch.device):
        # (prefix_len, d_model)
        prefix_tokens = self.prefix_tokens.to(device)
        prefix_embed = self.embedding(prefix_tokens)

        # (prefix_len, 2 * n_layers * n_heads * head_dim)
        past = self.feed_forward(prefix_embed)

        # (batch_size, prefix_len, 2 * n_layers * n_heads * head_dim)
        past = past.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # (batch_size, prefix_len, 2, n_layers, n_heads, head_dim)
        past = past.view(
            batch_size,
            self.prefix_len,
            2,
            self.n_layers,
            self.n_heads,
            self.head_dim,
        )

        # (n_layers, 2, batch_size, n_heads, prefix_len, head_dim)
        past = past.permute(3, 2, 0, 4, 1, 5).contiguous()

        # tuple of length n_layers, each: (k, v) with shape (batch, n_heads, prefix_len, head_dim)
        past_key_values = tuple((past[i, 0], past[i, 1]) for i in range(self.n_layers))
        return past_key_values

class GPT2WithPrefixTuning(nn.Module):
    def __init__(
            self,
            checkpoint: str = "gpt2",
            prefix_len: int = 10,
            prefix_hidden_dim: int = 512,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.prefix_len = prefix_len
        self.prefix_hidden_dim = prefix_hidden_dim

        self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
        cfg = self.model.config

        # Freeze GPT-2 params
        for p in self.model.parameters():
            p.requires_grad = False

        n_layers = cfg.n_layer
        n_heads = cfg.n_head
        d_model = cfg.n_embd
        head_dim = d_model // n_heads

        self.prefix = PrefixEncoder(
            n_layers=n_layers,
            n_heads=n_heads,
            head_dim=head_dim,
            d_model=d_model,
            prefix_len=prefix_len,
            hidden_dim=prefix_hidden_dim,
        )

    @property
    def gpt2(self):
        return self.model.gpt2
    
    @property
    def trainable_params(self):
        return self.prefix.parameters()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        device = input_ids.device
        past_key_values = self.prefix(batch_size=batch_size, device=device)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        prefix_mask = torch.ones((batch_size, self.prefix_len), device=device, dtype=attention_mask.dtype)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=False
        )
        



        
    


