import types
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)

    def forward(self, x, *args, **kwargs):
        return x + self.up(self.act(self.down(x)))

def add_adapters(model, bottleneck_size=64):
    backbone = model.bert if hasattr(model, "bert") else model
    assert hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"), "Expected a BERT-like model with .encoder.layer"

    hidden_size = backbone.config.hidden_size

    for layer in backbone.encoder.layer:
        layer.attention.output.adapter = Adapter(hidden_size, bottleneck_size)
        layer.attention.output.adapter.to(layer.attention.output.dense.weight.device)

        layer.output.adapter = Adapter(hidden_size, bottleneck_size)
        layer.output.adapter.to(layer.output.dense.weight.device)

    def selfoutput_forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

    def ffnoutput_forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
    # Update forward methods for attention output and feed-forward output
    for layer in backbone.encoder.layer:
        layer.attention.output.forward = types.MethodType(selfoutput_forward, layer.attention.output)
        layer.output.forward = types.MethodType(ffnoutput_forward, layer.output)

    return model
