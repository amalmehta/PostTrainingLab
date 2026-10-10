"""A tiny pre-LayerNorm transformer policy, written out by hand.

It is deliberately plain (no nn.TransformerEncoder, ReLU instead of GELU)
so site/app.js can reproduce the exact forward pass from exported weights.
"""
import base64

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import SEQ_LEN, VOCAB

N_ACTIONS = 4


class Block(nn.Module):
    def __init__(self, d, heads, mlp):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)

    def forward(self, x):
        B, T, D = x.shape
        h = self.heads
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, h, D // h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-1, -2)) / (D // h) ** 0.5
        y = att.softmax(-1) @ v
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.relu(self.fc1(self.ln2(x))))


class Policy(nn.Module):
    def __init__(self, d=48, heads=4, layers=3, mlp=96):
        super().__init__()
        self.config = dict(d=d, heads=heads, layers=layers, mlp=mlp)
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.blocks = nn.ModuleList(Block(d, heads, mlp) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_ACTIONS)

    def forward(self, tokens):
        """tokens [B, SEQ_LEN] long -> action logits [B, 4] (read off CLS)."""
        x = self.tok(tokens) + self.pos.weight[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x[:, 0]))

    @torch.no_grad()
    def act(self, tokens, greedy=False, temperature=1.0, gen=None):
        logits = self(torch.as_tensor(tokens))
        if greedy:
            return logits.argmax(-1).numpy()
        probs = (logits / temperature).softmax(-1)
        return torch.multinomial(probs, 1, generator=gen).squeeze(-1).numpy()


def export_weights(model, dtype="f4"):
    """JSON-friendly dict: config + each tensor as base64 little-endian floats
    (float32 by default; "f2" halves the size for training snapshots)."""
    tensors = {}
    for name, t in model.state_dict().items():
        arr = t.detach().cpu().numpy().astype("<" + dtype)
        tensors[name] = dict(shape=list(arr.shape),
                             data=base64.b64encode(arr.tobytes()).decode())
    return dict(config=model.config, dtype=dtype, tensors=tensors)


def load_weights(blob):
    model = Policy(**blob["config"])
    dt = "<" + blob.get("dtype", "f4")
    sd = {k: torch.from_numpy(np.frombuffer(base64.b64decode(v["data"]), dtype=dt)
                              .astype(np.float32).reshape(v["shape"]).copy())
          for k, v in blob["tensors"].items()}
    model.load_state_dict(sd)
    return model
