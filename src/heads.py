"""
Model head definitions for triage (binary) and defect diagnosis (multi-label).
Both heads operate on frozen image embeddings — no backbone fine-tuning.
"""
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Single linear layer — the 'linear probe' baseline."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    """Two-layer MLP with GELU + dropout."""
    def __init__(self, d_in: int, d_out: int,
                 hidden: int = 256, p: float = 0.3, multilabel: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, d_out),
        )
        self.multilabel = multilabel

    def forward(self, x):
        return self.net(x)


class JointHead(nn.Module):
    """
    Unified multi-task head (C3).
    Shared trunk → two output branches:
      - triage:  1 logit  (answerable?)
      - defect:  n_defect logits  (which quality flaw?)
    """
    def __init__(self, d_in: int, n_defect: int = 7,
                 hidden: int = 256, p: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(p),
        )
        self.triage_head = nn.Linear(hidden, 1)
        self.defect_head = nn.Linear(hidden, n_defect)

    def forward(self, x):
        h = self.trunk(x)
        return self.triage_head(h), self.defect_head(h)


def build_head(head_type: str, d_in: int, d_out: int,
               hidden: int = 256, p: float = 0.3,
               multilabel: bool = False) -> nn.Module:
    """Factory function for head construction."""
    if head_type == "linear":
        return LinearHead(d_in, d_out)
    elif head_type == "mlp":
        return MLPHead(d_in, d_out, hidden=hidden, p=p, multilabel=multilabel)
    elif head_type == "joint":
        return JointHead(d_in, n_defect=d_out, hidden=hidden, p=p)
    else:
        raise ValueError(f"Unknown head_type: {head_type}")
