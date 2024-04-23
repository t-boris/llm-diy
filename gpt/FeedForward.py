from torch import nn

from gpt.GELUActivation import GELU


class FeedForward(nn.Module):
    """

    FeedForward

    A class representing a feedforward neural network module.

    Methods:
    - __init__(self, cfg): Initializes the FeedForward module.
    - forward(self, x): Performs a forward pass through the feedforward network.

    Attributes:
    - layers: A sequential container holding the layers of the feedforward network.

    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg.drop_rate)
        )

    def forward(self, x):
        return self.layers(x)