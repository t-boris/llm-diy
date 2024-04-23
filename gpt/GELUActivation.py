import torch
from torch import nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        """
        Compute the forward pass of the given method.

        :param x: The input value.
        :return: The output value after applying the forward pass.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))