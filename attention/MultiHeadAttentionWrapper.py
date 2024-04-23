import torch
from torch import nn

from attention.CasualAttention import CasualAttention


class MultiHeadAttentionWrapper(nn.Module):
    """
    .. class:: MultiHeadAttentionWrapper(nn.Module)

        Wrapper class for multi-head attention mechanism.

        :param d_in: The input dimension.
        :type d_in: int
        :param d_out: The output dimension.
        :type d_out: int
        :param block_size: The block size for attention computation.
        :type block_size: int
        :param dropout: The dropout probability.
        :type dropout: float
        :param num_heads: The number of attention heads.
        :type num_heads: int
        :param qkv_bias: Whether to include biases in the query, key, and value projections. Default is False.
        :type qkv_bias: bool

        .. automethod:: forward
    """
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, block_size, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)