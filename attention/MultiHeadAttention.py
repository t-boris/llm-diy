import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    This class is used to perform multi-head attention on an input tensor.
    It takes the input tensor and applies linear transformations to compute
    queries, keys, and values. It then computes attention scores and applies
    a mask to ignore certain positions. Finally, it computes a weighted sum
    of the values based on the attention scores to obtain the output.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        block_size (int): Block size for mask calculation.
        dropout (float): Dropout probability.
        num_heads (int): Number of heads.
        qkv_bias (bool, optional): Whether to include bias in linear
            transformations for queries, keys, and values. Defaults to False.
    """
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        """
        Forward pass of the method.

        :param x: Input tensor of shape (batch_size, nums_tokens, d_in).
        :return: Output tensor of shape (batch_size, nums_tokens, d_out).
        """
        b, nums_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, nums_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, nums_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, nums_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:nums_tokens, :nums_tokens]
        mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, nums_tokens, self.d_out)
        return self.out_proj(context_vec)