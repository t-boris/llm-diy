import torch
from torch import nn


def casual_attention_mask(keys, attn_scores, block_size):
    """
    :param keys: The tensor of shape (batch_size, num_keys, key_dim) representing the query keys for the attention mechanism.
    :param attn_scores: The attention scores between query keys and context keys. This is a tensor of shape (batch_size, num_queries, num_contexts).
    :param block_size: The size of the attention block. This represents the number of tokens within which the attention is allowed to flow within a sequence.
    :return: The attention weights after applying the causal attention mask. This is a tensor of shape (batch_size, num_queries, num_contexts).
    """
    mask = torch.triu(torch.ones(block_size, block_size), diagonal=1)
    masked = attn_scores.masked_fill(mask == 1, float('-inf'))
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)

    return attn_weights


class CasualAttention(nn.Module):
    """

    Module representing a casual attention mechanism.

    This attention mechanism is useful for sequential data, such as text or time series data.

    Attributes:
        d_out (int): The output dimension of the attention mechanism.
        block_size (int): The block size used for the attention computation, which determines the maximum sequence length.
        W_query (nn.Linear): Linear layer for query transformation.
        W_key (nn.Linear): Linear layer for key transformation.
        W_value (nn.Linear): Linear layer for value transformation.
        dropout (nn.Dropout): Dropout layer.
        mask (torch.Tensor): Buffer for storing a triangular mask matrix that helps with masking future tokens during attention computation.

    Methods:
        forward(x):
            Applies the casual attention mechanism to the input sequence.
            Args:
                x (torch.Tensor): The input sequence of shape (batch_size, num_tokens, d_in).
            Returns:
                context_vec (torch.Tensor): The context vector resulting from the attention computation, of shape (batch_size, num_tokens, d_out).

    """
    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.block_size = block_size
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        """
        :param x: The input tensor of shape (b, nums_tokens, d_in), where b is the batch size, nums_tokens is the number of tokens, and d_in is the input dimension.
        :return: The context vector computed using the input tensor x.

        This method computes the context vector by performing self-attention mechanism on the input tensor x.

        1. Compute keys, queries, and values using the input tensor x.
        2. Calculate attention scores by performing dot product between queries and keys transpose.
        3. Mask the attention scores using the provided mask, setting masked positions to -inf.
        4. Apply softmax on the attention scores divided by the square root of the input dimension.
        5. Apply dropout on the attention weights.
        6. Compute the context vector by multiplying the attention weights with the values.
        7. Return the computed context vector.

        Example usage:
        ```
        x = torch.randn(b, nums_tokens, d_in)
        context_vec = forward(self, x)
        ```
        """
        b, nums_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:nums_tokens, :nums_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return context_vec
