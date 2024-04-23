from torch import nn
from torch.nn import LayerNorm

from attention.MultiHeadAttention import MultiHeadAttention
from gpt.FeedForward import FeedForward

"""
    The given code defines a TransformerBlock class in PyTorch that includes a multi-head attention mechanism 
    (MultiHeadAttention) and a feed forward network (FeedForward), both configured based on a provided configuration 
    dictionary (cfg), such as GPT_CONFIG_124M. 
   
    Layer normalization (LayerNorm) is applied before each of these two components, and dropout is applied after them 
    to regularize the model and prevent overfitting. This is also known as Pre-LayerNorm. Older architectures, such as 
    the original transformer model, applied layer normalization after the self-attention and feed-forward networks 
    instead, known as Post-LayerNorm, which often leads to worse training dynamics. 

    The class also implements the forward pass, where each component is followed by a shortcut connection that adds 
    the input of the block to its output. This critical feature helps gradients flow through the network during 
    training and improves the learning of deep models
"""


class TransformerBlock(nn.Module):
    """
    TransformerBlock Class

    This class represents a single block in a Transformer model.

    Attributes:
        att (MultiHeadAttention): The multi-head attention layer.
        ff (FeedForward): The feed-forward neural network layer.
        norm1 (LayerNorm): The layer normalization layer.
        norm2 (LayerNorm): The layer normalization layer.
        drop_redid (Dropout): The dropout layer.

    Methods:
        forward(x): Performs forward pass through the Transformer block.

    """
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            block_size=cfg["ctx_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self. ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_redid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_redid(x)
        x = shortcut + x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_redid(x)
        x = shortcut + x
        return x


