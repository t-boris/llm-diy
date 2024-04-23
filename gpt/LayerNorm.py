import torch
from torch import nn


class LayerNorm(nn.Module):
    """

    :class: LayerNorm

    This class implements the Layer Normalization operation for neural networks.

    :param emb_dim: The size of the input dimension (int).
    :type emb_dim: int

    Attributes:
        - eps (float): A small value added to the denominator for numerical stability.
        - scale (nn.Parameter): A learnable parameter for scaling the normalized input.
        - shift (nn.Parameter): A learnable parameter for shifting the normalized input.

    Methods:
        - forward(x): Takes an input tensor x and applies Layer Normalization to it. Returns the normalized tensor.

    Example usage:
    ```python
    emb_dim = 256
    layer_norm = LayerNorm(emb_dim)
    input_tensor = torch.randn(10, emb_dim)
    output_tensor = layer_norm.forward(input_tensor)
    ```
    """
    def __init__(self, emb_dim):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        :param x: Input tensor.
        :return: Tensor after performing the forward pass through normalization layers.

        """
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift