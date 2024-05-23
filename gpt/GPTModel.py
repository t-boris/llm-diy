import torch
from torch import nn

from gpt.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        """
        __init__ constructor of this GPTModel class initializes the token and positional embedding layers using the
        configurations passed in via a Python dictionary, cfg. These embedding layers are responsible for converting
        input token indices into dense vectors and adding positional information
        Next, it creates a sequential stack of TransformerBlock modules equal to the number of layers specified in cfg.
        Following the transformer blocks, a LayerNorm layer is applied, standardizing the outputs from the transformer
        blocks to stabilize the learning process. Finally, a linear output head without bias is defined, which projects
        the transformer's output into the vocabulary space of the tokenizer to generate logits for each token in the
        vocabulary.

        :param cfg (dict): A dictionary containing the configuration parameters for the model.
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["ctx_len"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        The forward method takes a batch of input token indices, computes their embeddings, applies the positional
        embeddings, passes the sequence through the transformer blocks, normalizes the final output, and then computes
        the logits for each token in the vocabulary.

        :param in_idx: A tensor of shape (batch_size, seq_len) containing input token indices.
        :return: A tensor of shape (batch_size, seq_len, vocab_size) containing the logits for each token in the
        vocabulary.
        """
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

