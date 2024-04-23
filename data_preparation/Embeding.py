import torch

output_dim = 256  # The output dimension of the embedding layer
vocab_size = 50257  # The size of the vocabulary
max_length = 4  # The maximum length of the input and target sequences


def get_token_embeddings_layer():
    return torch.nn.Embedding(vocab_size, output_dim)


def get_positional_embeddings():
    block_size = max_length
    pos_embedding_layer = torch.nn.Embedding(block_size, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(block_size))
    return pos_embeddings


def get_input_embeddings():
    token_embedding_layer = get_token_embeddings_layer()
    pos_embedding_layer = get_positional_embeddings()
    return token_embedding_layer + pos_embedding_layer
