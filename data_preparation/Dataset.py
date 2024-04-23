import torch
from data_preparation.Tokenizer import Tokenizer
from torch.utils.data import Dataset, DataLoader


def create_dataloader(text: str, batch_size=4, max_length=256, stride=128, shuffle=True):
    """
    Create a DataLoader for training or testing a GPT model.

    Args:
        text (str): The input text data.
        batch_size (int): The batch size for the DataLoader.
        max_length (int): The maximum length of each input and target sequence.
        stride (int): The stride for creating overlapping input and target sequences.
        shuffle (bool): Whether to shuffle the data in the DataLoader.

    Returns:
        DataLoader: A DataLoader for training or testing a GPT model.

    """
    tokenizer = Tokenizer()
    dataset = GPTDataSetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

class GPTDataSetV1(Dataset):
    """
    A class representing a dataset for training or testing a GPT model.

    Args:
        text (str): The input text data.
        tokenizer (Tokenizer): The tokenizer used to encode the text.
        max_length (int): The maximum length of each input and target sequence.
        stride (int): The stride for creating overlapping input and target sequences.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used to encode the text.
        input_ids (list): A list of input sequences encoded as token IDs.
        target_ids (list): A list of target sequences encoded as token IDs.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the input and target sequences at the specified index.

    """
    def __init__(self, text: str, tokenizer: Tokenizer, max_length: int, stride: int):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range (0, len(token_ids) - max_length, stride):
            self.input_ids.append(token_ids[i:i+max_length])
            self.target_ids.append(token_ids[i+1:i+max_length+1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]