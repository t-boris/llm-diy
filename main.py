import importlib

from data_preparation.Dataset import create_dataloader
from data_preparation.Tokenizer import *

if __name__ == '__main__':
    vocab: dict[str, int]

    with open("data/The_Verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

        dataloader = create_dataloader(raw_text, batch_size=1, max_length=2, stride=2, shuffle=False)

        data_iter = iter(dataloader)
        first_batch = next(data_iter)
        print(first_batch)
        second_batch = next(data_iter)
        print(second_batch)





