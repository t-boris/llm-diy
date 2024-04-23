import re
import tiktoken


def tokenize_text(raw_text: str) -> list[str]:
    """
    Tokenizes the given raw text into a list of strings.

    :param raw_text: The raw text to be tokenized.
    :return: The tokenized list of strings.
    """
    preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    return preprocessed


def create_vocabulary(tokens: list[str]) -> dict[str, int]:
    """
    Create Vocabulary

    This method creates a vocabulary from a list of tokens by assigning each unique token a unique integer index.

    :param tokens: A list of tokens to create vocabulary from.
    :type tokens: list[str]
    :return: A dictionary that maps each unique token to its corresponding integer index.
    :rtype: dict[str, int]
    """
    vocabulary = {}
    all_tokens = sorted(list(set(tokens)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    for token in all_tokens:
        vocabulary[token] = len(vocabulary)
    return vocabulary


class SimpleTokenizerV1:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Encode the given text into a list of integers.

        :param text: The text to be encoded.
        :type text: str
        :return: A list of integers representing the encoded text.
        :rtype: list[int]
        """
        return [self.str_to_int[token] for token in tokenize_text(text)]

    def decode(self, ids: list[int]) -> str:
        """
        Decode the given list of integers into a text.
        :param ids: The list of integers to be decoded.
        :type ids: list[int]
        :return: The decoded text.
        :rtype: str
        """
        punctuation_pattern = r'([,.?_!"()\']|--|\s)'
        replacement = r'\1'

        text = " ".join(self.int_to_str[id] for id in ids)
        text = re.sub(punctuation_pattern, replacement, text)

        return text


class SimpleTokenizerV2:
    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """
        Encode the given text into a list of integers.

        :param text: The text to be encoded.
        :type text: str
        :return: A list of integers representing the encoded text.
        :rtype: list[int]
        """
        preprocessed = tokenize_text(text)
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """
        Decode the given list of integers into a text.
        :param ids: The list of integers to be decoded.
        :type ids: list[int]
        :return: The decoded text.
        :rtype: str
        """
        punctuation_pattern = r'([,.?_!"()\']|--|\s)'
        replacement = r'\1'

        text = " ".join(self.int_to_str[id] for id in ids)
        text = re.sub(punctuation_pattern, replacement, text)

        return text


class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
