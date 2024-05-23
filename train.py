import time

import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.nn import DataParallel
from torch.multiprocessing import Pool, set_start_method

from data_preparation.Tokenizer import Tokenizer
from gpt.GPTModel import GPTModel


def load_and_tokenize_data(tokenizer, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        tokenized_data = f.read().split()
    print(f"Loaded data with {len(tokenized_data)} characters.")
    encoded_data = [int(token_id) for token_id in tokenized_data]
    print(f"Encoded data with {len(encoded_data)} tokens.")
    return encoded_data


def create_sequences(encoded_data, seq_length):
    sequences = [encoded_data[i:i+seq_length] for i in range(0, len(encoded_data), seq_length)]
    if len(sequences[-1]) < seq_length:
        sequences[-1].extend([0] * (seq_length - len(sequences[-1])))
    return sequences


def train_model():
    # Config the model
    GPT_CONFIG_124M = {
        "n_layers": 12,
        "n_heads": 12,
        "emb_dim": 768,
        "ctx_len": 1024,
        "vocab_size": 50257,
        "drop_rate": 0.1,
        "qkv_bias": True
    }

    tokenizer = Tokenizer()

    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    with Pool() as pool:
        encoded_data = pool.apply(load_and_tokenize_data, (tokenizer, "data/tokenized_bookcorpus.txt"))

    print(f"Encoded data with {len(encoded_data)} tokens.")

    # Create sequences of fixed length
    seq_length = GPT_CONFIG_124M["ctx_len"]
    with Pool() as pool:
        sequences = pool.apply(create_sequences, (encoded_data, seq_length))

    print(f"Created {len(sequences)} sequences of length {seq_length}.")

    # Convert sequences to tensors
    input_ids = torch.tensor(sequences[:-1])
    labels = torch.tensor(sequences[1:])
    print(f"Created input and label tensors of shape {input_ids.shape} and {labels.shape}.")

    dataset = TensorDataset(input_ids, labels)
    print(f"Created dataset with {len(dataset)} samples.")

    # Split data to train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(f"Created train and test loaders with {len(train_loader)} and {len(test_loader)} batches.")

    # Init GPT Model
    model = GPTModel(GPT_CONFIG_124M)

    # Wrap the model with DataParallel
    if torch.cuda.is_available():
        model = DataParallel(model).cuda()
    else:
        model = DataParallel(model)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train the model
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")
        model.train()
        train_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            print(f"Batch: {batch_idx}")
            if torch.cuda.is_available():
                data, targets = data.cuda(), targets.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, GPT_CONFIG_124M["vocab_size"]), targets.view(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Average Train Loss: {avg_train_loss:.4f}")

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, targets in test_loader:
                if torch.cuda.is_available():
                    data, targets = data.cuda(), targets.cuda()
                output = model(data)
                loss = criterion(output.view(-1, GPT_CONFIG_124M["vocab_size"]), targets.view(-1))
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch: {epoch + 1}/{num_epochs}, Test Loss: {avg_test_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "gpt_model.pth")


if __name__ == '__main__':
    start = time.time()
    train_model()
    print(f"Training completed. Time taken: {time.time() - start:.2f} seconds.")