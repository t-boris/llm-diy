from datasets import load_dataset


def download_and_save_dataset(dataset_name, split_type, output_file):
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split_type)

    # Open a file to write
    with open(output_file, 'w', encoding='utf-8') as file:
        # Iterate over items in the dataset
        for item in dataset:
            # Write text to the file, each text entry on a new line
            file.write(item['text'] + '\n')


if __name__ == '__main__':
    # Specify the dataset, split type, and output file
    DATASET_NAME = 'bookcorpus'  # Adjust as needed if this identifier changes
    SPLIT_TYPE = 'train'  # Could be 'train', 'test', or 'validation'
    OUTPUT_FILE = 'data/bookcorpus.txt'  # Path to your output file

    # Download the dataset and save it to the file
    download_and_save_dataset(DATASET_NAME, SPLIT_TYPE, OUTPUT_FILE)
