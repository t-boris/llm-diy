import re


class TextPreprocessor:
    """
    Initialize TextPreprocessor object.
    """
    def __init__(self):
        pass

    def clean_text(self, text):
        """
        Clean the given text by removing URLs, special characters, and extra whitespace.

        :param text: The text to be cleaned.
        :return: The cleaned text.
        """
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_text(self, text):
        """
        Normalize the given text by converting it to lowercase.

        :param text: The text to be normalized.
        :return: The normalized text.
        """
        return text.lower()

    def preprocess_line(self, text):
        """Apply all preprocessing steps to a single line of text."""
        text = self.clean_text(text)
        text = self.normalize_text(text)
        return text

    def process_large_file(self, file_path, output_path):
        """Process a large text file line by line and write output to another file."""
        with open(file_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                processed_line = self.preprocess_line(line)
                outfile.write(processed_line + '\n')


# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    input_path = 'data/bookcorpus.txt'  # Adjust to your file path
    output_path = 'data/processed_bookcorpus.txt'  # Output file path
    preprocessor.process_large_file(input_path, output_path)
