'''Language model to boost performance of the speech recognition model'''

from datasets import load_dataset
from unicodedata import normalize
from pathlib import Path
import json
import re


def clean_texts(examples: dict) -> dict:
    '''Clean the texts of the dataset.

    Args:
        examples (dict):
            Examples of the dataset.

    Returns:
        dict:
            Cleaned examples of the dataset.
    '''
    # Import vocabulary
    with Path('vocab.json').open() as f:
        vocab = list(json.load(f).keys())

    # NFKC normalize the transcriptions
    examples['text'] = normalize('NFKC', examples['text'])

    # Make the text lowercase
    examples['text'] = examples['text'].lower()

    # Remove all characters that are not in the vocabulary, or are whitespace
    regex = f'[^{re.escape("".join(vocab))} ]'
    examples['text'] = re.sub(regex, '', examples['text'])

    # Replace multiple spaces with a single space
    examples['text'] = re.sub(r' +', ' ', examples['text'])

    return examples


def train_ngram_model():
    '''Trains an ngram language model'''

    # Load the dataset
    dataset = load_dataset('DDSC/reddit-da', split='train')

    # Preprocess the dataset
    dataset = dataset.map(clean_texts)

    # Push the preprocessed dataset to the HF Hub
    dataset.push_to_hub('DDSC/reddit-da-asr-preprocessed')


if __name__ == '__main__':
    train_ngram_model()
