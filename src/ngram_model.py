'''Language model to boost performance of the speech recognition model'''

from datasets import load_dataset
from unicodedata import normalize
from pathlib import Path
from functools import partial
import json
import re


def clean_texts(examples: dict, vocab: dict) -> dict:
    '''Clean the texts of the dataset.

    Args:
        examples (dict):
            Examples of the dataset.

    Returns:
        dict:
            Cleaned examples of the dataset.
    '''
    # NFKC normalize the transcriptions
    examples['text'] = normalize('NFKC', examples['text'])

    # Make the text lowercase
    examples['text'] = examples['text'].lower()

    # Remove links
    examples['text'] = re.sub(r'http[^ ]*', '', examples['text'])
    examples['text'] = re.sub(r'\[[Ll][Ii][Nn][Kk]\]', '', examples['text'])

    # Remove all characters that are not in the vocabulary, or are whitespace
    regex = f'[^{re.escape("".join(vocab.keys()))} ]'
    examples['text'] = re.sub(regex, '', examples['text'])

    # Replace multiple spaces with a single space
    examples['text'] = re.sub(r' +', ' ', examples['text'])

    return examples


def train_ngram_model():
    '''Trains an ngram language model'''

    # Load the dataset
    dataset = load_dataset('DDSC/reddit-da', split='train')

    # Load the vocabulary
    with Path('vocab.json').open() as f:
        vocab = json.load(f)
    vocab.pop('<unk>')
    vocab.pop('<pad>')

    # Preprocess the dataset
    dataset = dataset.map(partial(clean_texts, vocab=vocab))

    # Push the preprocessed dataset to the HF Hub
    dataset.push_to_hub('DDSC/reddit-da-asr-preprocessed', split='train')


if __name__ == '__main__':
    train_ngram_model()
