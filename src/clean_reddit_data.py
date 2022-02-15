'''Script that cleans the Danish Reddit dataset, for use with ASR'''

from datasets import load_dataset
from pathlib import Path
import json
from functools import partial


def clean_and_upload_reddit_data():
    '''Cleans the data and uploads it to the Hugging Face Hub'''
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


def clean_texts(examples: dict, vocab: dict) -> dict:
    '''Clean the texts of the dataset.

    Args:
        examples (dict):
            Examples of the dataset.

    Returns:
        dict:
            Cleaned examples of the dataset.
    '''
    # Remove links
    examples['text'] = re.sub(r'http[^ ]*', '', examples['text'])
    examples['text'] = re.sub(r'\[[Ll][Ii][Nn][Kk]\]', '', examples['text'])

    # NFKC normalize the transcriptions
    examples['text'] = normalize('NFKC', examples['text'])

    # Make the text lowercase
    examples['text'] = examples['text'].lower()

    # Remove all characters that are not in the vocabulary, or are whitespace
    regex = f'[^{re.escape("".join(vocab.keys()))} ]'
    examples['text'] = re.sub(regex, '', examples['text'])

    # Replace multiple spaces with a single space
    examples['text'] = re.sub(r' +', ' ', examples['text'])

    return examples


if __name__ == '__main__':
    clean_and_upload_reddit_data()
