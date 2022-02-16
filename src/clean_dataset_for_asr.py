'''Script that cleans the datasets for use with ASR'''

from datasets import load_dataset, Dataset
from pathlib import Path
import json
from functools import partial
from unicodedata import normalize
import re


def clean_and_upload_dataset(input_dataset: str = 'DDSC/reddit-da',
                             output_prefix: str = '-preprocessed',
                             upload: bool = True):
    '''Cleans the data and uploads it to the Hugging Face Hub.

    Args:
        input_dataset (str, optional):
            The Hugging Face ID of the dataset to clean. Defaults to
            'DDSC/reddit-da'.
        output_prefix (str, optional):
            The prefix of the output dataset. Defaults to '-preprocessed'.
        upload (bool, optional):
            Whether to upload the dataset to the Hugging Face Hub. If not, the
            dataset will instead be saved to disk. Defaults to True.
    '''
    # Load the dataset
    try:
        dataset = load_dataset(input_dataset, split='train')
    except FileNotFoundError:
        dataset = Dataset.from_json(input_dataset)

    # Load the vocabulary
    with Path('vocab.json').open() as f:
        vocab = json.load(f)
    vocab.pop('<unk>')
    vocab.pop('<pad>')

    # Preprocess the dataset
    dataset = dataset.map(partial(clean_texts, vocab=vocab))

    # Determine output name based on whether `input_dataset` is the path to a
    # local dataset or a HF Hub ID
    if '.' in input_dataset:
        output_name = '.'.join(input_dataset.split('.')[:-1]) + output_prefix
    else:
        output_name = input_dataset + output_prefix

    # Push the preprocessed dataset to the HF Hub
    if upload:
        dataset.push_to_hub(output_name, split='train')
    else:
        dataset.save_to_disk(output_name)


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
    examples['text'] = re.sub(r'www\.[^ ]*', '', examples['text'])
    examples['text'] = re.sub(r'\[[Ll][Ii][Nn][Kk]\]', '', examples['text'])

    # NFKC normalize the transcriptions
    examples['text'] = normalize('NFKC', examples['text'])

    # Make the text lowercase
    examples['text'] = examples['text'].lower()

    # Remove all characters that are not in the vocabulary, or are whitespace
    regex = f'[^{re.escape("".join(vocab.keys()))} \n]'
    examples['text'] = re.sub(regex, '', examples['text'])

    # Replace multiple spaces with a single space
    examples['text'] = re.sub(r' +', ' ', examples['text'])

    return examples


if __name__ == '__main__':
    clean_and_upload_dataset('data/lexdk.jsonl', upload=False)
