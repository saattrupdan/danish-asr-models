'''Evaluate ASR models on a custom test dataset'''

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset as ds_load_dataset
from datasets import Audio, Dataset
from typing import Optional, Dict
from unicodedata import normalize
from functools import partial
import re


def evaluate(model_id: str,
             dataset_id: str,
             dataset_subset: Optional[str] = None,
             dataset_split: Optional[str] = None,
             sampling_rate: int = 16_000) -> Dict[str, float]:
    '''Evaluate ASR models on a custom test dataset.

    Args:
        dataset_id (str):
            The ID of the dataset to evaluate.
        dataset_subset (str or None, optional):
            The subset of the dataset to evaluate. If None then no subset will
            be used. Defaults to None.
        dataset_split (str or None, optional):
            The split of the dataset to evaluate. If None then no split will be
            used. Defaults to None.
        sampling_rate (int, optional):
            The sampling rate of the dataset. Defaults to 16_000.

    Returns:
        dict:
            A dictionary with the metric names as keys and the metric values as
            values.
    '''
    # Load the dataset
    try:
        dataset = ds_load_dataset(dataset_id,
                                  dataset_subset,
                                  split=dataset_split)
    except ValueError:
        dataset = Dataset.from_file(f'{dataset_id}/dataset.arrow')

    # Clean the transcriptions
    dataset = dataset.map(clean_transcription)

    # Resample the audio
    audio = Audio(sampling_rate=sampling_rate)
    dataset = dataset.cast_column('audio', audio)

    # Load the pretrained processor and model
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # Preprocess the dataset
    preprocess_fn = partial(preprocess, processor=processor)
    dataset = dataset.map(preprocess_fn)

    # Evaluate the model
    metrics = model.evaluate(dataset)

    # Return the metrics
    return metrics


def clean_transcription(examples: dict) -> dict:
    '''Cleans the transcription of an example.

    Args:
        examples (dict):
            A dictionary containing the examples to preprocess.

    Returns:
        dict:
            A dictionary containing the cleaned transcription.
    '''
    # NFKC normalize the transcriptions
    examples['sentence'] = normalize('NFKC', examples['sentence'])

    # Remove punctuation
    regex = r'[\[\]\{\}\(\)\,\?\.\!\-\—\–\;\:\"\“\%\”\�]'
    examples['sentence'] = re.sub(regex, '', examples['sentence'])

    # Replace spaces with a pipe, to emphasise the word boundaries
    examples['sentence'] = re.sub(r' +', '|', examples['sentence'])

    # Make the transcription lowercase
    examples['sentence'] = examples['sentence'].lower()

    return examples


def preprocess(examples: dict,
               processor: Wav2Vec2Processor) -> dict:
    '''Preprocess the audio of an example.

    Args:
        examples (dict):
            A dictionary containing the examples to preprocess.
        processor (Wav2Vec2Processor):
            The processor to use.

    Returns:
        dict:
            A dictionary containing the preprocessed examples.
    '''
    # Get the dictionary from the examples containing the audio data
    audio = examples['audio']

    # Preprocess the audio
    examples['input_values'] = (
        processor(audio['array'],
                       sampling_rate=audio['sampling_rate'])
            .input_values[0]
    )
    examples['input_length'] = len(examples['input_values'])

    # Preprocess labels
    examples['labels'] = processor.tokenizer.encode(list(examples['sentence']))

    # Return the preprocessed examples
    return examples


if __name__ == '__main__':
    scores = evaluate(model_id='saattrupdan/wav2vec2-xls-r-300m-cv8-da',
                      dataset_id='data/alvenir-asr-test-set')
    print(scores)
