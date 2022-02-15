'''Evaluate ASR models on a custom test dataset'''

from transformers import (Wav2Vec2Processor,
                          Wav2Vec2ProcessorWithLM,
                          Wav2Vec2ForCTC,
                          Trainer)
import transformers.utils.logging as tf_logging
from datasets import load_dataset as ds_load_dataset
from datasets import Audio, Dataset
from unicodedata import normalize
from functools import partial
import re
import click
from data_collator import DataCollatorCTCWithPadding
from compute_metrics import compute_metrics


@click.command()
@click.option('--model_id', '-m',
              type=str,
              help='The ID of the model to evaluate')
@click.option('--dataset_id', '-d',
              type=str,
              help='The ID of the dataset to evaluate')
@click.option('--dataset_subset',
              show_default=True,
              default='',
              type=str,
              help='The subset of the dataset to evaluate')
@click.option('--dataset_split',
              show_default=True,
              default='test',
              type=str,
              help='The split of the dataset to evaluate')
@click.option('--sampling_rate',
              show_default=True,
              default=16_000,
              type=int,
              help='The sampling rate of the audio')
@click.option('--use_lm',
              is_flag=True,
              show_default=True,
              help='Whether a language model should be used during decoding.')
def evaluate(model_id: str,
             dataset_id: str,
             dataset_subset: str,
             dataset_split: str,
             sampling_rate: int,
             use_lm: bool):
    '''Evaluate ASR models on a custom test dataset'''
    # Load the dataset
    try:
        subset = None if dataset_subset == '' else dataset_subset
        dataset = ds_load_dataset(dataset_id,
                                  subset,
                                  split=dataset_split,
                                  use_auth_token=True)
    except ValueError:
        dataset = Dataset.from_file(f'{dataset_id}/dataset.arrow')

     # Clean the transcriptions
    dataset = dataset.map(clean_transcription)

    # Resample the audio
    audio = Audio(sampling_rate=sampling_rate)
    dataset = dataset.cast_column('audio', audio)

    # Load the pretrained processor and model
    if use_lm:
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    # Preprocess the dataset
    preprocess_fn = partial(preprocess, processor=processor)
    dataset = dataset.map(preprocess_fn)

    # Initialise data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor,
                                               padding='longest')

    # Initialise the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, processor=processor),
        eval_dataset=dataset,
        tokenizer=processor.tokenizer
    )

    # Disable most of the `transformers` logging
    tf_logging.set_verbosity_error()

    # Remove trainer logging
    trainer.log = lambda _: None

    # Evaluate the model
    metrics = trainer.evaluate(dataset)

    # Print the metrics
    print(f'Scores for {model_id} on {dataset_id}:')
    print(metrics)


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
    evaluate()
