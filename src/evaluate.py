'''Evaluate ASR models on a custom test dataset'''

from transformers import (Wav2Vec2Processor,
                          Wav2Vec2ProcessorWithLM,
                          Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer)
import transformers.utils.logging as tf_logging
from datasets import load_dataset as ds_load_dataset
from datasets import Audio, Dataset
from functools import partial
import click
from data_collator import DataCollatorCTCWithPadding
from compute_metrics import compute_metrics
from data import clean_transcription


@click.command()
@click.option('--model_id', '-m',
              type=str,
              help='The ID of the model to evaluate')
@click.option('--dataset_id', '-d',
              type=str,
              help='The ID of the dataset to evaluate')
@click.option('--subset',
              show_default=True,
              default='',
              type=str,
              help='The subset of the dataset to evaluate')
@click.option('--split',
              show_default=True,
              default='test',
              type=str,
              help='The split of the dataset to evaluate')
@click.option('--sampling_rate',
              show_default=True,
              default=16_000,
              type=int,
              help='The sampling rate of the audio')
@click.option('--no_lm',
              is_flag=True,
              default=False,
              show_default=True,
              help='Whether no language model should be used during decoding')
def evaluate(model_id: str,
             dataset_id: str,
             subset: str,
             split: str,
             sampling_rate: int,
             no_lm: bool):
    '''Evaluate ASR models on a custom test dataset'''
    # Load the dataset
    try:
        dataset = ds_load_dataset(dataset_id,
                                  None if subset == '' else subset,
                                  split=split,
                                  use_auth_token=True)
    except ValueError:
        dataset = Dataset.from_file(f'{dataset_id}/dataset.arrow')

    # Load the pretrained processor and model
    if no_lm:
        processor = Wav2Vec2Processor.from_pretrained(
            model_id,
            use_auth_token=True
        )
    else:
        try:
            processor = Wav2Vec2ProcessorWithLM.from_pretrained(
                model_id,
                use_auth_token=True
            )
        except (FileNotFoundError, ValueError):
            processor = Wav2Vec2Processor.from_pretrained(
                model_id,
                use_auth_token=True
            )
    model = Wav2Vec2ForCTC.from_pretrained(model_id, use_auth_token=True)

     # Clean and tokenize the transcriptions
    preprocess_fn = partial(preprocess_transcriptions, processor=processor)
    dataset = dataset.map(preprocess_fn)

    # Resample the audio
    audio = Audio(sampling_rate=sampling_rate)
    dataset = dataset.cast_column('audio', audio)

    # Initialise data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor,
                                               padding='longest')

    # Initialise the trainer
    trainer = Trainer(
        args=TrainingArguments('.', remove_unused_columns=False),
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

    # Extract the WER
    wer = 100 * metrics['eval_wer']

    # Print the metrics
    print(f'\n*** RESULTS ON {dataset_id.split("/")[-1].upper()} ***')
    print(f'{model_id} achieved a WER of {wer:.2f}.\n')


def preprocess_transcriptions(examples: dict,
                              processor: Wav2Vec2Processor) -> dict:
    '''Clean and Tokenize the transcriptions of an example.

    Args:
        examples (dict):
            A dictionary containing the examples to preprocess.
        processor (Wav2Vec2Processor):
            The processor to use.

    Returns:
        dict:
            A dictionary containing the preprocessed examples.
    '''
    # Clean the transcription
    examples['sentence'] = clean_transcription(examples['sentence'])

    # Preprocess labels
    examples['labels'] = processor.tokenizer.encode(list(examples['sentence']))

    # Return the preprocessed examples
    return examples


if __name__ == '__main__':
    evaluate()
