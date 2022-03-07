'''Functions related to the data loading and processing'''

from transformers import (Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor,
                          Wav2Vec2Processor)
from datasets import (load_dataset as ds_load_dataset,
                      Dataset,
                      DatasetDict,
                      Audio)
from unicodedata import normalize
from typing import Optional, Tuple
from pathlib import Path
import json
import re


class AudioDataset:
    '''A dataset containing audio data.

    Args:
        dataset_id (str, optional):
            The HF dataset id. Defaults to
            'mozilla-foundation/common_voice_8_0'.
        dataset_subset (str, optional):
            The HF dataset subset. Defaults to 'da'.
        sampling_rate (int, optional):
            The sampling rate of the audio data. Defaults to 16_000.
        train_name (str, optional):
            The name of the train split. Defaults to 'train'.
        validation_name (str or None, optional):
            The name of the validation split. If None then the validation set
            is created from the train split. Defaults to 'validation'.
        test_name (str or None, optional):
            The name of the test split. If None then the test set is created
            from the validation or train split. Defaults to 'test'.
    '''
    def __init__(self,
                 dataset_id: str = 'mozilla-foundation/common_voice_8_0',
                 dataset_subset: Optional[str] = 'da',
                 sampling_rate: int = 16_000,
                 train_name: str = 'train',
                 validation_name: Optional[str] = 'validation',
                 test_name: Optional[str] = 'test'):

        self.dataset_id = dataset_id
        self.dataset_subset = dataset_subset
        self.sampling_rate = sampling_rate
        self.train_name = train_name
        self.validation_name = validation_name
        self.test_name = test_name

        # Load the dataset
        self.train, self.val, self.test = self._load_dataset()

    def preprocess(self):
        '''Preprocess the dataset'''

        # Clean the transcriptions
        self.train = self.train.map(self._clean_examples,
                                    load_from_cache_file=False)
        self.val = self.val.map(self._clean_examples,
                                load_from_cache_file=False)
        self.test = self.test.map(self._clean_examples,
                                  load_from_cache_file=False)

        # Resample the audio
        audio = Audio(sampling_rate=self.sampling_rate)
        self.train = self.train.cast_column('audio', audio)
        self.val = self.val.cast_column('audio', audio)
        self.test = self.test.cast_column('audio', audio)

        # Extract and dump the vocabulary from the training dataset
        self._dump_vocabulary(self.train)

        # Intitialise the preprocessor
        self.initialise_preprocessor()

        # Preprocess the datasets
        self.train.set_transform(self._preprocess)
        self.val.set_transform(self._preprocess)
        self.test.set_transform(self._preprocess)

        return self

    def initialise_preprocessor(self):
        '''Initialise the preprocessor'''
        # Intialise the tokenizer
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            './',
            unk_token='<unk>',
            pad_token='<pad>',
            bos_token='<s>',
            eos_token='</s>',
            word_delimiter_token='|'
        )

        # Initialise the feature extractor
        self.extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        # Initialise the processor, which wraps the tokenizer and the extractor
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.extractor,
            tokenizer=self.tokenizer
        )

        return self

    @staticmethod
    def _load_dataset_split(dataset_id: str,
                            name: Optional[str] = None,
                            split: str = 'train',
                            use_auth_token: bool = True) -> Dataset:
        '''Load a dataset split.

        Args:
            dataset_id (str):
                The HF dataset id.
            name (str or None, optional):
                The name of the dataset split. If None then the dataset split
                is created from the train split. Defaults to None.
            split (str, optional):
                The HF dataset split. Defaults to 'train'.
            use_auth_token (bool, optional):
                Whether to use the auth token. Defaults to True.

        Returns:
            Dataset:
                The loaded dataset split.
        '''
        try:
            return ds_load_dataset(path=dataset_id,
                                   name=name,
                                   split=split,
                                   use_auth_token=use_auth_token,
                                   download_mode='force_redownload')
        except ValueError:
            return DatasetDict.load_from_disk(dataset_id)[split].select(range(100))

    def _load_dataset(self) -> Tuple[Dataset, Dataset, Dataset]:
        '''Loads a dataset.

        Returns:
            tuple:
                A triple (train, val, test), containing the three splits of the
                dataset.
        '''
        # Load train dataset
        train = self._load_dataset_split(dataset_id=self.dataset_id,
                                         name=self.dataset_subset,
                                         split=self.train_name)

        # Load validation and test datasets. If both `validation_name` and
        # `test_name` are not None then these are simply loaded. If only
        # `test_name` is not None then a validation set is created from the
        # train dataset.
        if self.test_name is not None:
            test = self._load_dataset_split(dataset_id=self.dataset_id,
                                            name=self.dataset_subset,
                                            split=self.test_name)
            if self.validation_name is not None:
                val = self._load_dataset_split(dataset_id=self.dataset_id,
                                               name=self.dataset_subset,
                                               split=self.validation_name)
            else:
                split_dict = train.train_test_split(test_size=0.1, seed=4242)
                train = split_dict['train']
                val = split_dict['test']

        # If only `validation_name` is not None then the validation set is used
        # as a test set and a new validation set is created from the train
        # dataset.
        elif self.validation_name is not None:
            test = self._load_dataset_split(dataset_id=self.dataset_id,
                                            name=self.dataset_subset,
                                            split=self.validation_name)
            split_dict = train.train_test_split(test_size=0.1, seed=4242)
            train = split_dict['train']
            val = split_dict['test']

        # If both `validation_name` and `test_name` are None then validation
        # and test sets are created from the train dataset.
        else:
            # Split train dataset into train and a combined validation and test
            # set
            split_dict = train.train_test_split(test_size=0.2, seed=4242)
            train = split_dict['train']
            val_test = split_dict['test']

            # Create validation set from the combined validation and test set
            split_dict = val_test.train_test_split(test_size=0.5, seed=4242)
            val = split_dict['train']
            test = split_dict['test']

        return train, val, test

    @staticmethod
    def _clean_examples(examples: dict) -> dict:
        '''Cleans the transcription of an example.

        Args:
            examples (dict):
                A dictionary containing the examples to preprocess.

        Returns:
            dict:
                A dictionary containing the cleaned transcription.
        '''
        # Clean the transcription
        examples['sentence'] = clean_transcription(examples['sentence'])

        return examples

    def _preprocess(self, examples: dict) -> dict:
        '''Preprocess the audio of an example.

        Args:
            examples (dict):
                A dictionary containing the examples to preprocess.

        Returns:
            dict:
                A dictionary containing the preprocessed examples.
        '''
        # Get the dictionary from the examples containing the audio data
        audio_arrays = [audio['array'] for audio in examples['audio']]

        # Get the sampling rate
        sampling_rate = examples['audio'][0]['sampling_rate']

        # Preprocess the audio
        examples['input_values'] = (
            self.processor(audio_arrays, sampling_rate=sampling_rate)
                .input_values
        )

        # Preprocess labels
        with self.processor.as_target_processor():
            examples["labels"] = self.processor(examples["sentence"]).input_ids

        # Return the preprocessed examples
        return examples

    @staticmethod
    def _dump_vocabulary(dataset: Dataset):
        '''Extracts the vocabulary from the dataset and dumps it to a file.

        Args:
            dataset (Dataset):
                The dataset from which to extract the vocabulary. Needs to
                contain a feature named 'sentence'.
        '''
        # Get all the text in the transcriptions
        all_text = '|'.join(dataset['sentence'])

        # Get the unique characters in the text
        unique_characters = set(all_text)

        # Form the vocabulary dictionary
        vocab = {char: idx for idx, char in enumerate(unique_characters)}

        # Manually add special tokens
        vocab['<unk>'] = len(vocab)
        vocab['<pad>'] = len(vocab)
        vocab['<s>'] = len(vocab)
        vocab['</s>'] = len(vocab)

        # Dump the vocabulary to a json file
        with Path('vocab.json').open('w') as f:
            json.dump(vocab, f)


def clean_transcription(doc: str) -> str:
    '''Cleans the transcription of a document.

    Args:
        doc (str):
            A document to be cleaned.

    Returns:
        str:
            The cleaned document.
    '''
    # NFKC normalize the transcriptions
    doc = normalize('NFKC', doc)

    # Remove punctuation
    regex = r'[\[\]\{\}\(\)\,\?\.\!\-\—\–\;\:\"\“\'\’\%\”\�\•\n\r\⁄\’]'
    doc = re.sub(regex, '', doc)

    # Remove non-vocabulary characters
    conversion_dict = {
        'aa': 'å',
        'ğ': 'g',
        'ñ': 'n',
        'ń': 'n',
        'è': 'e',
        'μ': 'm',
        '§': ' paragraf ',
        '‰': ' promille ',
        'ú': 'u',
        'ş': 's',
        'ê': 'e',
        'ã': 'a',
        'ü': 'ue',
        'ë': 'e',
        'ć': 'c',
        'ä': 'æ',
        'í': 'i',
        'š': 's',
        'î': 'i',
        'ě': 'e',
        'ð': 'd',
        'á': 'a',
        'ó': 'o',
        'þ': 'th',
        'ı': 'i',
        'ö': 'ø',
        'ç': 'c',
        'ș': 's',
        '0': ' nul ',
        '1': ' et ',
        '2': ' to ',
        '3': ' tre ',
        '4': ' fire ',
        '5': ' fem ',
        '6': ' seks ',
        '7': ' syv ',
        '8': ' otte ',
        '9': ' ni ',
    }
    for key, value in conversion_dict.items():
        doc = doc.replace(key, value)

    # Remove empty whitespace
    doc = re.sub(u'\u0301', ' ', doc)
    doc = re.sub(u'\u200b', ' ', doc)

    # Replace spaces with a pipe, to emphasise the word boundaries
    doc = re.sub(r' +', '|', doc)

    # Make the transcription lowercase and strip whitespace
    doc = doc.lower().strip().strip('|')

    return doc
