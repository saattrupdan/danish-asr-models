'''Class used to load in the Alvenir test dataset'''

from datasets import Dataset, DatasetDict, Audio
import pandas as pd
from pathlib import Path
from typing import Union


def load_data(
        data_dir: Union[Path, str] = 'data/alvenir-test-set'
    ) -> DatasetDict:
    '''Loads the Alvenir dataset.

    Args:
        data_dir (str or Path, optional):
            The directory where the dataset is stored. Defaults to
            'data/alvenir-test-set'.

    Returns:
        DatasetDict:
            The loaded dataset, with 'train', 'validation' and 'test' as keys.

    Raises:
        FileNotFoundError:
            If the dataset directory does not exist.
    '''
    # Ensure that `data_dir` is a Path object
    data_dir = Path(data_dir)

    # If `data_dir` does not exist, raise an error
    if not data_dir.exists():
        raise FileNotFoundError(f'{data_dir} does not exist')

    # Load the metadata files
    metadata_1 = pd.read_csv(data_dir / '20211116' / 'Metadata.csv',
                             header=0)
    metadata_2 = pd.read_csv(data_dir / '20211125' / 'Metadata.csv',
                             header=0)
    metadata_1['supfolder'] = '20211116'
    metadata_2['supfolder'] = '20211125'
    metadata = pd.concat([metadata_1, metadata_2]).reset_index(drop=True)

    # Set up a `path` column and remove the `Folder`, `Subfolder` and `File
    # Name` columns
    def create_path(row: dict) -> str:
        folder = data_dir / row['supfolder'] / row['Folder'] / row['Subfolder']
        path = folder / row['File Name']
        return str(path.resolve())
    metadata['audio'] = [create_path(dct)
                         for dct in metadata.to_dict('records')]
    metadata.drop(columns=['supfolder', 'Folder', 'Subfolder', 'File Name'],
                  inplace=True)

    # Change the other column names
    renaming_dict = {
        'Speaker ID': 'speaker_id',
        'Gender': 'gender',
        'Age': 'age',
        'Age Range': 'age_range',
        'Corpus Code': 'corpus_code',
        'Prompt': 'text',
        'QA Result': 'qa_result'
    }
    metadata.rename(columns=renaming_dict, inplace=True)

    # Do stratified splits into train/val/test, based on the age
    train = (metadata.groupby('age', group_keys=False)
                     .apply(lambda x: x.sample(frac=0.8)))
    val_test = metadata.loc[[idx for idx in metadata.index
                             if idx not in train.index]]
    val = (val_test.groupby('age', group_keys=False)
                   .apply(lambda x: x.sample(frac=0.5)))
    test = val_test.loc[[idx for idx in val_test.index
                         if idx not in val.index]]

    # Convert the dataframe to a HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train, preserve_index=False)
    val_dataset = Dataset.from_pandas(val, preserve_index=False)
    test_dataset = Dataset.from_pandas(test, preserve_index=False)

    # Cast `path` as the audio path column
    train_dataset = train_dataset.cast_column('audio', Audio())
    val_dataset = val_dataset.cast_column('audio', Audio())
    test_dataset = test_dataset.cast_column('audio', Audio())

    # Collect the datasets in a DatasetDict
    dataset = DatasetDict(train=train_dataset,
                          validation=val_dataset,
                          test=test_dataset)

    return dataset


if __name__ == '__main__':
    dataset = load_data()
    breakpoint()
