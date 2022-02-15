'''Class used to load in the Alvenir test dataset'''

from datasets import Dataset, Audio
import pandas as pd
from pathlib import Path
from typing import Union


def build_and_store_data(
    data_dir: Union[Path, str] = 'data/alvenir-test-set',
    output_name: str = 'data/alvenir-asr-test-set'):
    '''Loads the Alvenir dataset.

    Args:
        data_dir (str or Path, optional):
            The directory where the dataset is stored. Defaults to
            'data/alvenir-test-set'.
        output_name (str, optional):
            The name of the dataset. Defaults to 'data/alvenir-asr-test-set'.

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
        'Prompt': 'sentence',
        'QA Result': 'qa_result'
    }
    metadata.rename(columns=renaming_dict, inplace=True)

    # Convert the dataframe to a HuggingFace Dataset
    dataset = Dataset.from_pandas(metadata, preserve_index=False)

    # Cast `path` as the audio path column
    dataset = dataset.cast_column('audio', Audio())

    # Store the dataset
    dataset.save_to_disk(output_name)


if __name__ == '__main__':
    build_and_store_data()
