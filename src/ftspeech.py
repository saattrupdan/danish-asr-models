'''Class used to load in the FTSpeech dataset'''

from datasets import Dataset, Audio
from pathlib import Path
from typing import Union
import tarfile


def build_and_store_data(input_path: Union[Path, str] = 'data/ftspeech.tar.gz',
                         output_path: Union[Path, str] = 'data/ftspeech'):
    '''Loads the Alvenir dataset.

    Args:
        input_path (str or Path, optional):
            The directory where the dataset is stored. Defaults to
            'data/ftspeech.tar.gz'.
        output_path (str or Path, optional):
            The name of the dataset. Defaults to 'data/ftspeech'.

    Raises:
        FileNotFoundError:
            If `input_path` does not exist.
    '''
    # Ensure that the paths are Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Create directory to store the uncompressed raw dataset
    raw_data_dir = input_path.parent / 'ftspeech_raw'

    # If `input_path` does not exist, raise an error
    if not input_path.exists():
        raise FileNotFoundError(f'{input_path} does not exist')

    # Uncompress the dataset to `raw_data_dir`
    with tarfile.open(input_path) as tar:
        tar.extractall(raw_data_dir)

    # Load file with transcriptions
    # TODO

    # Convert the dataframe to a HuggingFace Dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)

    # Cast `path` as the audio path column
    dataset = dataset.cast_column('audio', Audio())

    # Store the dataset
    dataset.save_to_disk(output_name)


if __name__ == '__main__':
    build_and_store_data()
