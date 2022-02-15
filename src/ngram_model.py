'''Language model to boost performance of the speech recognition model'''

from transformers import AutoProcessor, Wav2Vec2ProcessorWithLM
from datasets import load_dataset
from huggingface_hub import Repository
from pyctcdecode import build_ctcdecoder
from pathlib import Path
import os


def train_ngram_model(model_id: str,
                      dataset_id: str = 'DDSC/reddit-da-asr-preprocessed',
                      n: int = 5):
    '''Trains an ngram language model.

    Args:
        model_id (str):
            The model id of the finetuned speech model, which we will merge
            with the ngram model.
        dataset_id (str, optional):
            The dataset to use for training. Defaults to
            'DDSC/reddit-da-asr-preprocessed'.
        n (int, optional):
            The ngram order to use for training. Defaults to 5.
    '''
    # Load the dataset
    dataset = load_dataset(dataset_id)

    # Ensure that the data folder exists
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir()

    # Dump dataset to a text file
    text_path = data_dir / 'text_data.txt'
    with open(text_path, 'w') as f:
        f.write(' '.join(dataset['text']))

    # Ensure that the `kenlm` directory exists, and download if otherwise
    kenlm_dir = Path('kenlm')
    if not kenlm_dir.exists():
        os.system('wget -O - https://kheafield.com/code/kenlm.tar.gz | tar xz')

    # Compile `kenlm` if it hasn't already been compiled
    kenlm_build_dir = kenlm_dir / 'build'
    if not kenlm_build_dir.exists():
        os.system('mkdir kenlm/build && '
                  'cd kenlm/build && '
                  'cmake .. && '
                  'make -j2')

    # Train the n-gram language model
    ngram_path = data_dir / f'raw_{n}gram.arpa'
    os.system(f'kenlm/build/bin/lmplz -o {n} < "text.txt" > "{ngram_path}"')

    # Add end-of-sentence marker </s> to the n-gram language model
    correct_ngram_path = data_dir / f'{n}gram.arpa'
    with ngram_path.open('r') as f_in:
        with correct_ngram_path.open('w') as f_out:

            # Iterate over the lines in the input file
            has_added_eos = False
            for line in f_in:

                # Increment the 1-gram count by 1
                if not has_added_eos and "ngram 1=" in line:
                    count = line.strip().split("=")[-1]
                    f_out.write(line.replace(f"{count}", f"{int(count)+1}"))

                # Add the end-of-sentence marker right after the the
                # start-of-sentence marker
                elif not has_added_eos and "<s>" in line:
                    f_out.write(line)
                    f_out.write(line.replace("<s>", "</s>"))
                    has_added_eos = True

                # Otherwise we're just copying the line verbatim
                else:
                    f_out.write(line)

    # Load the pretrained processor
    processor = AutoProcessor.from_pretrained(model_id)

    # Extract the vocabulary, which will be used to build the CTC decoder
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = sorted(vocab_dict.items(), key=lambda item: item[1])
    sorted_vocab_dict = {k.lower(): v for k, v in sorted_vocab_dict}

    # Build the CTC decoder
    decoder = build_ctcdecoder(labels=list(sorted_vocab_dict.keys()),
                               kenlm_model_path=str(correct_ngram_path))

    # Build the processor with LM included
    processor_with_lm = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    # Clone the repo containing the finetuned model
    repo = Repository(local_dir=model_id.split('/')[-1], clone_from=model_id)

    # Save the new processor to the repo
    processor_with_lm.save_pretrained(model_id.split('/')[-1])

    # Compress the ngram model
    os.system(f'kenlm/build/bin/build_binary '
              f'{model_id.split("/")[-1]}/language_model/{n}gram.arpa '
              f'{model_id.split("/")[-1]}/language_model/{n}gram.bin')

    # Remove the uncompressed ngram model
    ngram_path.unlink()

    # Push the changes to the repo
    repo.push_to_hub(commit_message="Upload LM-boosted decoder")


if __name__ == '__main__':
    train_ngram_model('saattrupdan/wav2vec2-xls-r-300m-cv8-da')
