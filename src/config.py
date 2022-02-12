'''Config class that carries all the hyperparameters needed for training'''

from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    '''Config class that carries all the hyperparameters needed for training.

    Args:
        dataset_id (str, optional):
            The id of the dataset to finetune on. Defaults to
            'mozilla-foundation/common_voice_8_0'.
        dataset_subset (str or None, optional):
            The subset of the dataset to finetune on. If None then no subset
            will be used. Defaults to 'da'.
        sampling_rate (int, optional):
            The sample rate of the audio files. Defaults to 16_000.
        train_name (str, optional):
            The name of the train dataset. Defaults to 'train'.
        validation_name (str or None, optional):
            The name of the validation dataset. If None then a validation set
            will be created from the train dataset. Defaults to 'validation'.
        test_name (str or None, optional):
            The name of the test dataset. If None then the validation set will
            be used as a test set and a new validation set will be created from
            the train dataset. If a validation set is not available either,
            then both a validation and test set will be created from the train
            dataset. Defaults to 'test'.
        attention_dropout (float, optional):
            The dropout rate for the attention layer. Defaults to 0.0.
        hidden_dropout (float, optional):
            The dropout rate for the hidden layer. Defaults to 0.0.
        feat_proj_dropout (float, optional):
            The dropout rate for the feature projection layer. Defaults to 0.0.
        mask_time_prob (float, optional):
            The probability of masking the time dimension. Defaults to 0.05.
        layerdrop (float, optional):
            The dropout rate for the layers. Defaults to 0.0.
        ctc_loss_reduction (str, optional):
            The reduction to use for the CTC loss. Defaults to 'sum'.
        freeze_feature_encoder (bool, optional):
            Whether to freeze the feature encoder. Defaults to False.
        batch_size (int, optional):
            The batch size for training. Defaults to 16.
        gradient_accumulation_steps (int, optional):
            The number of steps to accumulate gradients for. Defaults to 2.
        epochs (int, optional):
            The number of epochs to train for. Defaults to 100.
        learning_rate (float, optional):
            The learning rate for the optimizer. Defaults to 3e-4.
        warmup_steps (int, optional):
            The number of warmup steps for the learning rate scheduler.
            Defaults to 100.
        fp16 (bool, optional):
            Whether to use 16-bit floating point precision. Note that this is
            only possible if GPU is enabled. Defaults to True.
        push_to_hub (bool, optional):
            Whether to push the model to the hub. Defaults to True.
    '''
    # Dataset parameters
    dataset_id: str = 'mozilla-foundation/common_voice_8_0'
    dataset_subset: Optional[str] = 'da'
    sampling_rate: int = 16_000
    train_name: str = 'train'
    validation_name: Optional[str] = 'validation'
    test_name: Optional[str] = 'test'

    # Model hyperparameters
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    mask_time_prob: float = 0.05
    layerdrop: float = 0.0
    ctc_loss_reduction: str = 'sum'
    freeze_feature_encoder: bool = False

    # Training hyperparameters
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    epochs: int = 100
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    fp16: bool = True
    push_to_hub: bool = True
