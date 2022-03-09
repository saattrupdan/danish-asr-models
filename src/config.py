'''Config class that carries all the hyperparameters needed for training'''

from pydantic import BaseModel
from typing import Optional


class Config(BaseModel):
    '''Config class that carries all the hyperparameters needed for training.

    Args:
        pretrained_model_id (str):
            The model id of the pretrained model to finetune.
        finetuned_model_id (str):
            The model id of the finetuned model.
        dataset_id (str):
            The id of the dataset to finetune on.
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
        activation_dropout (float, optional):
            The dropout rate for the activation layer. Defaults to 0.1.
        attention_dropout (float, optional):
            The dropout rate for the attention layer. Defaults to 0.1.
        hidden_dropout (float, optional):
            The dropout rate for the hidden layer. Defaults to 0.1.
        feat_proj_dropout (float, optional):
            The dropout rate for the feature projection layer. Defaults to 0.1.
        final_dropout (float, optional):
            The dropout rate for the final layer. Defaults to 0.1.
        mask_time_prob (float, optional):
            The probability of masking the time dimension. Defaults to 0.075.
        mask_feature_prob (float, optional):
            The probability of masking the feature dimension. Defaults to
            0.075.
        mask_feature_length (int, optional):
            The length of the masking of the feature dimension. Defaults to
            10.
        layerdrop (float, optional):
            The dropout rate for the layers. Defaults to 0.1.
        ctc_loss_reduction (str, optional):
            The reduction to use for the CTC loss. Defaults to 'sum'.
        freeze_feature_encoder (bool, optional):
            Whether to freeze the feature encoder. Defaults to False.
        batch_size (int, optional):
            The batch size for training. Defaults to 4.
        gradient_accumulation_steps (int, optional):
            The number of steps to accumulate gradients for. Defaults to 8.
        epochs (int, optional):
            The number of epochs to train for. Defaults to 500.
        learning_rate (float, optional):
            The learning rate for the optimizer. Defaults to 4e-5.
        warmup_steps (int, optional):
            The number of warmup steps for the learning rate scheduler.
            Defaults to 500.
        early_stopping (bool, optional):
            Whether to use early stopping. Defaults to True.
        early_stopping_patience (int, optional):
            The patience for early stopping. Only relevant if `early_stopping`
            is True. Defaults to 5.
        fp16 (bool, optional):
            Whether to use 16-bit floating point precision. Note that this is
            only possible if GPU is enabled. Defaults to True.
        push_to_hub (bool, optional):
            Whether to push the model to the hub. Defaults to True.
        resume_from_checkpoint (bool, optional):
            Whether to resume training from a checkpoint. Defaults to False.
    '''
    # Model IDs
    pretrained_model_id: str
    finetuned_model_id: str

    # Dataset hyperparameters
    dataset_id: str
    dataset_subset: Optional[str] = None
    sampling_rate: int = 16_000
    train_name: str = 'train'
    validation_name: Optional[str] = 'validation'
    test_name: Optional[str] = 'test'

    # Model hyperparameters
    activation_dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    feat_proj_dropout: float = 0.1
    final_dropout: float = 0.1
    mask_time_prob: float = 0.075
    mask_feature_prob: float = 0.075
    mask_feature_length: int = 10
    layerdrop: float = 0.1
    ctc_loss_reduction: str = 'sum'
    freeze_feature_encoder: bool = False

    # Training hyperparameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    epochs: int = 500
    learning_rate: float = 4e-5
    warmup_steps: int = 500
    early_stopping: bool = True
    early_stopping_patience: int = 5
    fp16: bool = True
    push_to_hub: bool = True
    resume_from_checkpoint: bool = False
