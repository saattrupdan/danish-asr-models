'''Finetuning script for Danish speech recognition'''

from transformers import (Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer,
                          EarlyStoppingCallback)
from typing import Optional, Union
from data import AudioDataset
from data_collator import DataCollatorCTCWithPadding
from compute_metrics import compute_metrics
from functools import partial
from config import Config


def train(config: Optional[Union[dict, Config]] = None, **kwargs):
    '''Finetune a pretrained audio model on a dataset.

    Args:
        config (Config, dict or None):
            Config object or dict containing the parameters for the finetuning.
            If None then a Config object is created from the default
            parameters. Defaults to None.
        **kwargs:
            Keyword arguments to be passed to the config
    '''
    # If no config is provided, create a config object from the default
    # parameters
    if config is None:
        config = Config(**kwargs)

    # If a dict is provided, create a config object from the dict
    elif isinstance(config, dict):
        config = Config(**{**config, **kwargs})

    # If a Config object is provided, update the config object with the
    # provided keyword arguments
    elif isinstance(config, Config) and len(kwargs) > 0:
        config = Config(**{**config.__dict__, **kwargs})

    # Load dataset
    dataset = AudioDataset(dataset_id=config.dataset_id,
                           dataset_subset=config.dataset_subset,
                           sampling_rate=config.sampling_rate,
                           train_name=config.train_name,
                           validation_name=config.validation_name,
                           test_name=config.test_name)

    # Preprocess the dataset
    dataset.preprocess()

    # Initialise data collator
    data_collator = DataCollatorCTCWithPadding(processor=dataset.processor,
                                               padding='longest')

    # Initialise the model
    model = Wav2Vec2ForCTC.from_pretrained(
        config.pretrained_model_id,
        activation_dropout=config.activation_dropout,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        final_dropout=config.final_dropout,
        mask_time_prob=config.mask_time_prob,
        mask_feature_prob=config.mask_feature_prob,
        mask_feature_length=config.mask_feature_length,
        layerdrop=config.layerdrop,
        ctc_loss_reduction=config.ctc_loss_reduction,
        pad_token_id=dataset.tokenizer.pad_token_id,
        bos_token_id=dataset.tokenizer.bos_token_id,
        eos_token_id=dataset.tokenizer.eos_token_id,
        vocab_size=len(dataset.tokenizer.get_vocab()),
    )

    # Freeze the feature encoder
    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # Initialise training arguments
    training_args = TrainingArguments(
        output_dir=config.finetuned_model_id.split('/')[-1],
        hub_model_id=config.finetuned_model_id,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        fp16=config.fp16,
        push_to_hub=config.push_to_hub,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        group_by_length=True,
        #length_column_name='input_length',
        gradient_checkpointing=True,
        save_total_limit=2,
        load_best_model_at_end=config.early_stopping,
        metric_for_best_model='wer',
        greater_is_better=False,
        seed=4242
    )

    # Create early stopping callback
    if config.early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )
        callbacks = [early_stopping_callback]
    else:
        callbacks = []

    # Initialise the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=partial(compute_metrics, processor=dataset.processor),
        train_dataset=dataset.train,
        eval_dataset=dataset.val,
        tokenizer=dataset.tokenizer,
        callbacks=callbacks
    )

    # Save the preprocessor
    dataset.processor.save_pretrained(config.finetuned_model_id.split('/')[-1])

    # Train the model
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    # Save the model
    model.save_pretrained(config.finetuned_model_id.split('/')[-1])

    # Push the model to the hub
    if config.push_to_hub:
        trainer.push_to_hub()



if __name__ == '__main__':

    xlsr_300m_config = Config(
        pretrained_model_id='facebook/wav2vec2-xls-r-300m',
        finetuned_model_id='saattrupdan/wav2vec2-xls-r-300m-cv8-da',
    )

    alvenir_config = Config(
        pretrained_model_id='Alvenir/wav2vec2-base-da',
        finetuned_model_id='saattrupdan/alvenir-wav2vec2-base-cv8-da',
        final_dropout=0.3
    )

    voxrex_config = Config(
        pretrained_model_id='KBLab/wav2vec2-large-voxrex',
        finetuned_model_id='saattrupdan/kblab-voxrex-wav2vec2-large-cv8-da',
        final_dropout=0.3
    )

    voxpopuli_config = Config(
        pretrained_model_id='facebook/wav2vec2-large-sv-voxpopuli',
        finetuned_model_id='saattrupdan/voxpopuli-wav2vec2-large-cv8-da',
        final_dropout=0.3
    )

    ftspeech_config = Config(
        dataset_id='/media/secure/dan/ftspeech/ftspeech',
        dataset_subset=None,
        validation_name='dev_balanced',
        test_name='test_balanced',
        pretrained_model_id='facebook/wav2vec2-xls-r-300m',
        finetuned_model_id='saattrupdan/wav2vec2-xls-r-300m-ftspeech',
        batch_size=1,
        gradient_accumulation_steps=32,
        learning_rate=1e-4,
        warmup_steps=2000,
    )

    train(ftspeech_config)
