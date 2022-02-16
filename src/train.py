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
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        final_dropout=config.final_dropout,
        mask_time_prob=config.mask_time_prob,
        mask_feature_prob=config.mask_feature_prob,
        layerdrop=config.layerdrop,
        ctc_loss_reduction=config.ctc_loss_reduction,
        pad_token_id=dataset.tokenizer.pad_token_id,
        vocab_size=len(dataset.tokenizer)
    )

    # Freeze the feature encoder
    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # Initialise training arguments
    training_args = TrainingArguments(
        output_dir=config.finetuned_model_id.split('/')[-1],
        hub_model_id=config.finetuned_model_id,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        fp16=config.fp16,
        push_to_hub=config.push_to_hub,
        evaluation_strategy='steps',
        eval_steps=300,
        save_steps=300,
        logging_steps=100,
        group_by_length=True,
        gradient_checkpointing=True,
        save_total_limit=2,
        length_column_name='input_length',
        load_best_model_at_end=config.early_stopping,
        metric_for_best_model='wer',
        greater_is_better=False
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
    trainer.train()

    # Save the model
    model.save_pretrained(config.finetuned_model_id.split('/')[-1])

    # Push the model to the hub
    if config.push_to_hub:
        trainer.push_to_hub()



if __name__ == '__main__':

    xlsr_300m_config = Config(
        pretrained_model_id='facebook/wav2vec2-xls-r-300m',
        finetuned_model_id='saattrupdan/wav2vec2-xls-r-300m-cv8-da',
        mask_time_prob=0.075,
        mask_feature_prob=0.075,
        learning_rate=4e-5,
        epochs=500,
        warmup_steps=500,
        batch_size=4,
        gradient_accumulation_steps=8,
        attention_dropout=0.1,
        feat_proj_dropout=0.1,
        hidden_dropout=0.1,
        final_dropout=0.1,
        layerdrop=0.1,
        early_stopping=True,
        early_stopping_patience=5
    )

    alvenir_config = Config(
        pretrained_model_id='Alvenir/wav2vec2-base-da',
        finetuned_model_id='saattrupdan/alvenir-wav2vec2-base-cv8-da',
        mask_time_prob=0.075,
        mask_feature_prob=0.075,
        learning_rate=4e-5,
        epochs=500,
        warmup_steps=500,
        batch_size=8,
        gradient_accumulation_steps=4,
        attention_dropout=0.1,
        feat_proj_dropout=0.1,
        hidden_dropout=0.1,
        final_dropout=0.3,
        layerdrop=0.1,
        early_stopping=True,
        early_stopping_patience=5
    )

    train(xlsr_300m_config)
