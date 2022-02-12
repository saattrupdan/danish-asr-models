'''Finetuning script for Danish speech recognition'''

from transformers import (Wav2Vec2ForCTC,
                          TrainingArguments,
                          Trainer)
from typing import Optional, Union
from data import AudioDataset
from data_collator import DataCollatorCTCWithPadding
from compute_metrics import compute_metrics
from functools import partial
from config import Config


def train(pretrained_model_id: str,
          finetuned_model_id: str,
          config: Optional[Union[dict, Config]] = None):
    '''Finetune a pretrained audio model on a dataset.

    Args:
        pretrained_model_id (str):
            The model id of the pretrained model to finetune.
        finetuned_model_id (str):
            The model id of the finetuned model.
        config (Config, dict or None):
            Config object or dict containing the parameters for the finetuning.
            If None then a Config object is created from the default
            parameters. Defaults to None.
    '''
    # If no config is provided, create a config object from the default
    # parameters
    if config is None:
        config = Config()

    # If a dict is provided, create a config object from the dict
    elif isinstance(config, dict):
        config = Config(**config)

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
        pretrained_model_id,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        mask_time_prob=config.mask_time_prob,
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
        output_dir=finetuned_model_id.split('/')[-1],
        hub_model_id=finetuned_model_id,
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
        length_column_name='input_length'
    )

    # Initialise the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=partial(compute_metrics, processor=dataset.processor),
        train_dataset=dataset.train,
        eval_dataset=dataset.val,
        tokenizer=dataset.tokenizer
    )

    # Save the preprocessor
    dataset.processor.save_pretrained(finetuned_model_id.split('/')[-1])

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(finetuned_model_id.split('/')[-1])

    # Push the model to the hub
    if config.push_to_hub:
        trainer.push_to_hub()



if __name__ == '__main__':
    config = Config(mask_time_prob=0.05,
                    learning_rate=4e-5,
                    epochs=100,
                    batch_size=4,
                    gradient_accumulation_steps=8)
    train(pretrained_model_id='facebook/wav2vec2-xls-r-300m',
          finetuned_model_id='saattrupdan/wav2vec2-xls-r-300m-cv8-da',
          config=config)
