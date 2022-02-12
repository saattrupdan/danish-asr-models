'''Finetuning script for Danish speech recognition'''

from transformers import Wav2Vec2ForCTC, TrainingArguments, Trainer
from typing import Optional, Union
from data import AudioDataset
from data_collator import DataCollatorCTCWithPadding
from compute_metrics import compute_metrics
from config import Config


def train(config: Optional[Union[dict, Config]] = None):
    '''Finetune a pretrained audio model on a dataset.

    Args:
        config (Config, dict or None):
            Config object or dict containing the parameters for the finetuning.
            If None then a Config object is created from the default
            parameters. Defaults to None.
    '''
    # If no config is provided, create a config object from the default
    # parameters
    if config is None:
        config = Config()
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

    # Push the tokenizer to the hub
    if config.push_to_hub:
        dataset.tokenizer.push_to_hub(config.finetuned_model_id)

    # Initialise data collator
    data_collator = DataCollatorCTCWithPadding(processor=dataset.processor,
                                               padding='longest')

    # Initialise the model
    model = Wav2Vec2ForCTC.from_pretrained(
        config.pretrained_model_id,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        mask_time_prob=config.mask_time_prob,
        layerdrop=config.layerdrop,
        ctc_loss_reduction=config.ctc_loss_reduction,
        pad_token_id=dataset.tokenizer.pad_token_id,
        vocab_size=len(dataset.tokenizer)
    )

    # Freeze the feature extractor
    model.freeze_feature_encoder()

    # Initialise training arguments
    training_args = TrainingArguments(
        output_dir=config.finetuned_model_id.split('/')[-1],
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        fp16=config.fp16,
        push_to_hub=config.push_to_hub,
        evaluation_strategy='steps',
        eval_steps=400,
        save_steps=400,
        logging_steps=400,
        group_by_length=True,
        gradient_checkpointing=True,
        save_total_limit=2
    )

    # Initialise the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset.train,
        eval_dataset=dataset.val,
        tokenizer=dataset.tokenizer
    )

    print(dataset.train[0].keys())

    # Train the model
    trainer.train()

    # Push the model to the hub
    if config.push_to_hub:
        trainer.push_to_hub()



if __name__ == '__main__':
    train()
