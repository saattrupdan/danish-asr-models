'''Function to compute the metrics for speech recognition models'''

import numpy as np
from transformers import (Wav2Vec2Processor,
                          Wav2Vec2ProcessorWithLM,
                          EvalPrediction)
from datasets import load_metric
from typing import Dict, Union


def compute_metrics(pred: EvalPrediction,
                    processor: Union[Wav2Vec2Processor,
                                     Wav2Vec2ProcessorWithLM]
                    ) -> Dict[str, float]:
    '''Compute the word error rate of predictions.

    Args:
        pred (EvalPrediction):
            Prediction output of the speech recognition model.
        processor (Wav2Vec2Processor or Wav2Vec2ProcessorWithLM):
            Audio and transcription processor.

    Returns:
        dict:
            Dictionary with 'wer' as the key and the word error rate as the
            value.
    '''
    # Intitialise the metric
    wer_metric = load_metric('wer')

    # Get the padding token
    pad_token = processor.tokenizer.pad_token_id

    # Set the ground truth labels with label id -100 to be the padding token id
    pred.label_ids[pred.label_ids == -100] = pad_token

    # Get the ids of the predictions
    pred_logits = pred.predictions

    # In cases where the logits are -100 for all characters, convert the logits
    # for the padding character to be 100, to make sure that padding is chosen
    pad_arr = pred.predictions[:, :, pad_token]
    pred.predictions[:, :, pad_token] = np.where(pad_arr == -100, 100, pad_arr)

    breakpoint()

    # Decode the predictions, to get the transcriptions
    if isinstance(processor, Wav2Vec2ProcessorWithLM):
        pred_str = processor.batch_decode(pred_logits).text
    else:
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)

    # Decode the ground truth labels
    label_str = processor.tokenizer.batch_decode(pred.label_ids,
                                                 group_tokens=False)

    import itertools as it
    for label, pred in it.islice(zip(label_str, pred_str), 10):
        print()
        print(label)
        print(pred)
        print()

    # Compute the word error rate
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    # Return the word error rate
    return dict(wer=wer)
