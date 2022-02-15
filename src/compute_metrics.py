'''Function to compute the metrics for speech recognition models'''

import numpy as np
from transformers import Wav2Vec2Processor, EvalPrediction
from datasets import load_metric
from typing import Dict
import itertools as it


def compute_metrics(pred: EvalPrediction,
                    processor: Wav2Vec2Processor) -> Dict[str, float]:
    '''Compute the word error rate of predictions.

    Args:
        pred (EvalPrediction):
            Prediction output of the speech recognition model.
        processor (Wav2Vec2Processor):
            Audio and transcription processor.

    Returns:
        dict:
            Dictionary with 'wer' as the key and the word error rate as the
            value.
    '''
    # Intitialise the metric
    wer_metric = load_metric('wer')

    # Get the ids of the predictions
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode the predictions to get the transcriptions
    pred_str = processor.batch_decode(pred_ids)

    # Set the ground truth labels with label id -100 to be the padding token id
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode the ground truth labels
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    # Compute the word error rate
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    # Return the word error rate
    return dict(wer=wer)
