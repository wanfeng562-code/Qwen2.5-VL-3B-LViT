"""Foreground metrics for legacy_subset_v1; returns fractions, not percentages."""

import numpy as np


def binary_sample_scores(probability, target, threshold=0.5):
    prediction = np.asarray(probability) >= threshold
    truth = np.asarray(target) > 0
    if prediction.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ")
    pred_count, true_count = int(prediction.sum()), int(truth.sum())
    if pred_count == 0 and true_count == 0:
        return {"dice": 1.0, "iou": 1.0}
    intersection = int(np.count_nonzero(prediction & truth))
    union = pred_count + true_count - intersection
    return {
        "dice": 2.0 * intersection / (pred_count + true_count),
        "iou": intersection / union if union else 1.0,
    }


def macro_scores(sample_scores):
    scores = list(sample_scores)
    if not scores:
        raise ValueError("No samples to average")
    return {
        "sample_count": len(scores),
        "dice": sum(score["dice"] for score in scores) / len(scores),
        "iou": sum(score["iou"] for score in scores) / len(scores),
    }
