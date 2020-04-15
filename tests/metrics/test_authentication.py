import numpy as np
from sklearn.metrics import roc_curve
import enchanter.metrics as metrics


def test_eer_1():
    # FRR = FN / (FN + TP)  ... FNR
    # FAR = FP / (TN + FP)  ... FPR = FP / (FP + TN)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1]).astype(np.float32)
    y_score = np.array([0.2, 0.3, 0.6, 0.8, 0.4, 0.5, 0.7, 0.9]).astype(np.float32)
    far, tpr, thresholds = roc_curve(y_true, y_score)

    frr = 1.0 - tpr
    idx = np.argmin(np.abs(far - frr))
    eer_from_frr_and_far = np.abs(far[idx] + frr[idx]) / 2
    eer = np.round(metrics.calculate_eer(y_true, y_score), 2)

    assert eer_from_frr_and_far == eer
