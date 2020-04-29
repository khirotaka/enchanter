from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


__all__ = [
    "calculate_eer"
]


def calculate_eer(y_true, y_score):
    """
    Returns the equal error rate for a binary classification output.
    The equal error rate is the value of FPR (or FNR) when the ROC curves intersects the line connecting (0,0) to (1,1).

    References:
        `Should sklearn include the Equal Error Rate metric \
            <https://github.com/scikit-learn/scikit-learn/issues/15247>`_

    Args:
        y_true (np.ndarray (1D, float)):
        y_score (np.ndarray (1D, float)):

    Returns:
        return EER Score
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
    eer_score = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer_score
