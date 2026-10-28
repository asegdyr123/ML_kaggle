"""Global stochastic pairwise logistic objective (ranking, not calibration)."""
import numpy as np


def sigmoid(values):
    """Stable sigmoid, preserving float64 precision for exported ranking scores."""
    values = np.asarray(values, dtype=np.float64)
    z = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + z), z / (1 + z))


def pairwise_derivatives(margins, positives, negatives, *, hessian_floor=1e-6):
    """Derivatives of n/(2*pairs) * sum(logaddexp(0, m_neg-m_pos)).

    Indices describe fixed pairs and may repeat. The diagonal Hessian drops
    cross-row terms, the usual boosting approximation for coupled ranking loss.
    """
    margins = np.asarray(margins, dtype=np.float64).reshape(-1)
    positives = np.asarray(positives, dtype=np.intp).reshape(-1)
    negatives = np.asarray(negatives, dtype=np.intp).reshape(-1)
    if len(positives) == 0 or len(positives) != len(negatives):
        raise ValueError("Expected equal nonempty positive/negative pair arrays")
    d = sigmoid(margins[negatives] - margins[positives])
    curvature = d * (1 - d)
    n = len(margins)
    scale = n / (2 * len(positives))
    grad = (np.bincount(negatives, weights=d, minlength=n)
            - np.bincount(positives, weights=d, minlength=n)) * scale
    hess = (np.bincount(negatives, weights=curvature, minlength=n)
            + np.bincount(positives, weights=curvature, minlength=n)) * scale
    return grad, np.maximum(hess, hessian_floor)


class GlobalPairwiseObjective:
    """Uniform negatives across the training fold; fresh deterministic pairs/round.

    XGBoost 3.4's callable API predicts NumPy margins even with CUDA trees and
    accepts NumPy derivatives. A CuPy input can also cross explicitly to host.
    Sampling never uses validation rows, query IDs, or validation labels.
    """

    def __init__(self, negatives_per_positive=32, seed=2026):
        if negatives_per_positive < 1:
            raise ValueError("negatives_per_positive must be positive")
        self.negatives_per_positive = negatives_per_positive
        self.rng = np.random.default_rng(seed)
        self.training_matrix = None

    def __call__(self, margins, dtrain):
        if self.training_matrix is None:
            if len(dtrain.get_weight()):
                raise ValueError("Pairwise objective does not support sample weights")
            labels = dtrain.get_label()
            if not np.all((labels == 0) | (labels == 1)):
                raise ValueError("Pairwise objective requires binary training labels")
            self.positive_indices = np.flatnonzero(labels == 1)
            self.negative_indices = np.flatnonzero(labels == 0)
            if not len(self.positive_indices) or not len(self.negative_indices):
                raise ValueError("Pairwise objective requires both training classes")
            self.pair_positives = np.repeat(self.positive_indices, self.negatives_per_positive)
            self.training_matrix = dtrain
        elif dtrain is not self.training_matrix:
            raise ValueError("Pairwise objective is bound to one training DMatrix")
        sampled = self.rng.choice(self.negative_indices, size=len(self.pair_positives), replace=True)
        if hasattr(margins, "__cuda_array_interface__"):
            margins = margins.get()
        return pairwise_derivatives(margins, self.pair_positives, sampled)
