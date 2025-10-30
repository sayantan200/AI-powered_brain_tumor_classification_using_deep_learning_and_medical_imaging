from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def classification_report_from_probs(y_true: np.ndarray,
                                     y_prob: np.ndarray,
                                     class_names: List[str]) -> Dict[str, float]:
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true_labels, y_pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true_labels, y_pred_labels)

    return {
        'accuracy': float(acc),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
        'confusion_matrix': cm.tolist(),
    }


__all__ = ['classification_report_from_probs']




