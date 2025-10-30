from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(history: Dict[str, List[float]], out_path: str) -> None:
    plt.figure(figsize=(10, 4))
    # Accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history:
        plt.plot(history['accuracy'], label='train')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    if 'loss' in history:
        plt.plot(history['loss'], label='train')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str, normalize: bool = True) -> None:
    cm_to_plot = cm.astype('float')
    if normalize and cm_to_plot.sum(axis=1, keepdims=True).all():
        with np.errstate(all='ignore'):
            cm_to_plot = cm_to_plot / (cm_to_plot.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_to_plot, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


__all__ = ['plot_training_curves', 'plot_confusion_matrix']




