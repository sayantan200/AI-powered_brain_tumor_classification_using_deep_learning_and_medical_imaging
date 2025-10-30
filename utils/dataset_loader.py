from __future__ import annotations

import os
from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf


def make_datasets(train_dir: str,
                  test_dir: str,
                  image_size: Tuple[int, int] = (224, 224),
                  batch_size: int = 32,
                  val_split: float = 0.15,
                  seed: int = 42) -> Dict[str, tf.data.Dataset]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical'
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )

    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    def _augment(x, y):
        # Strong augmentation for better generalization
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)
        x = tf.image.random_brightness(x, 0.3)  # Increased brightness variation
        x = tf.image.random_contrast(x, 0.7, 1.3)  # Increased contrast variation
        x = tf.image.random_hue(x, 0.2)  # Increased hue variation
        x = tf.image.random_saturation(x, 0.7, 1.3)  # Increased saturation variation
        
        # Add rotation with higher probability
        if tf.random.uniform([]) < 0.5:
            x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Add noise for robustness
        if tf.random.uniform([]) < 0.3:
            noise = tf.random.normal(tf.shape(x), 0, 0.01)
            x = tf.clip_by_value(x + noise, 0, 1)
        
        # Add blur for robustness
        if tf.random.uniform([]) < 0.2:
            x = tf.nn.avg_pool2d(x, ksize=2, strides=1, padding='SAME')
            x = tf.image.resize(x, tf.shape(x)[1:3])
        
        return x, y

    train_ds = train_ds.map(_augment, num_parallel_calls=autotune)

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return {
        'train': train_ds,
        'val': val_ds,
        'test': test_ds,
        'class_names': class_names,
    }


__all__ = ['make_datasets']


