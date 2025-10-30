from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_efficientnet_b0(input_shape: Tuple[int, int, int], num_classes: int, train_base: bool = False) -> tf.keras.Model:
    base = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = train_base

    inputs = layers.Input(shape=input_shape)
    x = applications.efficientnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='efficientnetb0_classifier')
    return model


__all__ = ["build_efficientnet_b0"]




