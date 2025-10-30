from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_resnet50(input_shape: Tuple[int, int, int], num_classes: int, train_base: bool = False) -> tf.keras.Model:
    base = applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = train_base

    inputs = layers.Input(shape=input_shape)
    x = applications.resnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='resnet50_classifier')
    return model


__all__ = ["build_resnet50"]



