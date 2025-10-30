from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def build_vgg16(input_shape: Tuple[int, int, int], num_classes: int, train_base: bool = False) -> tf.keras.Model:
    base = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = train_base

    inputs = layers.Input(shape=input_shape)
    x = applications.vgg16.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs, name='vgg16_classifier')
    return model


__all__ = ["build_vgg16"]



