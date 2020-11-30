# -*- coding: utf-8 -*-
"""
create complex classifier
    * using tf custom layer method

@author: Jacky Gao
@date: 2020/11/30
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers


class Block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        self.filters = filters
        super(Block, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=(3,3),
                                           strides=(1,1),
                                           padding='same',
                                           # use_bias=True,
                                           kernel_initializer='he_normal',
                                           kernel_regularizer=None)
        self.bn = tf.keras.layers.BatchNormalization()
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters': self.filters,
        })
        return config

    def call(self, inputs, training=None, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x


def build_model(input_shape=(256,256,1)):
    # Amplitude
    XA = layers.Input(shape=input_shape, name='XA')
    xa = Block(8)(XA)
    xa = layers.MaxPooling2D(pool_size=(2,2))(xa)
    xa = Block(16)(xa)
    xa = layers.MaxPooling2D(pool_size=(2,2))(xa)
    xa = Block(32)(xa)
    xa = layers.MaxPooling2D(pool_size=(2,2))(xa)
    xa = Block(64)(xa)
    xa = layers.MaxPooling2D(pool_size=(2,2))(xa)
    xa = layers.Flatten()(xa)
    xa = layers.Dense(128, activation='relu')(xa)
    xa = layers.Dropout(0.5)(xa)
    YA = layers.Dense(6, activation='softmax', name='YA')(xa)
    
    # phase
    XP = layers.Input(shape=input_shape, name='XP')
    xp = Block(8)(XP)
    xp = layers.MaxPooling2D(pool_size=(2,2))(xp)
    xp = Block(16)(xp)
    xp = layers.MaxPooling2D(pool_size=(2,2))(xp)
    xp = Block(32)(xp)
    xp = layers.MaxPooling2D(pool_size=(2,2))(xp)
    xp = Block(64)(xp)
    xp = layers.MaxPooling2D(pool_size=(2,2))(xp)
    xp = layers.Flatten()(xp)
    xp = layers.Dense(128, activation='relu')(xp)
    xp = layers.Dropout(0.5)(xp)
    YP = layers.Dense(6, activation='softmax', name='YP')(xp)
    
    # complex
    xc = layers.Concatenate(axis=-1)([YA,YP])
    xc = layers.Dense(128, activation='relu')(xc)
    xc = layers.Dropout(0.5)(xc)
    YC = layers.Dense(6, activation='softmax', name='YC')(xc)
    return models.Model(inputs={'XA': XA, 'XP': XP},
                        outputs={'YC': YC})


if __name__ == '__main__':    
    model = build_model()
    model.summary()