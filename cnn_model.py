"""
convolution model architecture
"""
import tensorflow as tf
import numpy as np
import os
import keras

def cnn_prediction():
    model = tf.keras.Sequential()
    #1
    model.add(tf.keras.layers.Conv2D(filters=32, 
                                     kernel_size=(3, 3), 
                                     input_shape=(500, 1000, 3),
                                     activation='relu', 
                                     padding='same')
                                     )
    model.add(tf.keras.layers.Conv2D(filters=32, 
                                     kernel_size=(3, 3),
                                     activation='relu', 
                                     padding='same')
                                     )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #2
    model.add(tf.keras.layers.Conv2D(filters=64, 
                                     kernel_size=(3, 3),
                                     activation='relu', 
                                     padding='same')
                                     )
    model.add(tf.keras.layers.Conv2D(filters=64, 
                                     kernel_size=(3, 3),
                                     activation='relu', 
                                     padding='same')
                                     )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #3
    model.add(tf.keras.layers.Conv2D(filters=128, 
                                     kernel_size=(3, 3),
                                     activation='relu', 
                                     padding='same')
                                     )
    model.add(tf.keras.layers.Conv2D(filters=128, 
                                     kernel_size=(3, 3),
                                     activation='relu', 
                                     padding='same')
                                     )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #4
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(tf.keras.layers.Dense(5, activation='softmax'))

    return model


