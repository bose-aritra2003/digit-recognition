import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Import the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# # For RGB images we need to scale down the data from 0-255 to 0-1
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
# model = tf.keras.models.Sequential()  # A basic feed-forward model
#
# # Input Layer
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # Takes our 28x28 and makes it 1x784
#
# # Hidden Layers
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # A simple fully-connected layer, 128 units, relu activation
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # 2nd hidden layer
#
# # Output Layer
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # A softmax layer with 10 output units
#
# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
#
# # Train the model
# model.fit(x_train, y_train, epochs=5)
#
# # Save the model
# model.save('digits.model')

# Load the model
model = tf.keras.models.load_model('digits.model')

# Evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Loss: {val_loss}')
print(f'Accuracy: {val_acc}')
