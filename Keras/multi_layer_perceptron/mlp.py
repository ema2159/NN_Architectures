"""
This file implements a multi layer perceptron using Tensorflow's Keras API.
Taken from Keras official documentation:
https://www.tensorflow.org/tutorials/keras/basic_classification

"""
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


# Implement MLP
mlp = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

mlp.summary()
mlp.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Test The MLP with the fashion MNIST data set:
# Load Fashion MNIST data set
fashion_mnist = keras.datasets.fashion_mnist

# Split the data set into train and test data.
# The fashion_mnist.load_data() method returns a tuple with two elements, with each element being
# also a two element tuple (one tuple contains the train data and the other contains the test data)
# that contains the images and their labels as NumPy arrays.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# A string with the string names of each label for the Fashion MNIST data set.
# For example, if a label for a certain image is 2, then it's string name is "Pullover".
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Show a plot with examples of the data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Preprocess the data scaling it from 0 to 1:
train_images = train_images / 255.0
test_images = test_images / 255.0

# Train the model.
model.fit(train_images, train_labels, epochs=5)
