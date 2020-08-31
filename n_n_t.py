import tensorflow
import keras
import numpy
import matplotlib.pyplot as plt

# print("neural network is real")

data = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels)=data.load_data()