import tensorflow
import keras
import numpy
import matplotlib.pyplot as plt

# print("neural network is real")

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
# print(train_labels[6])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# print(class_names[train_labels[6]])

train_images = train_images / 255.0
test_images = test_images / 255.0

# print(train_images[100])

plt.imshow(train_images[100], cmap=plt.cm.binary)
plt.show()
