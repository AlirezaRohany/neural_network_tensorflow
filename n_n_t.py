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

# plt.imshow(train_images[100], cmap=plt.cm.binary)
# plt.show()

# creating a model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer, flatten our pixels (flatten input layer)
    keras.layers.Dense(128, activation="relu"),  # hidden layer, with 'rectified linear unit' activation function
    keras.layers.Dense(10, activation="softmax")  # output layer
])

# setting up some parameters for our model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# adam is typical optimizer, loss function can vary based on our application

# training our model
model.fit(train_images, train_labels, epochs=10)  # epochs is how many times it's gonna see the  same image

# testing our model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("\n", "Test accuracy:", test_acc)
