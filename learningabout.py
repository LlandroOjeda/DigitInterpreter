import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd


batch_size = 128
num_classes = 10
epochs = 10
index = 66

# load the MNIST data set into train and test sets
# x_train is numpy array 60000 elements long consisting of a 28x28 Matrix with values ranging from 0 to 255
(x_train, y_train), (x_test, y_test) = mnist.load_data()

df = pd.DataFrame(x_train[0])
print(df.shape)
print(x_train[0])
print(len(x_train))

print(y_train[index])
plt.imshow(x_train[index])
plt.show()
