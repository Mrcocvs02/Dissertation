import keras
from keras import layers
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) =mnist.load_data()

x_train = x_train.astype('float32')/255.
x_test = x_test.astype('float32')/255.
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))
print(x_train.shape)
print(x_test.shape)

encoding_dim=32

input_img = keras.Input(shape=(748,))
encoded =layers.Dense(encoding_dim, activation='relu')(input_img)
