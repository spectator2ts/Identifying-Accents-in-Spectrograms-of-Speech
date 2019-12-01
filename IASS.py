import numpy as np 
import pandas as pd 
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import time

# Define data path
TRAIN_PATH = "/kaggle/input/speech-spectrograms/speech-spectrograms/train/"
TEST_PATH = "/kaggle/input/speech-spectrograms/speech-spectrograms/test"

def match(y):
	"""
	y: = Array[String]
	return the matched x and y given the input y
	"""
    x = []
    for i, img in enumerate(y["file_id"]):
        path = os.path.join(TRAIN_PATH, '{}.png'.format(img))
        img_array = cv2.imread(path, 0)
        x.append(img_array)

    x = np.array(x).reshape(-1, x[0].shape[0], x[0].shape[1], 1) # reshape
    x = np.subtract(np.multiply(np.true_divide(x, 255.0), 2), 1) # normalize to (-1, 1)
    # x = np.true_divide(x_train, 255.0 # normalize to (0, 1) 
    y = to_categorical(y["accent"].values)
    return x, y

def LoadData():
	"""
	Load train and test data
	"""
    y = pd.read_csv("/kaggle/input/speech-spectrograms/speech-spectrograms/train_labels.csv")
    _len = len(y) // 10 * 8
    y_train, y_test = y[: _len], y[_len: ]
    x_train, y_train = match(y_train)
    x_test, y_test = match(y_test)
    return x_train, y_train, x_test, y_test

# Load data
x_train, y_train, x_test, y_test = LoadData()


# Build model
model = Sequential()

model.add(Conv2D(64, (2, 3), input_shape=x_train.shape[1:], kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,3), padding='same'))

model.add(Conv2D(128, (2, 3), kernel_regularizer=regularizers.l2(0.02)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,3), padding='same'))

model.add(Conv2D(128, (2, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation("relu"))

model.add(Conv2D(256, (2, 3), kernel_regularizer=regularizers.l2(0.03)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,3), padding='same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization(momentum=0.8))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(3))
model.add(Activation("softmax"))

#sgd = optimizers.SGD(lr=0.04, decay=1e-10, momentum=0.9, nesterov=True)
ada = optimizers.Adam(lr=3e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)
model.compile(loss="categorical_crossentropy",
              optimizer=ada,
              metrics=["accuracy"])

# Data augmentation
datagen = ImageDataGenerator(width_shift_range=0.4)
datagen.fit(x_train)

# Optimization
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    epochs=150, validation_data=(x_test, y_test))

model.summary()

