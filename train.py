import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

classes = ['Fall Down', 'Lying Down', 'Sit down', 'Sitting', 'Stand up', 'Standing', 'Walking']

num_of_timesteps = 5
num_classes = len(classes)

X_train, y_train = [], []

label = 0

for cl in classes:
    for file in os.listdir(f'./dataset/csv/train/{cl}'):
        data = pd.read_csv(f'./dataset/csv/train/{cl}/{file}')
        data = data.iloc[:, 1:].values
        n_sample = len(data)
        for i in range(num_of_timesteps, n_sample):
            X_train.append(data[i - num_of_timesteps : i, :])
            y_train.append(label)
    label = label + 1

print("Train set - completed")

X_test, y_test = [], []

label = 0

for cl in classes:
    for file in os.listdir(f'./dataset/csv/test/{cl}'):
        data = pd.read_csv(f'./dataset/csv/test/{cl}/{file}')
        data = data.iloc[:, 1:].values
        n_sample = len(data)
        for i in range(num_of_timesteps, n_sample):
            X_test.append(data[i - num_of_timesteps : i, :])
            y_test.append(label)
    label = label + 1

print("Test set - completed")

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train, y_train)
print(X_train.shape, y_train.shape)

X_test, y_test = np.array(X_test), np.array(y_test)
print(X_test, y_test)
print(X_test.shape, y_test.shape)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(LSTM(units=128, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 128))
model.add(Dropout(0.2))
model.add(Dense(units = num_classes, activation="softmax"))
model.compile(optimizer="adam", metrics = ['accuracy'], loss = "categorical_crossentropy")
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=16,validation_data=(X_test, y_test))
model.save("model/model.h5")

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
visualize_loss(history, "Training and Validation Loss")