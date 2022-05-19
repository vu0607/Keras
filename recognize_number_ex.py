from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib as plt
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
#Load data mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]
X_train, y_train = X_train[:50000], y_train[:50000]

#Reshape lai du lieu cho dung kich huoc ma keras yeu cau
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#Chuyen y ve one-hot vector, la vector ma tat ca cac so la 0 chi co 1 so la 1
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)

#Dinh nghia model
model = Sequential()

#Them convolutional layer voi 32 kernel, kich thuoc 3x3
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='sigmoid'))
model.add(Dense(10,activation='softmax'))
model.compile( loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=20, epochs=5, verbose=1)
loss, acc = model.evaluate(X_test, y_test)
print('Loss =',loss)
print('Accuracy= ', acc)


