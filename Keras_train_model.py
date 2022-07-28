#Load data, split train, val, test

import numpy
from numpy import loadtxt
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.layers.normalization import batch_normalization
from sklearn.model_selection import train_test_split
dataset = loadtxt('train.csv', delimiter= ',')
X = dataset[:, 0:8]
y = dataset[:, 8]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size= 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size= 0.2)
model = Sequential()
model.add(Dense(256, input_dim=8, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu')) #relu la ham khu tuyen tinh
model.add(Dense(8, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
             #Do thi dang >0.5 du doan (+), <0.5 du doan (-)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #Du lieu dang 1 hoac 0 su dung binary_cross, nhieu hon dung classify...
H = model.fit(X_train, y_train, epochs= 64, batch_size= 8, validation_data=(X_val, y_val) ) #epochs : so vong train, batch_size : so cap du lieu dua vao input
model.save('model.h5')

model = load_model('model.h5')
loss, acc = model.evaluate(X_test, y_test)
print('Loss', loss)
print('acc', acc)
X_new = X_test[10]
y_new = y_test[10]
X_new = numpy.expand_dims(X_new, axis= 0) # Them chieu cho X_new, vao vi tri dau tien axis = 0 
y_predict = model.predict(X_new)
if y_predict <= 0.5:
    print("Khong tieu duong (0)")
else: print("Mac benh tieu duong")
