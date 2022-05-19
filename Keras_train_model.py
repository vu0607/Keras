#Load data, split train, val, test
from numpy import loadtxt
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
dataset = loadtxt('train.csv', delimiter= ',')
X = dataset[:, 0:8]
y = dataset[:, 8]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size= 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size= 0.2)
model = Sequential()
model.add(Dense(16, input_dim= 8, activation= 'relu'))   #relu la ham khu tuyen tinh
model.add(Dense(8, activation= 'relu'))
model.add(Dense(1, activation= 'sigmoid'))               #Do thi dang >0.5 du doan (+), <0.5 du doan (-)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #Du lieu dang 1 hoac 0 su dung binary_cross, nhieu hon dung classify...
model.fit(X_train, y_train, epochs= 100, batch_size= 8, validation_data=(X_val, y_val) ) #epochs : so vong train, batch_size : so cap du lieu dua vao input
model.save('mymodel.h5')
model = load_model('mymodel.h5')
loss, acc = model.evaluate(X_test, y_test)
print('Loss', loss)
print('acc', acc)