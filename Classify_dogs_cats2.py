import pandas as pd
import numpy as np
import os
from keras.datasets import mnist
from keras import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import load_model
from sklearn.model_selection import train_test_split 


width= 128
height= 128
imagesize =(width, height)
image_channel = 3
filenames=os.listdir('dogs-vs-cats/train')
categories=[]
for filename in filenames:
    # class dog, cat 
    category=filename.split('dogs-vs-cats')[0]
    if category.startswith('dog'):
        categories.append(1) # dog = 1
    else:
        categories.append(0) # cat = 0
train_df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

test_filenames = os.listdir('dogs-vs-cats/test1')
test_df = pd.DataFrame({'filename': test_filenames})
nb_samples = test_df.shape[0]
print('train: {}\n test: {}'.format(train_df.shape[0], test_df.shape[0]))

#Model 
#layer1
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(width, height, image_channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

#layer2
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(width, height, image_channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

#layer3
model.add(Conv2D(128, (3,3), activation='relu', input_shape=(width, height, image_channel)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

#layer4
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(loss= 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.sumary()
model.fit