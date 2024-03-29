import scipy.io
import matplotlib.pyplot as plt
import cv2
import numpy as np
# from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# read files
train = scipy.io.loadmat('/Users/fanzining/CU/fall2019/5293/hw/individualProject/train_32x32.mat')
test = scipy.io.loadmat('/Users/fanzining/CU/fall2019/5293/hw/individualProject/test_32x32.mat')
X_train_RGB = train["X"]
y_train = train["y"]
len_train=len(y_train)
X_test_RGB = test["X"]
y_test = test["y"]
len_test=len(y_test)

# convert to gray image
X_train=np.ndarray(shape=(len_train,32,32,1))
X_test=np.ndarray(shape=(len_test,32,32,1))

for i in range(0,len_train):
    img = cv2.cvtColor(X_train_RGB[:,:,:,i],cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    X_train[i,:,:,:] = img

for i in range(0,len_test):
    img = cv2.cvtColor(X_test_RGB[:,:,:,i],cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=-1)
    X_test[i,:,:,:] = img

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 11)
y_test = keras.utils.to_categorical(y_test, 11)

#build cnn model 
model = Sequential()
model.add(Conv2D(16,kernel_size=5,activation='relu',input_shape=(32,32,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(32,kernel_size=5,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,kernel_size=5,activation='relu'))
model.add(Flatten())
model.add(Dense(11,activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=12,verbose=1)
score = model.evaluate(X_test, y_test, verbose=1)