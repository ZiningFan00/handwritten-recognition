import scipy.io
import matplotlib.pyplot as plt
import cv2
# from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# read files
train = scipy.io.loadmat('train_32x32.mat')
test = scipy.io.loadmat('test_32x32.mat')
X_train = train["X"]
y_train = train["y"]
len_train=len(y_train)
X_test = test["X"]
y_test = test["y"]
len_test=len(y_test)

def cnn(X,y):

    model=Sequential()

# X_train = X_train.reshape(len_train,32,32,1)
# X_test = X_test.reshape(len_test,32,32,1)

# for i in range(0,len(mat["y"])-1):

#     img=mat["X"][:,:,:,i]
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("img",img)
#     cv2.imshow("gray",gray_img)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

a=0