from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
model = Sequential()
def cps():
    model.add(Convolution2D(filters=32, kernel_size=(6, 6),activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

