from keras.layers import Input, MaxPooling2D, Conv2D, GlobalAveragePooling1D, MaxPooling1D, Conv1D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

def get_model(num_classes, image_size):
    bird_model = get_bird_model(num_classes, image_size)
    return bird_model

def get_bird_model(num_classes, image_size):
    #Create model
    model = Sequential()
    model.add(Conv2D(100, kernel_size=(3, 3), activation = 'relu', input_shape=image_size))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(100, kernel_size=(3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Conv2D(100, kernel_size=(3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model