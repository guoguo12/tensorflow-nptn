import keras
from keras.layers import (BatchNormalization, Conv2D, Dense, Flatten,
                          MaxPooling2D)
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential

def baseline_convnet(input_shape, num_classes):
    """
    Baseline two-layer ConvNet as described in the NPTN paper.
    """
    model = Sequential()

    model.add(Conv2D(3 * 48, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48 * 16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return model
