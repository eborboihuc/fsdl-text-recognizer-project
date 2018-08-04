from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]
    layer_size = 128
    dropout_amount = 0.2
    ##### Your code below (Lab 2)
   
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)

    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_amount))
        
    model.add(Flatten())
    

    model.add(Dense(layer_size, activation='elu'))
    model.add(Dropout(dropout_amount))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    ##### Your code above (Lab 2)

    return model

