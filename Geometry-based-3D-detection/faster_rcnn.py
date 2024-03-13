import random
import numpy as np
from optparse import OptionParser
import math
import cv2
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.models import Model

from keras import initializers, regularizers


from keras.layers import Input, Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Flatten, Dense


# In our model, we consider a 3D RPN with a VGG backbone arhitecture

def partial_vgg(input_tensor=None):

    input_shape = (None, None, None, 3)

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv3D(64, (3, 3, 3), activation='relu',
               padding='same', name='block1_conv1')(img_input)
    x = Conv3D(64, (3, 3, 3), activation='relu',
               padding='same', name='block1_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block1_pool')(x)
    print(x)

    # Block 2
    x = Conv3D(128, (3, 3, 3), activation='relu',
               padding='same', name='block2_conv1')(x)
    x = Conv3D(128, (3, 3, 3), activation='relu',
               padding='same', name='block2_conv2')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block2_pool')(x)
    print(x)

    # Block 3
    x = Conv3D(256, (3, 3, 3), activation='relu',
               padding='same', name='block3_conv1')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu',
               padding='same', name='block3_conv2')(x)
    x = Conv3D(256, (3, 3, 3), activation='relu',
               padding='same', name='block3_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(x)
    print(x)

    # Block 4
    x = Conv3D(512, (3, 3, 3), activation='relu',
               padding='same', name='block4_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu',
               padding='same', name='block4_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu',
               padding='same', name='block4_conv3')(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(x)
    print(x)

    # Block 5
    x = Conv3D(512, (3, 3, 3), activation='relu',
               padding='same', name='block5_conv1')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu',
               padding='same', name='block5_conv2')(x)
    x = Conv3D(512, (3, 3, 3), activation='relu',
               padding='same', name='block5_conv3')(x)

    return x

# RPN layer


def rpn_layer(base_layers, num_anchors):

    x = Conv3D(512, (3, 3, 3), padding='same', activation='relu')(base_layers)

    # classification layer: num_anchors (9) channels for 0, 1 sigmoid activation output
    x_class = Conv3D(num_anchors, (1, 1, 1), activation='sigmoid')(x)

    # regression layer: num_anchors*4 (36) channels for computing the regression of bboxes
    x_regr = Conv3D(num_anchors * 4, (1, 1, 1), activation='linear')(x)

    # classification of object(0 or 1), compute bounding boxes, base layers vgg
    return [x_class, x_regr, base_layers]
