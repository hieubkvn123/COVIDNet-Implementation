import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import initializers as init
from tensorflow.keras.models import Model, Sequential

def get_pepx_module(H, W, in_channels, out_channels, name=None, batchnorm=False):
    inputs = Input(shape=(H, W, in_channels))

    # 1. First stage projection and expansion
    proj_1 = Conv2D(in_channels // 2, kernel_size=(1,1), kernel_initializer=init.HeNormal())(inputs)
    exp_1 = Conv2D(int(3 * in_channels / 4), kernel_size=(1,1), kernel_initializer=init.HeNormal())(proj_1)

    # 2. Depth-wise convolution
    dw_conv = DepthwiseConv2D(kernel_size=(3,3), padding='same', kernel_initializer=init.HeNormal())(exp_1)

    # 2. Second stage projection and extension
    proj_2 = Conv2D(in_channels // 2, kernel_size=(1,1), kernel_initializer=init.HeNormal())(dw_conv)
    ext_2 = Conv2D(out_channels, kernel_size=(1,1), kernel_initializer=init.HeNormal())(proj_2)

    # Output
    output = ext_2
    if(batchnorm):
        output = BatchNormalization()(ext_2)
    
    pepx_module = Model(inputs=inputs, outputs=output, name=name)

    return pepx_module


