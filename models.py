import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras import initializers as init
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential

def get_pepx_module(H, W, in_channels, out_channels, name=None, batchnorm=False, regularizer='l2', lambda_=1e-5):
    inputs = Input(shape=(H, W, in_channels))

    # Initialize regularizer
    reg = None
    if(regularizer == 'l2'):
        reg = regularizers.l2(l2=lambda_)
    else:
        reg = regularizers.l1(l1=lambda_)

    # 1. First stage projection and expansion
    proj_1 = Conv2D(in_channels // 2, kernel_size=(1,1), kernel_initializer=init.HeNormal(), kernel_regularizer=reg)(inputs)
    exp_1 = Conv2D(int(3 * in_channels / 4), kernel_size=(1,1), kernel_initializer=init.HeNormal(), kernel_regularizer=reg)(proj_1)

    # 2. Depth-wise convolution
    dw_conv = DepthwiseConv2D(kernel_size=(3,3), padding='same', kernel_initializer=init.HeNormal(), kernel_regularizer=reg)(exp_1)

    # 2. Second stage projection and extension
    proj_2 = Conv2D(in_channels // 2, kernel_size=(1,1), kernel_initializer=init.HeNormal(), kernel_regularizer=reg)(dw_conv)
    ext_2 = Conv2D(out_channels, kernel_size=(1,1), kernel_initializer=init.HeNormal(), kernel_regularizer=reg)(proj_2)

    # Output
    output = ext_2
    if(batchnorm):
        output = BatchNormalization()(ext_2)
    
    pepx_module = Model(inputs=inputs, outputs=output, name=name)

    return pepx_module

def get_small_covid_net(H, W, C, n_classes=2, batchnorm=True, reg=None):
    inputs = Input(shape=(H, W, C))

    # First conv layer
    conv1 = Conv2D(56, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer=init.HeNormal())(inputs)
    relu1 = ReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(relu1)

    # First residual connection
    out_conv1_1x1 = Conv2D(56, kernel_size=(1,1), kernel_initializer=init.HeNormal())(pool1)

    # Pepx stage 1
    pepx11 = get_pepx_module(H//4, W//4, 56, 56, name='pepx_11', batchnorm=batchnorm, regularizer=reg)(pool1)
    pepx12 = get_pepx_module(H//4, W//4, 56, 56, name='pepx_12', batchnorm=batchnorm, regularizer=reg)(pepx11 + out_conv1_1x1)
    pepx13 = get_pepx_module(H//4, W//4, 56, 56, name='pepx_13', batchnorm=batchnorm, regularizer=reg)(pepx11 + pepx12 + out_conv1_1x1)

    # Second residual connection
    out_conv2_1x1 = Conv2D(112, kernel_size=(1,1), kernel_initializer=init.HeNormal())(pepx11 + pepx12 + pepx13 + out_conv1_1x1)
    out_conv2_1x1 = MaxPooling2D(pool_size=(2, 2))(out_conv2_1x1)

    # Pepx stage 2
    pepx21 = get_pepx_module(H//8, W//8, 56, 112, name='pepx_21', batchnorm=batchnorm, regularizer=reg)(
        MaxPooling2D((2,2))(pepx11) + MaxPooling2D((2,2))(pepx12) + MaxPooling2D((2,2))(pepx13) + MaxPooling2D((2,2))(out_conv1_1x1)    
    )
    pepx22 = get_pepx_module(H//8, W//8, 112, 112, name='pepx_22', batchnorm=batchnorm, regularizer=reg)(pepx21 + out_conv2_1x1)
    pepx23 = get_pepx_module(H//8, W//8, 112, 112, name='pepx_23', batchnorm=batchnorm, regularizer=reg)(pepx21 + pepx22 + out_conv2_1x1)
    pepx24 = get_pepx_module(H//8, W//8, 112, 112, name='pepx_24', batchnorm=batchnorm, regularizer=reg)(pepx21 + pepx22 + pepx23 + out_conv2_1x1)


    # Third residual connection
    out_conv3_1x1 = Conv2D(216, kernel_size=(1,1), kernel_initializer=init.HeNormal())(pepx21 + pepx22 + pepx23 + pepx24 + out_conv2_1x1)
    out_conv3_1x1 = MaxPooling2D(pool_size=(2,2))(out_conv3_1x1)

    # Pepx stage 3
    pepx31 = get_pepx_module(H//16, W//16, 112, 216, name='pepx_31', batchnorm=batchnorm, regularizer=reg)(
        MaxPooling2D((2,2))(pepx21) + MaxPooling2D((2,2))(pepx22) + MaxPooling2D((2,2))(pepx23) + MaxPooling2D((2,2))(pepx24) + MaxPooling2D((2,2))(out_conv2_1x1)
    )
    pepx32 = get_pepx_module(H//16, W//16, 216, 216, name='pepx_32', batchnorm=batchnorm, regularizer=reg)(pepx31 + out_conv3_1x1)
    pepx33 = get_pepx_module(H//16, W//16, 216, 216, name='pepx_33', batchnorm=batchnorm, regularizer=reg)(pepx31 + pepx32 + out_conv3_1x1)
    pepx34 = get_pepx_module(H//16, W//16, 216, 216, name='pepx_34', batchnorm=batchnorm, regularizer=reg)(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
    pepx35 = get_pepx_module(H//16, W//16, 216, 216, name='pepx_35', batchnorm=batchnorm, regularizer=reg)(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)
    pepx36 = get_pepx_module(H//16, W//16, 216, 216, name='pepx_36', batchnorm=batchnorm, regularizer=reg)(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)

    # Fourth residual connection
    out_conv4_1x1 = Conv2D(424, kernel_size=(1,1), kernel_initializer=init.HeNormal())(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1)
    out_conv4_1x1 = MaxPooling2D(pool_size=(2,2))(out_conv4_1x1)

    # Pepx stage 4
    pepx41 = get_pepx_module(H//32, W//32, 216, 424, name='pepx_41', batchnorm=batchnorm, regularizer=reg)(
        MaxPooling2D((2,2))(pepx31) + MaxPooling2D((2,2))(pepx32) + MaxPooling2D((2,2))(pepx33) + 
        MaxPooling2D((2,2))(pepx34) + MaxPooling2D((2,2))(pepx35) + MaxPooling2D((2,2))(pepx36) + MaxPooling2D((2,2))(out_conv3_1x1)
    )
    pepx42 = get_pepx_module(H//32, W//32, 424, 424, name='pepx_42', batchnorm=batchnorm, regularizer=reg)(pepx41 + out_conv4_1x1)
    pepx43 = get_pepx_module(H//32, W//32, 424, 424, name='pepx_43', batchnorm=batchnorm, regularizer=reg)(pepx41 + pepx42 + out_conv4_1x1)
    
    # Classification head
    flatten = Flatten()(pepx41 + pepx42 + pepx43 + out_conv4_1x1)
    logits = ReLU()(Dense(512, kernel_regularizer=regularizers.l2(l2=1e-4))(flatten))
    outputs = Dense(n_classes)(logits)

    model = Model(inputs=inputs, outputs=[logits, outputs], name='COVID_Net_Small')

    return model
