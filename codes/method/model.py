from tensorflow.python.keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, MaxPool2D, ReLU, \
    Conv3D, Conv3DTranspose, MaxPool3D, Activation, BatchNormalization, Subtract, UpSampling3D, AveragePooling3D, \
    Dense

from tensorflow.python.keras.models import Model
import tensorflow as tf
import numpy as np

#####################################
############## 3D Unet ##############
#####################################
def unet_3d(input_shape,
            output_channel,
            kernel_size=3,
            filters_root=32,
            conv_times=3,
            up_down_times=4,
            if_relu=False,
            if_residule = False):

    def conv3d_relu_dropout(input_, filters_, kernel_size_, name):
        output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, padding='same', name=name+'/Conv3D')(input_)
        output_ = ReLU(name=name+'/Activation')(output_)
        return output_

    def conv3d_transpose_relu_dropout(input_, filters_, kernel_size_, name):
        output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(2, 2, 1),
                                                            name=name+'/Conv3DTranspose')(input_)
        output_ = ReLU(name=name+'/Activation')(output_)
        return output_

    skip_connection = []
    ipt = Input(input_shape, name='UNet3D/Keras_Input')
    net = conv3d_relu_dropout(ipt, filters_root, kernel_size, name='UNet3D/InputConv')

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size, name='UNet3D/Down_%d/ConvLayer_%d' % (layer, i))
        skip_connection.append(net)
        net = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), name='UNet3D/Down_%d/MaxPool3D' % layer)(net)

    filters = 2 ** up_down_times * filters_root
    for i in range(0, conv_times):
        net = conv3d_relu_dropout(net, filters, kernel_size, name='UNet3D/Bottom/ConvLayer_%d' % i)

    for layer in range(up_down_times - 1, -1, -1):
        filters = 2 ** layer * filters_root
        net = conv3d_transpose_relu_dropout(net, filters, kernel_size, name='UNet3D/Up_%d/UpSample' % layer)
        net = Concatenate(axis=-1, name='UNet3D/Up_%d/SkipConnection' % layer)([net, skip_connection[layer]])
        for i in range(0, conv_times):
            net = conv3d_relu_dropout(net, filters, kernel_size, name='UNet3D/Up_%d/ConvLayer_%d' % (layer, i))

    net = Conv3D(filters=output_channel, kernel_size=1, padding='same', name='UNet3D/OutputConv')(net)
    # extra laryer
    if if_relu:
        net = ReLU(name='UNet3D/extra_Activation')(net)
    if if_residule:
        net = Subtract()([ipt, net])

    return Model(inputs=ipt, outputs=net)

#####################################
############## 2D Unet ##############
#####################################
def unet_2d(input_shape,
            output_channel,
            kernel_size=3,
            filters_root=32,
            conv_times=3,
            up_down_times=4,
            if_relu = False,
            if_residule = False):

    def conv2d_relu_dropout(input_, filters_, kernel_size_, name):
        output_ = Conv2D(filters=filters_, kernel_size=kernel_size_, padding='same', name=name+'/Conv2D')(input_)
        output_ = ReLU(name=name+'/Activation')(output_)
        return output_

    def conv2d_transpose_relu_dropout(input_, filters_, kernel_size_, name):
        output_ = Conv2DTranspose(filters=filters_, kernel_size=kernel_size_, padding='same', strides=(2, 2),
                                  name=name+'/Conv2Transpose')(input_)
        output_ = ReLU(name=name+'/Activation')(output_)
        return output_

    skip_connection = []
    ipt = Input(input_shape, name='UNet2D/Keras_Input')
    net = conv2d_relu_dropout(ipt, filters_root, kernel_size, name='UNet2D/InputConv')

    for layer in range(up_down_times):
        filters = 2 ** layer * filters_root
        for i in range(0, conv_times):
            net = conv2d_relu_dropout(net, filters, kernel_size, name='UNet2D/Down_%d/ConvLayer_%d' % (layer, i))

        skip_connection.append(net)
        net = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='UNet2D/Down_%d/MaxPool3D' % layer)(net)

    filters = 2 ** up_down_times * filters_root
    for i in range(0, conv_times):
        net = conv2d_relu_dropout(net, filters, kernel_size, name='UNet2D/Bottom/ConvLayer_%d' % i)

    for layer in range(up_down_times - 1, -1, -1):
        filters = 2 ** layer * filters_root
        net = conv2d_transpose_relu_dropout(net, filters, kernel_size, name='UNet2D/Up_%d/UpSample' % layer)
        net = Concatenate(axis=-1, name='UNet2D/Up_%d/SkipConnection' % layer)([net, skip_connection[layer]])

        for i in range(0, conv_times):
            net = conv2d_relu_dropout(net, filters, kernel_size, name='UNet2D/Up_%d/ConvLayer_%d' % (layer, i))

    net = Conv2D(filters=output_channel, kernel_size=1, padding='same', name='UNet2D/OutputConv')(net)
    # extra laryer
    if if_relu:
        net = ReLU(name='UNet2D/extra_Activation')(net)
    if if_residule:
        net = Subtract()([ipt, net])

    return Model(inputs=ipt, outputs=net)

