"""
Contains model definitions for SqueezeNet, AlexNet, GoogleNet and VGGFace used in
Grm, K., et al. Strengths and Weaknesses of Deep Learning Models for Face 
Recognition Against Image Degradations. 
Published in: IET Biometrics, 2017.

Obtained from https://github.com/kgrm/face-recog-eval
"""
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.layers import AveragePooling2D, BatchNormalization, Dense, Lambda
from keras.layers import merge, Dropout, Input, Activation
from keras.regularizers import l2, l1
# from keras.utils.visualize_util import plot
from keras import backend as K
import h5py
import numpy as np
import sys, os
# from itertools import cycle

K.set_image_dim_ordering('th')

def load_existing_weights(filename, model):
    new_filename = filename
    layerdict = dict([(layer.name, layer) for layer in model.layers])
    f = h5py.File(new_filename)
    for layer in model.layers:
        name = layer.name
        if name in f.keys() and ("conv" in name or "Conv" in name or
                                 "fire" in name or "Dense" in name or
                                 "FC" in name):# or "prob" in name):
            filters = [np.array(val) for val in f[name].values()]
            if not len(filters):
                continue
            oldparams = layerdict[name].get_weights()
            for old, new in zip(oldparams, filters):
                if old.shape != new.shape:
                    print(old.shape, new.shape)
                assert old.shape == new.shape
            layerdict[name].set_weights(filters)
            print(name)
    if new_filename != filename:
        os.remove(new_filename)
    f.close()

def GELU(x):
    return x * K.sigmoid(1.704*x)


def fire_layer(x, filters_squeeze, filters_1x1, filters_3x3,
               weight_cost, name_prefix, act="relu", init="glorot_uniform"):
    squeeze = Convolution2D(filters_squeeze, 1, 1,
                            border_mode="same",
                            init=init,
                            activation=act,
                            W_regularizer=l1(weight_cost),
                            b_regularizer=l1(weight_cost),
                            name="%s_squeeze"%name_prefix)(x)
    expand_1x1 = Convolution2D(filters_1x1, 1, 1,
                            border_mode="same",
                            init=init,
                            activation=act,
                            W_regularizer=l1(weight_cost),
                            b_regularizer=l1(weight_cost),
                            name="%s_expand_1x1"%name_prefix)(squeeze)
    expand_3x3 = Convolution2D(filters_3x3, 3, 3,
                            border_mode="same",
                            init=init,
                            activation=act,
                            W_regularizer=l1(weight_cost),
                            b_regularizer=l1(weight_cost),
                            name="%s_expand_3x3"%name_prefix)(squeeze)
    if filters_3x3 == 0:
        return expand_1x1
    else:
        return merge([expand_1x1, expand_3x3],
                     mode="concat", 
                     concat_axis=1, 
                     name="%s_output"%name_prefix)

def conv1x1(N_filt, r, name):
    return Convolution2D(N_filt, 1, 1, border_mode="same", activation="relu",
                         W_regularizer=l2(r), name=name)

def squeezenet(N_classes, 
               inshape=(3, 224, 224), 
               r=1e-3, 
               simple_bypass=False, 
               p_dropout=0.5,
               output="prob",
               init="glorot_uniform",
               FC_width=2048,
               fire10_512=False,
               fire10_1024=False,
               fire11_1024=False):
    x = Input(shape=inshape)
    y = Convolution2D(96, 7, 7, 
                      border_mode="same",
                      activation="relu",
                      subsample=(2,2),
                      W_regularizer=l2(r),
                      b_regularizer=l2(r),
                      name="conv1",
                      init=init)(x)

    y = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode="same")(y)

    y = fire_layer(y, 16, 64 , 64 , r, "fire2", act="relu", init=init)
    y1 = fire_layer(y, 16, 64 , 64 , r, "fire3", act="relu", init=init)

    if simple_bypass:
        y = merge([y, y1], mode="sum", name="bypass1")
    else:
        y = y1
    y = fire_layer(y, 32, 128, 128, r, "fire4", act="relu", init=init)

    y = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode="same")(y)

    y1 = fire_layer(y, 32, 128, 128, r, "fire5", act="relu", init=init)

    if simple_bypass:
        y = merge([y, y1], mode="sum", name="bypass2")
    else:
        y = y1

    y = fire_layer(y, 48, 192, 192, r, "fire6", act="relu", init=init)
    y1 = fire_layer(y, 48, 192, 192, r, "fire7", act="relu", init=init)

    if simple_bypass:
        y = merge([y, y1], mode="sum", name="bypass3")
    else:
        y = y1

    y = fire_layer(y, 64, 256, 256, r, "fire8", act="relu", init=init)

    y = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode="same")(y)

    y1 = fire_layer(y, 64, 256, 256, r, "fire9", act="relu", init=init)

    if (simple_bypass and (not fire11_1024)):
        y = merge([y, y1], mode="sum", name="bypass4")
    else:
        y = y1

    if fire10_512:
        y = fire_layer(y, 64, 256, 256, r, "fire10", 
                       act="relu", init=init)
    elif fire10_1024:
        y = fire_layer(y, 128, 512, 512, r, "fire10",
                       act="relu", init=init)
    elif fire11_1024:
        y = fire_layer(y, 64, 256, 256, r, "fire10", "relu", init)
        if simple_bypass:
            y = merge([y, y1], mode="sum", name="bypass4")
        else:
            y = y1
        y = fire_layer(y, 128, 512, 512, r, "fire11", "relu", init)

    y = AveragePooling2D((14,14))(y)
    conv_feats = Flatten()(y)
    dense_feats = Dense(FC_width, activation="relu", init=init,
                       W_regularizer=l2(r), b_regularizer=l2(r),
                       name="FC1")(conv_feats)
    y = Dropout(p_dropout)(dense_feats)
    y = Dense(N_classes, activation="softmax", name="prob",
              W_regularizer=l2(r),
              b_regularizer=l2(r),
              init=init)(y)

    if output == "prob":
        m = Model(input=x, output=y)
    elif output == "convFeatures":
        m = Model(input=x, output=conv_feats)
    elif output == "denseFeatures":
        m = Model(input=x, output=dense_feats)
    else:
        m = 1
    return m

def Conv2D(c, h, w, strides=(1, 1),  act_fun="relu", border="same", r=1e-4):
    def f(input_):
        return Convolution2D(c, h, w, subsample=strides,
                             activation=act_fun, border_mode=border,
                             W_regularizer=l2(r), b_regularizer=l2(r))(input_)
    return f

def AlexNet(N_classes=1000, r=1e-4, p_dropout=0.5,  borders="same",
            inshape=(3, 224, 224), include_top=True, include_softmax=True):
    if type(borders) != list:
        borders = [borders] * 5
    x = Input(inshape)
    y = Conv2D(96, 11, 11, (4, 4), r=r, border=borders[0])(x)
    y = MaxPooling2D((3, 3), (2, 2))(y)
    y = BatchNormalization(axis=1)(y)
    y = Conv2D(256, 5, 5, r=r, border=borders[1])(y)
    y = MaxPooling2D((3, 3), (2, 2))(y)
    y = BatchNormalization(axis=1)(y)
    y = Conv2D(384, 3, 3, r=r, border=borders[2])(y)
    y = Conv2D(384, 3, 3, r=r, border=borders[3])(y)
    y = Conv2D(256, 3, 3, r=r, border=borders[4])(y)
    if include_top:
        y = MaxPooling2D((3, 3), (2, 2))(y)
        y = Flatten()(y)
        y = Dense(4096, activation="relu", 
                  W_regularizer=l2(r), b_regularizer=l2(r))(y)
        y = Dropout(p_dropout)(y)
        y = Dense(4096, activation="relu", 
                  W_regularizer=l2(r), b_regularizer=l2(r))(y)
        if include_softmax:
            y = Dropout(p_dropout)(y)
            y = Dense(N_classes, activation="softmax", W_regularizer=l2(r),
                      name="%d_way_softmax"%N_classes, b_regularizer=l2(r))(y)
    m = Model(x, y)
#    m.summary()
    m.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return m

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    '''Utility function to apply conv + BN.
    '''
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x


def InceptionV3(N_classes=1000, include_top=True):
    img_input = Input((3, 299, 299))
    channel_axis = 1

    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(i))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = merge([branch3x3_1, branch3x3_2],
                          mode='concat', concat_axis=channel_axis,
                          name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                             mode='concat', concat_axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(9 + i))
    
    x = AveragePooling2D((8, 8), strides=(8, 8), name="avg_pool")(x)
    x = Flatten(name="flatten")(x)
    if include_top:
        # Classification block
        x = Dense(N_classes, activation='softmax', 
                  name="%d_way_softmax"%N_classes)(x)

    m = Model(img_input, x, name='inception_v3')
#    m.summary()
    m.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return m

def vgg_face(weights_path=None, output_layer = "prob", N_classes = 2622, 
             r=0.0, p_dropout_fc6=0.5, p_dropout_fc7=None):
    if p_dropout_fc7 is None:
        p_dropout_fc7 = p_dropout_fc6

    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3',
                            W_regularizer=l2(r),
                            b_regularizer=l2(r))(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6',
                W_regularizer=l2(r), b_regularizer=l2(r))(flat)
    fc6_drop = Dropout(p_dropout_fc6)(fc6)
    fc7 = Dense(4096, activation='relu', name='fc7',
                W_regularizer=l2(r), b_regularizer=l2(r))(fc6_drop)
    fc7_drop = Dropout(p_dropout_fc7)(fc7)
    if N_classes == 2622:
        outname = "fc8"
    else:
        outname = "prob" + str(N_classes)
    out = Dense(N_classes, activation='softmax', name=outname,
                W_regularizer=l2(r), b_regularizer=l2(r))(fc7_drop)

    if output_layer == "prob":
        model = Model(input=img, output=out)
    elif output_layer == "fc6":
        model = Model(input=img, output=fc6, name="vgg-face")

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    model.compile("adam", "mse")

    return model

def randpredict(model, shape, scale):
    arr = np.random.rand(*shape) * scale
    ret = model.predict(arr)
    print(ret.dtype, ret.shape, ret.min(), ret.max())

if __name__ == "__main__":
    m1 = squeezenet(50, output="denseFeatures",
                       simple_bypass=True, fire11_1024=True)
    print('Loading 1')
    m1.load_weights("weights/luksface-weights.h5")
    m2 = AlexNet(N_classes=1000, r=1e-4, p_dropout=0.5,  borders="same",
                 inshape=(3, 224, 224), include_softmax=False)
    print('Loading 2')
    m2.load_weights("weights/alexnet_weights.h5", by_name=True)
    m3 = InceptionV3(include_top=False)
    print('Loading 3')
    m3.load_weights("weights/googlenet_weights.h5")
    m4 = vgg_face(output_layer="fc6")
    print('Loading 4')
    m4.load_weights("weights/vgg_face_weights.h5", by_name=True)
    print('Squeeze')
    randpredict(m1, (1, 3, 224, 224), 255)
    print('AlexNet')
    randpredict(m2, (1, 3, 224, 224), 255)
    print('GoogleNet')
    randpredict(m3, (1, 3, 299, 299), 255)
    print('VggFace ')
    randpredict(m4, (1, 3, 224, 224), 255)
