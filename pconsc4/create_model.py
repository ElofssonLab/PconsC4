from __future__ import division
import os
from collections import namedtuple

import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dropout, BatchNormalization
from keras.layers import Conv2D, Conv1D, MaxPooling2D, UpSampling2D
from keras.layers.core import Lambda
from keras.layers.advanced_activations import ELU

DROPOUT = 0.1
ACTIVATION = ELU
INIT = "he_normal"
REG = None

PconsC4 = namedtuple('PconsC4', ['contact_model', 'ss_model'])


def self_outer(x):
    outer_x = x[:, :, None, :] * x[:, None, :, :]
    return outer_x


def add_1D_conv(model, filters, kernel_size, padding="same",
                kernel_initializer=INIT, kernel_regularizer=REG):
    model = Conv1D(filters, kernel_size, padding=padding,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer)(model)
    model = ACTIVATION()(model)
    model = BatchNormalization()(model)
    model = Dropout(DROPOUT)(model)
    return model


def add_2D_conv(model, filters, kernel_size, data_format="channels_last",
                padding="same", depthwise_initializer=INIT,
                depthwise_regularizer=REG, dropout=True):
    model = Conv2D(filters, kernel_size, data_format=data_format,
                   padding=padding,
                   kernel_initializer=depthwise_initializer,
                   kernel_regularizer=depthwise_regularizer)(model)

    model = ACTIVATION()(model)
    model = BatchNormalization()(model)
    if dropout:
        model = Dropout(DROPOUT)(model)
    return model



def _add_binary_head(model, dist_cutoff, kernel_size):
    out_binary = Conv2D(1, kernel_size, activation="sigmoid", data_format="channels_last", padding="same",
                        kernel_initializer=INIT, kernel_regularizer=REG, name="out_binary_%s" % dist_cutoff)(model)
    return out_binary


def _wrap_model(model, binary_cutoffs):
    # inputs for sequence features
    inputs_seq = [Input(shape=(None, 22), dtype=K.floatx(), name="seq"),
                  # sequence
                  Input(shape=(None, 23), dtype=K.floatx(), name="self_info"),
                  # self-information
                  Input(shape=(None, 23), dtype=K.floatx(),
                        name="part_entr")]  # partial entropy

    bottleneck_model, ss_model = create_ss_model()
    bottleneck = bottleneck_model(inputs_seq)

    # input for 2D features
    inputs_2D = [Input(shape=(None, None, 1), name="gdca", dtype=K.floatx()),
                 # gdca
                 Input(shape=(None, None, 1), name="mi_corr", dtype=K.floatx()),
                 # mi_corr
                 Input(shape=(None, None, 1), name="nmi_corr",
                       dtype=K.floatx()),  # nmi_corr
                 Input(shape=(None, None, 1), name="cross_h",
                       dtype=K.floatx())]  # cross_h

    # input for masking missing residues
    input_mask = Input(shape=(None, None, 1), name="mask")

    out_lst = model(inputs_2D + [bottleneck])
    out_mask_lst = []
    out_names = ["out_sscore_mask"] + ["out_binary_%s_mask" % d for d in
                                       binary_cutoffs]

    for i, out in enumerate(out_lst):
        out = keras.layers.Multiply(name=out_names[i])([out, input_mask])
        out_mask_lst.append(out)

    wrapped_model = Model(inputs=inputs_2D + inputs_seq + [input_mask],
                          outputs=out_mask_lst)
    wrapped_model.trainable = False

    return wrapped_model, ss_model


def create_ss_model():
    """Load the 1D sequence-based model

    This is safe because there are no Lambda layers, hence no dependency on Python's bytecode.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = "models/ss_pred_resnet_elu_nolr_dropout01_l26_large_v3_saved_model.h5"

    ss_model = load_model(os.path.join(base_path, model_path))

    inputs_seq_ = [Input(shape=(None, 22), dtype=K.floatx()),  # sequence
                   Input(shape=(None, 23), dtype=K.floatx()),
                   # self-information
                   Input(shape=(None, 23), dtype=K.floatx())]  # partial entropy

    try:
        seq_feature_model = ss_model.layers_by_depth[5][0]
    except:
        seq_feature_model = ss_model._layers_by_depth[5][0]
    assert 'model' in seq_feature_model.name, seq_feature_model.name
    seq_feature_model.name = 'sequence_features'

    bottleneck_seq = seq_feature_model(inputs_seq_)
    bottleneck_model = Model(inputs=inputs_seq_, outputs=bottleneck_seq)
    return bottleneck_model, ss_model


def create_unet(filters=64, binary_cutoffs=()):
    # Create 1D sequence model

    bottleneck_seq = Input(shape=(None, 128), dtype=K.floatx())
    model_1D_outer = Lambda(self_outer)(bottleneck_seq)
    model_1D_outer = BatchNormalization()(model_1D_outer)

    inputs_2D = [Input(shape=(None, None, 1), dtype=K.floatx()),  # plm/gdca
                 Input(shape=(None, None, 1), dtype=K.floatx()),  # mi_corr
                 Input(shape=(None, None, 1), dtype=K.floatx()),  # nmi_corr
                 Input(shape=(None, None, 1), dtype=K.floatx())]  # cross_h

    unet = keras.layers.concatenate(inputs_2D + [model_1D_outer])

    unet = add_2D_conv(unet, filters, 1)
    unet = add_2D_conv(unet, filters, 3)
    unet = add_2D_conv(unet, filters, 3)
    link1 = unet
    unet = MaxPooling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 2, 3)
    unet = add_2D_conv(unet, filters * 2, 3)
    link2 = unet
    unet = MaxPooling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 4, 3)
    unet = add_2D_conv(unet, filters * 4, 3)
    link3 = unet
    unet = MaxPooling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 8, 3)
    unet = add_2D_conv(unet, filters * 8, 3)
    link4 = unet
    unet = MaxPooling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 16, 3)
    unet = add_2D_conv(unet, filters * 16, 3)

    unet = UpSampling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 8, 2)
    unet = keras.layers.concatenate([unet, link4])
    unet = add_2D_conv(unet, filters * 8, 3)
    unet = add_2D_conv(unet, filters * 8, 3)
    unet = UpSampling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 4, 2)
    unet = keras.layers.concatenate([unet, link3])
    unet = add_2D_conv(unet, filters * 4, 3)
    unet = add_2D_conv(unet, filters * 4, 3)
    unet = UpSampling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters * 2, 2)
    unet = keras.layers.concatenate([unet, link2])
    unet = add_2D_conv(unet, filters * 2, 3)
    unet = add_2D_conv(unet, filters * 2, 3)
    unet = UpSampling2D(data_format="channels_last")(unet)
    unet = add_2D_conv(unet, filters, 2)
    unet = keras.layers.concatenate([unet, link1])
    split = unet
    unet = add_2D_conv(unet, filters, 3)
    unet = add_2D_conv(unet, filters, 3)

    out_binary_lst = []
    if binary_cutoffs:
        for d in binary_cutoffs:
            out_binary_lst.append(_add_binary_head(unet, d, 7))

    unet = add_2D_conv(split, filters, 3)
    unet = add_2D_conv(unet, filters, 3)

    out_sscore = Conv2D(1, 7, activation="sigmoid", data_format="channels_last",
                        padding="same",
                        kernel_initializer=INIT, kernel_regularizer=REG,
                        name="out_sscore")(unet)

    model = Model(inputs=inputs_2D + [bottleneck_seq],
                  outputs=[out_sscore] + out_binary_lst)
    model.trainable = False
    return model


def get_pconsc4(version=0):
    """Get the PconsC4 architecture.

    Version:
        0: first model
        -1: latest
    """
    base_path = os.path.dirname(os.path.abspath(__file__))

    binary_cutoffs = [6, 8, 10]
    unet = create_unet(binary_cutoffs=binary_cutoffs)

    # Pick the right model
    if version == -1:
        version = 0

    if version == 0:
        model_path = 'models/submodel_v0_weights.h5'
    else:
        raise(ValueError('Unkown version number {}'.format(version)))

    unet.load_weights(os.path.join(base_path, model_path), by_name=True)

    contact_model, ss_model = _wrap_model(unet, binary_cutoffs)

    return PconsC4(contact_model=contact_model, ss_model=ss_model)


if __name__ == '__main__':
    m = get_pconsc4()
    m.contact_model.summary()
