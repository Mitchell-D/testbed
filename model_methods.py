from pathlib import Path
from random import random
import pickle as pkl

import numpy as np
import os
import sys
import h5py
import json
import random as rand
from list_feats import dynamic_coeffs,static_coeffs

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking,Reshape,ReLU,Conv1D,Add
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

def_lstm_kwargs = {
            ## output activation
            "activation":"tanh",
            ## cell state activation
            "recurrent_activation":"sigmoid",
            ## initial activation for 'previous output'
            "kernel_initializer":"glorot_uniform",
            ## initial activation for  'previous cell state'
            "recurrent_initializer":"orthogonal",
            "kernel_regularizer":None,
            ## Between-cell cell state dropout rate
            "dropout":0.0,
            ## Between-cell previous output dropout rate
            "recurrent_dropout":0.0,
            }

def_dense_kwargs = {
        "activation":"sigmoid",
        "use_bias":True,
        "bias_initializer":"zeros",
        "kernel_initializer":"glorot_uniform",
        "kernel_regularizer":None,
        "bias_regularizer":None,
        "activity_regularizer":None,
        "kernel_constraint":None,
        "bias_constraint":None,
        }

def get_dense_stack(name:str, layer_input:Layer, node_list:list,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Simple stack of dense layers with optional dropout and batchnorm
    """
    dense_kwargs = {**def_dense_kwargs.copy(), **dense_kwargs}
    l_prev = layer_input
    for i in range(len(node_list)):
        l_new = Dense(
                units=node_list[i],
                **dense_kwargs,
                name=f"{name}_dense_{i}"
                )(l_prev)
        if batchnorm:
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_new)
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new
    return l_prev

def get_lstm_rec(window_size, num_window_feats, num_horizon_feats,
        num_static_feats, num_pred_feats, input_lstm_depth_nodes,
        output_dense_nodes, input_dense_nodes=None, bidirectional=True,
        batchnorm=True, dropout_rate=0.0, lstm_kwargs={}, dense_kwargs={}):
    """
    Sequence -> Vector network with a LSTM window encoder and a dense layer
    stack for next-step prediction
    """
    w_in = Input(shape=(window_size,num_window_feats,), name="in_window")
    h_in = Input(shape=(1,num_horizon_feats,), name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    s_seq = RepeatVector(window_size)(s_in)
    seq_in = Concatenate(axis=-1)([w_in,s_seq])

    prev_layer = seq_in
    if not input_dense_nodes is None:
        prev_layer = TimeDistributed(Dense(input_dense_nodes))(prev_layer)

    ## Get a LSTM stack that accepts a (horizon,feats) sequence and outputs
    ## a single vector
    prev_layer = mm.get_lstm_stack(
            name="enc_lstm",
            layer_input=prev_layer,
            node_list=input_lstm_depth_nodes,
            return_seq=False,
            bidirectional=bidirectional,
            lstm_kwargs=lstm_kwargs,
            )
    ## Concatenate the encoder output with the horizon data
    prev_layer = Concatenate(axis=-1)([
        prev_layer, Reshape(target_shape=(num_horizon_feats,))(h_in)
        ])
    prev_layer = mm.get_dense_stack(
            name="dec_dense",
            node_list=output_dense_nodes,
            layer_input=prev_layer,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            dense_kwargs=dense_kwargs,
            )

    inputs = {"window":w_in,"horizon":h_in,"static":s_in}
    ## Reshape output to match the data tensor
    output = Reshape(target_shape=(1,num_pred_feats))(
            Dense(num_pred_feats)(prev_layer))
    return Model(inputs=inputs, outputs=[output])

def get_lstm_s2s(window_size, horizon_size,
        num_window_feats, num_horizon_feats, num_static_feats, num_pred_feats,
        input_lstm_depth_nodes, output_lstm_depth_nodes,
        input_dense_nodes=None, bidirectional=True, batchnorm=True,
        dropout_rate=0.0, input_lstm_kwargs={}, output_lstm_kwargs={}):
    """
    Construct a sequence->sequence LSTM which encodes a vector from "window"
    features, then decodes it using "horizon" covariate variables into a
    predicted value.
    """
    w_in = Input(shape=(window_size,num_window_feats,), name="in_window")
    h_in = Input(shape=(horizon_size,num_horizon_feats,), name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    s_seq = RepeatVector(window_size)(s_in)
    seq_in = Concatenate(axis=-1)([w_in,s_seq])

    prev_layer = seq_in
    if not input_dense_nodes is None:
        prev_layer = TimeDistributed(Dense(input_dense_nodes))(prev_layer)

    ## Get a LSTM stack that accepts a (horizon,feats) sequence and outputs
    ## a single vector
    prev_layer = mm.get_lstm_stack(
            name="enc_lstm",
            layer_input=prev_layer,
            node_list=input_lstm_depth_nodes,
            return_seq=False,
            bidirectional=bidirectional,
            lstm_kwargs=input_lstm_kwargs,
            )

    ## Copy the input sequence encoded vector along the horizon axis and
    ## concatenate the vector with each of the horizon features
    #enc_copy_shape = (horizon_size,input_lstm_depth_nodes[-1])
    prev_layer = RepeatVector(horizon_size)(prev_layer)
    prev_layer = Concatenate(axis=-1)([h_in,prev_layer])

    prev_layer = mm.get_lstm_stack(
            name="dec_lstm",
            layer_input=prev_layer,
            node_list=output_lstm_depth_nodes,
            return_seq=True,
            bidirectional=bidirectional,
            lstm_kwargs=output_lstm_kwargs,
            )
    inputs = {"window":w_in,"horizon":h_in,"static":s_in}
    output = TimeDistributed(Dense(num_pred_feats))(prev_layer)
    return Model(inputs=inputs, outputs=[output])

def get_lstm_stack(name:str, layer_input:Layer, node_list:list, return_seq,
                   bidirectional:False, batchnorm=True, dropout_rate=0.0,
                   lstm_kwargs={}):
    """
    Returns a Layer object after adding a LSTM sequence stack

    :@param name: Unique string name identifying this entire LSTM stack.
    :@param layer_input: layer recieved by this LSTM stack. Typically
        expected to have axes like (batch, sequence, features).
    :@param node_list: List of integers describing the number of nodes in
        subsequent layers starting from the stack input, for example
        [32,64,64,128] would map inputsequences with 32 elements
    """
    lstm_kwargs = {**def_lstm_kwargs.copy(), **lstm_kwargs}
    l_prev = layer_input
    for i in range(len(node_list)):
        ## Intermediate LSTM layers must always return sequences in order
        ## to stack; this is only False if return_seq is False and the
        ## current LSTM layer is the LAST one.
        rseq = (not (i==len(node_list)-1), True)[return_seq]
        tmp_lstm = LSTM(units=node_list[i], return_sequences=rseq,
                        **lstm_kwargs, name=f"{name}_lstm_{i}")
        ## Add a bidirectional wrapper if requested
        if bidirectional:
            tmp_lstm = Bidirectional(
                    tmp_lstm, name=f"{name}_bd_{i}")
        l_new = tmp_lstm(l_prev)
        if batchnorm:
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_new)
        ## Typically dropout is best after batch norm
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new
    return l_prev

def get_tcn_enc(layer_input:Layer, dilation_rate,
        num_filters=3, kernel_size=4):
    """
    Get a single TCN module per https://arxiv.org/pdf/1906.04397.pdf
    """
    ## Module includes two dilated convolution layers
    prev = ReLU()(BatchNormalization()(Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            )(layer_input)))
    prev = BatchNormalization()(Conv1D(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            )(prev))
    ## Ensure that the feature dimensions match for the residual connection
    if layer_input.shape[-1] != prev.shape[-1]:
        layer_input = Conv1D(
                filters=num_filters,
                kernel_size=1,
                padding="same",
                )(layer_input)
    output = Add()([layer_input, prev])
    return ReLU()(output)

def temporal_convolution(window_and_horizon_size, num_window_feats,
        num_horizon_feats, num_static_feats, num_pred_feats, dense_units,
        dilation_layers, num_filters=3, kernel_size=4):
    """ """
    w_in = Input(
            shape=(window_and_horizon_size,num_window_feats),
            name="in_window")
    h_in = Input(
            shape=(window_and_horizon_size,num_horizon_feats),
            name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    s_seq = RepeatVector(window_and_horizon_size)(s_in)
    seq_in = Concatenate(axis=-1)([w_in,s_seq])

    enc_prev = seq_in
    for dilation in dilation_layers:
        enc_prev = get_tcn_enc(
                layer_input=enc_prev,
                dilation_rate=dilation,
                num_filters=num_filters,
                kernel_size=kernel_size,
                )

    dec_prev = ReLU()(BatchNormalization()(Dense(dense_units)(h_in)))
    dec_prev = BatchNormalization()(Dense(dense_units)(dec_prev))

    if dec_prev.shape[-1] != enc_prev.shape[-1]:
        enc_prev = Dense(dense_units)(enc_prev)

    output = ReLU()(Add()([enc_prev, dec_prev]))
    output = Dense(units=num_pred_feats)(output)
    #output = Reshape((window_and_horizon_size, num_pred_feats))(output)

    inputs = {"window":w_in,"horizon":h_in,"static":s_in}
    model = Model(inputs=inputs, outputs=[output])
    return model


def basic_dense(name:str, node_list:list, num_window_feats:int,
        num_horizon_feats:int, num_static_feats:int, num_pred_feats:int,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Dense layer next-step predictor model that simply appends window and
    horizon feats as the input, and outputs the prediction features for the
    next time step. The only reason there's a distinction between window and
    horizon features is to conform to the input tensor style used by others.
    """
    w_in = Input(shape=(1,num_window_feats,), name="in_window")
    h_in = Input(shape=(1,num_horizon_feats,), name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    mod_in = Reshape(target_shape=(1,num_static_feats))(s_in)
    all_in = Concatenate(axis=-1)([w_in,h_in,mod_in])
    dense = get_dense_stack(
            name=name,
            node_list=node_list,
            layer_input=all_in,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dense_kwargs,
            )
    output = Dense(units=num_pred_feats,
            activation="linear",name="output")(dense)
    inputs = {"window":w_in,"horizon":h_in,"static":s_in}
    model = Model(inputs=inputs, outputs=[output])
    return model

def basic_lstmae(
        seq_len:int, feat_len:int, enc_nodes:list, dec_nodes:list, latent:int,
        latent_activation="sigmoid", dropout_rate=0.0, batchnorm=True,
        mask_val=None, bidirectional=True, enc_lstm_kwargs={},
        dec_lstm_kwargs={}):
    """
    Basic LSTM sequence encoder/decoder with optional masking

    Inputs to a seq->seq model like this one are generally assumed to
    be shaped like (N, P, F) for N sequence samples, P points in each sequence,
    and F features per point.

    :@param seq_len: Size of sequence element dimension of input (2nd dim)
    :@param feat_len: Size of feature dimension of input tensor (3rd dim)
    :@param enc_nodes: List of integers corresponding to the cell state and
        hidden state size of the corresponding LSTM layers.
    :@param dec_nodes: Same as above, but for the decoder.
    :@param enc_lstm_kwargs: arguments passed directly to the LSTM layer on
        initialization; use this to change activation, regularization, etc.
    """
    ## Fill any default arguments with the user-provided ones
    enc_lstm_kwargs = {**def_lstm_kwargs.copy(), **enc_lstm_kwargs}
    dec_lstm_kwargs = {**def_lstm_kwargs.copy(), **dec_lstm_kwargs}

    ## Input is like (None, sequence size, feature count)
    l_seq_in = Input(shape=(seq_len, feat_len))

    ## Add a masking layer if a masking value is set
    l_prev = l_seq_in if mask_val is None \
            else Masking(mask_value=mask_val)(l_seq_in)

    ## Do a pixel-wise projection up to the LSTM input dimension.
    ## This seems like a semi-common practice before sequence input,
    ## especially for word embeddings.
    l_prev = TimeDistributed(
            Dense(enc_nodes[0], name="in_projection"),
            name="in_dist"
            )(l_prev)

    ## Add the encoder's LSTM layers
    l_enc_stack = get_lstm_stack(
            name="enc",
            layer_input=l_prev,
            node_list=enc_nodes,
            return_seq=False,
            bidirectional=bidirectional,
            batchnorm=batchnorm,
            lstm_kwargs=dec_lstm_kwargs,
            dropout_rate=dropout_rate
            )

    ## Encode to the latent vector
    l_enc_out = Dense(latent, activation=latent_activation,
                      name="latent_projection")(l_enc_stack)

    ## Copy the latent vector along the output sequence
    l_dec_in = RepeatVector(seq_len)(l_enc_out)

    ## Add decoder's LSTM layers
    l_dec_stack = get_lstm_stack(
            name="dec",
            layer_input=l_dec_in,
            node_list=dec_nodes,
            return_seq=True,
            bidirectional=bidirectional,
            batchnorm=batchnorm,
            lstm_kwargs=dec_lstm_kwargs,
            )

    ## Uniform transform from LSTM output to pixel distribution
    l_dist = Dense(feat_len, activation="linear", name="out_projection")
    l_dec_out = TimeDistributed(l_dist, name="out_dist")(l_dec_stack)

    ## Get instances of Model objects for each autoencoder component.
    ## Each instance correspond to the same weights per:
    ## https://keras.io/api/models/model/
    full = Model(l_seq_in, l_dec_out)
    #encoder = Model(l_seq_in, l_enc_out)
    #decoder = Model(l_enc_out, l_dec_out)
    return full#, encoder, decoder

def lstm_decoder(encoder, seq_len, feat_len, dec_nodes,
        dropout_rate=0.0, bidirectional=False, batchnorm=True,
        dec_lstm_kwargs={}):
    """ Extends an encoder with a new stacked lstm decoder """
    seq_in = RepeatVector(seq_len)(encoder.output)
    dec_stack = get_lstm_stack(
            name="dec",
            layer_input=seq_in,
            node_list=dec_nodes,
            return_seq=True,
            bidirectional=bidirectional,
            lstm_kwargs=dec_lstm_kwargs,
            dropout_rate=dropout_rate,
            )
    dec_dist = Dense(feat_len, activation="linear", name="out_projection")
    dec_out = TimeDistributed(dec_dist, name="out_dist")(dec_stack)
    return Model(encoder.input, dec_out)

if __name__=="__main__":
    sample_dir = Path("/rstor/mdodson/thesis")
    h5s_val = [sample_dir.joinpath("shuffle_2018.h5").as_posix()]
    h5s_train = [sample_dir.joinpath(f"shuffle_{y}.h5").as_posix()
        for y in [2015,2019,2021]]
    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200"]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "ncrain", "cape", "pevap", "apcp", "dswrf"]
    pred_feats = ['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200']
    static_feats = ["pct_sand", "pct_silt", "pct_clay", "elev", "elev_std"]

    g = gen_sample(
            h5_paths=h5s_train,
            window_size=24,
            horizon_size=24,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            as_tensor=False,
            )

    #for i in range(10000):
    #    x,y = next(g)
    #    print([x[k].shape for k in x.keys()], y.shape)
    gT,gV = get_sample_generator(
            train_h5s=h5s_train,
            val_h5s=h5s_val,
            window_size=24,
            horizon_size=24,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            )
    batches = [next(g) for i in range(2048)]
    pkl.dump(batches, Path("data/sample/batch_samples.pkl").open("wb"))
