from pathlib import Path
from random import random
import pickle as pkl

import numpy as np
import os
import sys
import h5py
import json
import random as rand

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

def get_residual_loss_fn(residual_ratio:float=.5, use_mse:bool=False):
    """
    Function factory for residual-based sequence predictor loss functions.

    The label values are the true state values, and are expected to have an
    additional element prepended to the sequence axis (axis 2) equal to the
    last observed state vector in the window.

    The prediction values are the residual predictions, or in other words the
    increment change in state. These are accumulated into a state magnitude
    using the last observed state, and are also compared directly to the
    forward-difference increments of change from the labels.

    :@param residual_ratio: float value in [0,1] setting the balance between
        residual and state magnitude error for predictions such that
        error = ratio * residual_error + (1-ratio) * magnitude_error
    :@param use_mse: If True, use mean squared error for  both loss components
        instead of the default (mean absolute error).
    """
    if use_mse:
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    def residual_loss(YS,PR):
        """
        Loss for residual sequence predictions.

        :@param YS: (B,S+1,F) Label states including last observed
        :@param PR: (B,S,F) Predicted residuals
        """
        ## PS := predicted state evolved from last observed state
        PS = (YS[:,0,:][:,tf.newaxis,:] + tf.cumsum(PR, axis=1))

        ## YR := label residual
        YR = YS[:,1:]-YS[:,:-1]

        ## reduce to residual and state loss
        r_loss = loss_fn(YR, PR)
        s_loss = loss_fn(YS[:,1:,:], PS)

        return r_loss * residual_ratio + s_loss * (1-residual_ratio)

    return residual_loss

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
    prev_layer = get_lstm_stack(
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
    prev_layer = get_dense_stack(
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
        num_window_feats, num_horizon_feats, num_static_feats,
        num_static_int_feats, num_pred_feats,
        input_lstm_depth_nodes, output_lstm_depth_nodes,
        static_int_embed_size, input_linear_embed_size=None,
        bidirectional=True, batchnorm=True, dropout_rate=0.0,
        input_lstm_kwargs={}, output_lstm_kwargs={}, **kwargs):
    """
    Construct a sequence->sequence LSTM which encodes a vector from "window"
    features, then decodes it using "horizon" covariate variables into a
    predicted value.
    """
    w_in = Input(shape=(window_size,num_window_feats,), name="in_window")
    h_in = Input(shape=(horizon_size,num_horizon_feats,), name="in_horizon")
    s_in = Input(shape=(num_static_feats,), name="in_static")
    si_in = Input(shape=(num_static_int_feats,), name="in_static_int")

    ## Simple matrix mult to embed one-hot encoded static integers
    si_embedded = Dense(static_int_embed_size, use_bias=False)(si_in)

    ## Concatenate static vectors to each step of the input window
    s_window = RepeatVector(window_size)(s_in)
    si_window = RepeatVector(window_size)(si_embedded)
    window = Concatenate(axis=-1)([w_in,s_window,si_window])

    prev_layer = window
    if not input_linear_embed_size is None:
        prev_layer = TimeDistributed(Dense(input_dense_nodes))(prev_layer)

    ## Get a LSTM stack that accepts a (horizon,feats) sequence and outputs
    ## a single vector as well as each LSTM layer's final context states.
    prev_layer,enc_states,enc_contexts = get_lstm_stack(
            name="enc_lstm",
            layer_input=prev_layer,
            node_list=input_lstm_depth_nodes,
            return_seq=False,
            bidirectional=bidirectional,
            lstm_kwargs=input_lstm_kwargs,
            return_states=True,
            )

    ## Matrix-multiply encoder context to sizes needed to initialize decoder.
    ## In practice, the
    init_states = [
            Dense(
                output_lstm_depth_nodes[i],
                use_bias=False,
                name=f"scale_state_{i}",
                )(enc_states[i])
            for i in range(min(
                (len(enc_states), len(output_lstm_depth_nodes))
                ))
            ]

    init_contexts = [
            Dense(
                output_lstm_depth_nodes[i],
                use_bias=False,
                name=f"scale_context_{i}",
                )(enc_contexts[i])
            for i in range(min(
                (len(enc_contexts), len(output_lstm_depth_nodes))
                ))
            ]

    '''
    ## Probably-weaker alternative to initializing by carrying context
    ## Copy the input sequence encoded vector along the horizon axis and
    ## concatenate the vector with each of the horizon features
    #enc_copy_shape = (horizon_size,input_lstm_depth_nodes[-1])
    prev_layer = RepeatVector(horizon_size)(prev_layer)
    prev_layer = Concatenate(axis=-1)([h_in,prev_layer])
    '''

    s_horizon = RepeatVector(horizon_size)(s_in)
    si_horizon = RepeatVector(horizon_size)(si_embedded)
    horizon = Concatenate(axis=-1)([h_in,s_horizon,si_horizon])

    prev_layer = horizon
    prev_layer = get_lstm_stack(
            name="dec_lstm",
            layer_input=horizon,
            node_list=output_lstm_depth_nodes,
            return_seq=True,
            bidirectional=bidirectional,
            initial_states=list(zip(init_states, init_contexts)),
            lstm_kwargs=output_lstm_kwargs,
            return_states=False,
            )
    inputs = (w_in, h_in, s_in, si_in)
    output = TimeDistributed(Dense(num_pred_feats))(prev_layer)
    return Model(inputs=inputs, outputs=[output])

def get_lstm_stack(name:str, layer_input:Layer, node_list:list, return_seq,
                   bidirectional=False, batchnorm=True, dropout_rate=0.0,
                   return_states=False, initial_states=[], lstm_kwargs={}):
    """
    Returns a Layer object after adding a LSTM sequence stack

    :@param name: Unique string name identifying this entire LSTM stack.
    :@param layer_input: layer recieved by this LSTM stack. Typically
        expected to have axes like (batch, sequence, features).
    :@param node_list: List of integers describing the number of nodes in
        subsequent layers starting from the stack input, for example
        [32,64,64,128] would map inputsequences with 32 elements
    :@param return_seq: If True, returns output from each cell. Otherwise,
        only the final output vector will be returned.
    :@param bidirectional: If True, separate LSTM cells will learn by passing
        over the sequence in each direction.
    :@param batchnorm: True for batch normalization between LSTM layers
    :@param dropout_rate: Dropout rate between LSTM cells
    :@param return_states: If True, each LSTM layer's context state is
        returned along with the output state.
    :@param initial_states: List of initial context vectors corresponding to
        the LSTM layers. If fewer initial states are provided than layers
        initialized, the deeper layers will be zero-initialized.
    :@param lstm_kwargs: Keyword arguments passed to each LSTM layer on init.
    """
    lstm_kwargs = {**def_lstm_kwargs.copy(), **lstm_kwargs}
    l_prev = layer_input
    output_states = []
    context_states = []
    for i in range(len(node_list)):
        ## Intermediate LSTM layers must always return sequences in order
        ## to stack; this is only False if return_seq is False AND the
        ## current LSTM layer is the LAST one.
        rseq = (not (i==len(node_list)-1), True)[return_seq]
        tmp_lstm = LSTM(
                units=node_list[i],
                return_sequences=rseq,
                return_state=True,
                name=f"{name}_lstm_{i}",
                **lstm_kwargs,
                )
        ## Add a bidirectional wrapper if requested
        if bidirectional:
            tmp_lstm = Bidirectional(
                    tmp_lstm, name=f"{name}_bd_{i}")
        if len(initial_states) > i:
            l_new,tmp_output,tmp_context  = tmp_lstm(
                    l_prev, initial_state=initial_states[i])
        else:
            l_new,tmp_output,tmp_context = tmp_lstm(l_prev)
        output_states.append(tmp_output)
        context_states.append(tmp_context)
        if batchnorm:
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_new)
        ## Typically dropout is best after batch norm
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new

    if return_states:
        return l_prev,output_states,context_states
    else:
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

if __name__=="__main__":
    pass
