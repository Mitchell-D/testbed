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

def get_seq_paths(sequence_h5_dir:Path,
        region_strs:list=[], season_strs:list=[], time_strs:list=[]):
    """
    Constrain sequences hdf5 files by their underscore-separated fields
    """
    return tuple([
        str(p) for p in sequence_h5_dir.iterdir()
        if len(p.stem.split("_")) == 4 and all((
            p.stem.split("_")[0] == "sequences",
            p.stem.split("_")[1] in region_strs,
            p.stem.split("_")[2] in season_strs,
            p.stem.split("_")[3] in time_strs,
            ))
        ])

def get_cyclical_lr(lr_min=1e-5, lr_max=1e-2, inc_epochs=5, dec_epochs=5,
        decay:float=0, log_scale=True):
    """
    Returns a cyclical loss function that varies from the minimum to the
    maximum rate at a period described by the increase and decrease intervals

    :@param lr_min: Minimum learning rate value when decay is zero
    :@param lr_max: Maximum learning rate when decay is zero
    :@param inc_epochs: Number of epochs between minimum and maxium LR
    :@param dec_epochs: Nummber of epochs between maximum and minimum LR
    :@param decay: amplitude decay rate denominating the oscillation
    :@param log_scale: When True, the intermediate learning rates between the
        minimum and maximum learning rates
    """
    if log_scale:
        inc = np.logspace(np.log10(lr_min), np.log10(lr_max), num=inc_epochs)
        dec = np.logspace(np.log10(lr_max), np.log10(lr_min), num=dec_epochs)
    else:
        inc = np.linspace(lr_min, lr_max, num=inc_epochs)
        dec = np.linspace(lr_max, lr_min, num=dec_epochs)
    period = np.concatenate((dec, inc))

    def _cyclical_lr(epoch, cur_lr):
        return period[epoch % period.size] / (1 + decay*epoch)

    return _cyclical_lr

def get_snow_loss_fn(zero_point:float, use_mse=False,
        residual_norm:list=None, residual_magnitude_bias:float=None,
        zero_point_epsilon=.0001):
    """
    Residual-only loss function that ignores error in negative residual
    predictions when the true residual is equal to zero and

    :@param zero_point: Value associated with zero state after normalization;
        typically  - mean / stddev  for gaussian normalization.
    :@param use_mse: If True, use mean squared error rather than mean absolute
    :@param residual_norm:
    """
    if use_mse:
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    if residual_norm is None:
        residual_norm = 1.
    residual_norm = tf.convert_to_tensor(residual_norm, dtype=tf.float32)
    if residual_magnitude_bias is None:
        residual_magnitude_bias = 0.

    zero_point = tf.convert_to_tensor(zero_point)
    zero_point_epsilon = tf.convert_to_tensor(zero_point_epsilon)

    def snow_residual_loss(YS,PR):
        """
        Loss for residual sequence predictions.

        :@param YS: (B,S+1,F) Label states including last observed
        :@param PR: (B,S,F) Predicted residuals
        """
        ## YR := label residual
        YR = YS[:,1:]-YS[:,:-1]

        ## Don't penalize predictions that are less than zero when the true
        ## state and residual are zero, so that negative guesses during no-snow
        ## cases aren't penalized. This way, the network is more flexible and
        ## post-processing can ignore invalid predictions while accumulating.
        m_zero_res = (YR == 0)
        m_zero_state = (YS[:,1:,:] - zero_point) < zero_point_epsilon
        m_pred_lt_zero = PR < 0
        snow_weight = tf.where(
                tf.math.reduce_all(tf.logical_and(tf.logical_and(
                    m_zero_res, m_zero_state), m_pred_lt_zero), axis=-1),
                0., 1.)

        ## Develop sample-wise residual magnitude biases
        mag_bias = (1. + residual_magnitude_bias * tf.math.abs(YR))
        mag_bias = tf.math.reduce_sum(mag_bias/residual_norm, axis=-1)
        r_loss = loss_fn(
                y_true=YR/residual_norm,
                y_pred=PR/residual_norm,
                sample_weight=mag_bias * snow_weight,
                )
        return r_loss
    return snow_residual_loss

def get_residual_loss_fn(residual_ratio:float=.5, use_mse:bool=False,
        residual_norm:list=None, residual_magnitude_bias:float=None):
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
    :@param use_mse: If True, use mean squared error for both loss components
        instead of the default (mean absolute error).
    :@param residual_norm: List of normalization coefficients equal in length
        to the number of predicted features, which divide their respective
        residual outputs in order to normalize their magnitude.
    :@param residual_magnitude_bias: Most residuals will be close to zero, so
        returning a residual near zero is often a "safer" solution than risking
        a steep spike. This constant value exacerbates the penalty proportional
        to the magnitude of the true residual, so that residual error in the
        event of heavy precipitation or rapid drydown is more severe.
    """
    assert 0 <= residual_ratio <= 1
    if use_mse:
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        loss_fn = tf.keras.losses.MeanAbsoluteError()

    if residual_norm is None:
        residual_norm = 1.
    residual_norm = tf.convert_to_tensor(residual_norm, dtype=tf.float32)
    if residual_magnitude_bias is None:
        residual_magnitude_bias = 0.

    def residual_loss(YS,PR):
        """
        Loss for residual sequence predictions.

        :@param YS: (B,S+1,F) Label states including last observed
        :@param PR: (B,S,F) Predicted residuals
        """
        ## PS := predicted state evolved from last observed state
        PS = (YS[:,0,:][:,tf.newaxis,:] + tf.cumsum(PR, axis=1))
        s_loss = loss_fn(YS[:,1:,:], PS)

        ## YR := label residual
        YR = YS[:,1:]-YS[:,:-1]

        ## Develop sample-wise residual magnitude biases
        mag_bias = (1. + residual_magnitude_bias * tf.math.abs(YR))
        mag_bias = tf.math.reduce_sum(mag_bias/residual_norm, axis=-1)
        r_loss = loss_fn(
                y_true=YR/residual_norm,
                y_pred=PR/residual_norm,
                sample_weight=mag_bias,
                )

        ## Ratio balance between residual and state loss
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

def get_lstm_s2s(window_size, horizon_size,
        num_window_feats, num_horizon_feats, num_static_feats,
        num_static_int_feats, num_pred_feats,
        input_lstm_depth_nodes, output_lstm_depth_nodes,
        static_int_embed_size, input_linear_embed_size=None, pred_coarseness=1,
        bidirectional=False, batchnorm=True, dropout_rate=0.0,
        bias_state_rescale=False, input_lstm_kwargs={}, output_lstm_kwargs={},
        _horizon_input_projection=True, **kwargs):
    """
    Construct a sequence->sequence LSTM which encodes a vector from "window"
    features, then decodes it using "horizon" covariate variables into a
    predicted value.

    :@param input_linear_embed_size: Inputs are initially passed through a
        simple learned affine "embedding" to cast them as latent vectors with
        the provided size before being passed through LSTM sequence.
    :@param pred_coarseness: The LSTM sequence model and subsequent predictions
        may be made at a coarser output resolution according to this factor.
        Coarsening is implemented with a 1D convolution over the embedded input
        vectors, using input_linear_input_size convolution filters. This has
        the effect of aggregating each non-overlapping group of pred_coarseness
        input vectors into a shorter sequence of combined inputs having
        input_linear_embed_size elements prior to being passed thru the LSTMs.
    :@param bidirectional: If True, use bidirectional LSTMs to learn in both
        directions, doubling the node count of LSTM layers. Currently, this is
        broken when using encoder states to initialize LSTM sequences
    :@param batchnorm: If True, use batch normalization after each LSTM layer.
    :@param dropout_rate: If nonzero, use dropout after each LSTM layer.
    :@param bias_state_rescale: When True, the fully-connected layer that
        rescales window sequence state vectors to the initial state vectors of
        the horizon lstm will include a bias term.
    :@param *_lstm_kwargs: More keyword arguments provided to every LSTM chain.
    :@parma _horizon_input_projection: During experimentation, lstm 1-16 were
        unintentionally trained without projecting the horizon inputs to the
        requested size. As a contingency so that Model objects can be re-built
        with this method, this parameter was added. From now on, it should
        remain True.
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
        prev_layer = TimeDistributed(
                Dense(input_linear_embed_size)
                )(prev_layer)

    ## Get a LSTM stack that accepts a (horizon,feats) sequence and outputs
    ## a single vector as well as each LSTM layer's final context states.
    prev_layer,enc_states,enc_contexts = get_lstm_stack(
            name="enc_lstm",
            layer_input=prev_layer,
            node_list=input_lstm_depth_nodes,
            return_seq=False,
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            bidirectional=bidirectional,
            lstm_kwargs=input_lstm_kwargs,
            return_states=True,
            )

    ## Matrix-multiply encoder context to sizes needed to initialize decoder.
    init_states = [
            Dense(
                output_lstm_depth_nodes[i],
                use_bias=bias_state_rescale,
                name=f"scale_state_{i}",
                )(enc_states[i])
            for i in range(min(
                (len(enc_states), len(output_lstm_depth_nodes))
                ))
            ]

    init_contexts = [
            Dense(
                output_lstm_depth_nodes[i],
                use_bias=bias_state_rescale,
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
    if not input_linear_embed_size is None and _horizon_input_projection:
        prev_layer = TimeDistributed(
                Dense(input_linear_embed_size)
                )(prev_layer)

    ## If output coarsening is requested, implement it with a 1D convolution
    if pred_coarseness > 1:
        if input_linear_embed_size is None:
            input_linear_embed_size = output_lstm_depth_nodes[0]
        prev_layer = Conv1D(
                input_linear_embed_size,
                kernel_size=pred_coarseness,
                strides=pred_coarseness,
                padding="valid",
                name="coarsen",
                )(prev_layer)
    prev_layer = get_lstm_stack(
            name="dec_lstm",
            layer_input=prev_layer,
            node_list=output_lstm_depth_nodes,
            return_seq=True,
            bidirectional=bidirectional,
            initial_states=list(zip(init_states, init_contexts)),
            batchnorm=batchnorm,
            dropout_rate=dropout_rate,
            lstm_kwargs=output_lstm_kwargs,
            return_states=False,
            )
    inputs = (w_in, h_in, s_in, si_in)
    ## simple affine reprojection of outputs
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
            tmp = tmp_lstm(l_prev)
            if not len(tmp)==3:
                raise ValueError(
                    f"LSTM return length {len(tmp)} tuple expecting length 3."
                    "This may be because the LSTM is bidirectional")
            l_new,tmp_output,tmp_context = tmp
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
