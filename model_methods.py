
from pathlib import Path
from random import random
import pickle as pkl

import numpy as np
import os
import sys
import h5py
import random as rand

#import keras_tuner
import tensorflow as tf
from tensorflow.keras.layers import Layer,Masking,Reshape
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
            l_new = BatchNormalization(name=f"{name}_bnorm_{i}")(l_prev)
        if dropout_rate>0.0:
            l_new = Dropout(dropout_rate)(l_new)
        l_prev = l_new
    return l_prev

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

def basic_dense(name:str, node_list:list,
        window_feats:int, horizon_feats:int, static_feats:int, pred_feats:int,
        batchnorm=True, dropout_rate=0.0, dense_kwargs={}):
    """
    Dense layer next-step predictor model that simply appends window and
    horizon feats as the input, and outputs the prediction features for the
    next time step. The only reason there's a distinction between window and
    horizon features is to conform to the input tensor style used by others.
    """
    w_in = Input(shape=(1,window_feats,), name="in_window")
    h_in = Input(shape=(1,horizon_feats,), name="in_horizon")
    s_in = Input(shape=(static_feats,), name="in_static")
    mod_in = Reshape(target_shape=(1,static_feats))(s_in)
    all_in = Concatenate(axis=-1)([w_in,h_in,mod_in])
    dense = get_dense_stack(
            name=name,
            node_list=node_list,
            layer_input=all_in,
            dropout_rate=dropout_rate,
            batchnorm=batchnorm,
            dense_kwargs=dense_kwargs,
            )
    output = Dense(units=pred_feats, activation="linear", name="output")(dense)
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

def get_agg_loss(band_ratio, ceres_band_cutoff_idx=2):
    """
    Returns a loss function balancing flux and spatial features, where the
    spatial features are predicted with pixelwise MSE, and the flux features
    are averaged before being compared to the bulk CERES values
    (which are copied along the 2nd axis; identical for all sequence elements)

    Expects (B,S,F) shaped arrays for B batch samples, S sequence elements,
    and F features. The F features contain flux values up to the cutoff index
    for the 3rd (final) axis, then spatial values (ie dist, azimuth, etc).
    """
    @tf.function
    @tf.autograph.experimental.do_not_convert
    def agg_loss(y_true, y_pred):
        """ """
        t_space = y_true[:,:,ceres_band_cutoff_idx:]
        p_space = y_pred[:,:,ceres_band_cutoff_idx:]
        ## MSE independently for spatial and lw/sw predictions
        L_space = tf.math.reduce_mean(tf.square(t_space-p_space))

        ## Average all of the sequence outputs to get the prediction
        ## CERES bands are just copied along ax0
        t_bands = y_true[:,0,:ceres_band_cutoff_idx]
        ## Predicted bands should vary
        p_bands = y_pred[:,:,:ceres_band_cutoff_idx]
        ## Take the average of all model predictions in the footprint
        p_bands = tf.math.reduce_mean(p_bands, axis=1)
        L_bands = tf.math.reduce_mean(tf.square(t_bands-p_bands))

        return band_ratio*L_bands+(1-band_ratio)*L_space ## lstmed_2
    return agg_loss

def gen_hdf5_sample(
        h5_paths:list, static_data:np.array, window:int, horizon:int,
        window_feat_idxs:list, horizon_feat_idxs:list, pred_feat_idxs,
        seed:int=None, domain_mask:np.array=None):
    """
    Given a collection of hdf5 paths, gridded static inputs,

    :@param h5_paths: List of STRINGS to hdf5 files containing continuous
        equal-interval arrays shaped like (time, lat, lon, feature). Must be
        strings instead of pathlib Paths to convert to tensor.
    :@param static_data: (lat, lon, static_feature) shaped array of data which
        are consistent over time corresponding to each of the grid cells. Since
        the entire static array is provided alongside the arguments, it is
        assumed to already contain only all relevant features in order.
    :@param in_idxs: input feature indeces in the order they should be provided
        to the model
    """
    ## Must decode if f is a byte string. It's converted if casted as a tensor.
    h5_paths = [Path(f) if type(f)==str
            else Path(f.decode('ASCII'))
            for f in h5_paths]

    ## Open a mem map of hdf5 files with (time, lat, lon, feat) datasets
    assert all(f.exists() for f in h5_paths)
    feats = [h5py.File(f.as_posix(), "r")["/data/feats"] for f in h5_paths]
    ## All dataset shapes except the first dimension must be uniform shaped
    grid_shape = feats[0].shape[1:]
    assert all(s.shape[1:]==grid_shape for s in feats[1:])
    ## lat/lon components of static data shape must match those of the features
    assert static_data.shape[:2] == grid_shape[:2]
    ## If no domain mask is provided, assume the full grid has valid samples
    if domain_mask is None:
        domain_mask = np.full(grid_shape[:2], True)
    ## Domain mask must be (lat, lon) shaped, matching the feats & static data
    assert domain_mask.shape == grid_shape[:2]
    domain_list = list(zip(*np.where(domain_mask)))

    while True:
        ## Choose the hdf5 dataset to use
        h5_choice = feats[rand.randrange(len(feats))]
        ## Time index of the first predicted feats
        ## (time after last observed)
        time_idx = rand.randrange(window, h5_choice.shape[0]-horizon)
        ## 2D spatial index like (lat,lon)
        dy,dx = domain_list[rand.randrange(len(domain_list))]
        ## Extract the full range from the hdf, then split window/horizon
        XY = h5_choice[time_idx-window:time_idx+horizon,dy,dx,:]
        ## window is (window, window_feat_idxs) shaped
        X = {"window":XY[:window,window_feat_idxs]}
        ## horizon is (horizon, horizon_feat_idxs) shaped
        X["horizon"] = XY[window:,horizon_feat_idxs]
        ## static is (static_feats,) shaped
        X["static"] = static_data[dy,dx]
        Y = XY[window:,pred_feat_idxs]
        yield X,Y
