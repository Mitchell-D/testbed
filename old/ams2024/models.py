import pickle as pkl
from pathlib import Path
import numpy as np
from datetime import datetime
from datetime import timedelta

# It's temporary thanks to warnings from the conda build of tensorflow I need
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Bidirectional
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import TimeDistributed, Flatten, RepeatVector
from tensorflow.keras.regularizers import L2
from tensorflow.keras import Input, Model

def direct_seq_to_seq(
        window_steps:int, horizon_steps:int, window_feature_count:int,
        horizon_feature_count:int, dist_nodes:int,
        enc_lstm_layers:list=[64,64], dec_lstm_layers:list=[64,64],
        bidirectional:bool=False, batch_normalize=True, dropout_rate=0,
        lstm_kwargs={}):
    """
    Direct sequence->sequence LSTM; in other words, encodes to a fixed-size
    vector, which is repeated along each forecast horizon and appended to
    "forecast" features (atmospheric forcings) before being decoded by another
    LSTM sequence.

    The model is direct as opposed to an autoencoder in the sense that it
    doesn't train a decoder head to reproduce the original sequence prior to
    generating a forecast.
    """
    enc_in = Input(shape=(window_steps, window_feature_count))
    fct_in = Input(shape=(horizon_steps, horizon_feature_count))

    prev_layer = enc_in
    for i in range(len(enc_lstm_layers)):
        # Don't return sequences for the last LSTM
        tmp_lstm = LSTM(
            units=enc_lstm_layers[i],
            return_sequences=(i<len(enc_lstm_layers)-1),
            **lstm_kwargs,
            )
        # Add bidirectional wrapper
        if bidirectional:
            tmp_lstm = Bidirectional(tmp_lstm)
        # Set the layer inputs
        tmp_lstm = tmp_lstm(prev_layer)
        # Add batch normalization and dropout if requested
        if batch_normalize:
            tmp_lstm = BatchNormalization()(tmp_lstm)
        if dropout_rate:
            tmp_lstm = Dropout(dropout_rate)(tmp_lstm)
        prev_layer = tmp_lstm

    r_vec = RepeatVector(horizon_steps)(prev_layer)
    concat = Concatenate(axis=2)([r_vec, fct_in])

    #'''
    prev_layer = concat
    for i in range(len(dec_lstm_layers)):
        # Return sequences for every LSTM layer since the "top-most" layer
        # outputs a prediction for each time step.
        tmp_lstm = LSTM(
            units=dec_lstm_layers[i],
            return_sequences=True,
            **lstm_kwargs,
            )
        # Add bidirectional wrapper
        if bidirectional:
            tmp_lstm = Bidirectional(tmp_lstm)
        # Set the layer inputs
        tmp_lstm = tmp_lstm(prev_layer)
        # Add batch normalization and dropout if requested
        if batch_normalize:
            tmp_lstm = BatchNormalization()(tmp_lstm)
        if dropout_rate:
            tmp_lstm = Dropout(dropout_rate)(tmp_lstm)
        prev_layer = tmp_lstm
    #'''

    # Apply a dense layer to each LSTM output using TimeDistributed
    tdist = TimeDistributed(
            Dense(dist_nodes, activation="linear")
            )(prev_layer)
    tdist = Flatten()(tdist)

    model = Model(inputs=[enc_in, fct_in], outputs=[tdist])
    return model

def one_shot_multi_horizon(
        window_steps:int, horizon_steps:int, feature_count:int,
        lstm_layers:list=[64,64], dist_nodes:int=16, dense_layers:list=[64,32],
        dist_activation="relu", dense_activation="relu",
        bidirectional:bool=False, batch_normalize=True, dropout_rate=0,
        lstm_kwargs={}):
    """
    Multi-layered LSTM, ultimately followed by a TimeDistributed layer
    """
    in_layer = Input(shape=(window_steps, feature_count))

    prev_layer = in_layer
    for i in range(len(lstm_layers)):
        tmp_lstm = LSTM(
            units=lstm_layers[i],
            return_sequences=True,
            **lstm_kwargs,
            )
        # Add bidirectional wrapper
        if bidirectional:
            tmp_lstm = Bidirectional(tmp_lstm)
        # Set the layer inputs
        tmp_lstm = tmp_lstm(prev_layer)
        # Add batch normalization and dropout if requested
        if batch_normalize:
            tmp_lstm = BatchNormalization()(tmp_lstm)
        if dropout_rate:
            tmp_lstm = Dropout(dropout_rate)(tmp_lstm)
        prev_layer = tmp_lstm

    # Apply a dense layer to each LSTM output using TimeDistributed
    tdist = TimeDistributed(
            Dense(dist_nodes, activation=dist_activation)
            )(prev_layer)
    # Flatten the output since dense layers can be 2d now (why?)
    tdist = Flatten()(tdist)

    # Add dense layers to decode the TimeDistributed output
    prev_layer = tdist
    for i in range(len(dense_layers)):
        tmp_dense = Dense(dense_layers[i], activation=dense_activation)
        tmp_dense = tmp_dense(prev_layer)
        if batch_normalize:
            tmp_dense = BatchNormalization()(tmp_dense)
        if dropout_rate:
            tmp_dense = Dropout(dropout_rate)(tmp_dense)
        prev_layer = tmp_dense

    out_layer = Dense(units=horizon_steps, activation="linear")(prev_layer)

    return Model(inputs=[in_layer], outputs=[out_layer],
                 name="one_shot_multi_horizon")

def lstm_static_bidir(
        window_size, feature_dims, static_dims, rec_reg_penalty=0.01,
        stat_reg_penalty=0.001, drop_rate=0.1):
    """
    :@return: Uncompiled Model object
    """
    r_input = Input(shape=(window_size, feature_dims), name="rec_in")
    s_input = Input(shape=(static_dims,), name="stat_in")

    # First bidirectional LSTM layer: output dims = 128 with dropout layer
    r_lstm_1 = LSTM(
            units=128,
            kernel_regularizer=L2(rec_reg_penalty),
            recurrent_regularizer=L2(rec_reg_penalty),
            return_sequences=True,
            )
    r_bd_1 = Bidirectional(r_lstm_1, name="bd_1")(r_input)
    r_1 = Dropout(drop_rate, name="drop_1")(r_bd_1)

    # Second bidirectional LSTM layer: output dims = 64
    r_lstm_2 = LSTM(
            units=64,
            kernel_regularizer=L2(rec_reg_penalty),
            recurrent_regularizer=L2(rec_reg_penalty)
            )
    r_bd_2 = Bidirectional(r_lstm_2, name="bd_2")(r_1)
    r_2 = Dropout(drop_rate, name="drop_2")(r_bd_2)

    # Dense static layer: output dims = 64
    s_1 = Dense(
            units=64,
            kernel_regularizer=L2(stat_reg_penalty),
            activation="relu",
            name="s_dense")(s_input)

    # Concatenation layer + 2 dense layers
    concat = Concatenate(axis=1, name="rs_concat")([r_2, s_1])
    combo_dense = Dense(units=64, activation="relu", name="combo")(concat)
    output = Dense(units=1, activation="linear", name="output")(combo_dense)

    model = Model(inputs=[r_input, s_input], outputs=[output])
    return model

def lstm_bidir_3(window_size, feature_dims, lstm_layers=[128,64,64],
                 batch_normalize:bool=False, dropout_rate:bool=None):
    """
    Any-depth bidirectional LSTM with batch normalization and no dropout
    or regularization layers
    """
    lstm_in = Input(shape=(window_size, feature_dims), name="input")
    lstm = None
    for i in range(len(lstm_layers)):
        lstm = Bidirectional(LSTM(
            units=lstm_layers[i],
            return_sequences = not (i == len(lstm_layers)-1),
            ), name=f"bdlstm_{i+1}")(lstm_in if i==0 else lstm)
        if batch_normalize:
            lstm = BatchNormalization()(lstm)
        if dropout_rate:
            lstm = Dropout(dropout_rate)(lstm)
    dense_1 = Dense(units=64, activation="relu", name="dense_1")(lstm)
    dense_1 = BatchNormalization()(dense_1) if batch_normalize else dense_1
    dense_2 = Dense(units=64, activation="relu", name="dense_2")(dense_1)
    output = Dense(units=1, activation="linear", name="output")(dense_2)
    return Model(inputs=[lstm_in], outputs=[output])

def basic_deep_lstm(window_size:int, feature_dims:int, output_dims:int,
                    batch_normalize:bool=False):
    """
    -> batch shape: (batch_size, window_size, feature_dims)
    -> output shape: (batch_size, output_dims)

    batch shape:
    Batch size is (timesteps - window_size) since the first "window"
    of features must be used to inform the next time step in training.

    :@param 1d_data_dict: Dictionary of 1D data following the standard of
        dictionaries generated by make_1d_dataset.py
    :@param window_size: Number of former timesteps of input features
        that are trained on for each additional prediction. To my
        understanding, periodic features with a frequency less than the time
        window won't be fully characterized unless the LSTM is stateful.
    :@param batch_normalize: if True, adds a BatchNormalization layer
    """
    nldas1D = Sequential()

    # First layer's output shape is (batch_size, window_size, hidden_size)
    # since return_sequences is True. This returns a 3D tensor containing an
    # abstract time series sequence for the next LSTM layer to process.
    nldas1D.add(LSTM(
        units=64,
        input_shape=(window_size, feature_dims),
        # Return sequences of input
        return_sequences=True,
        #activation="relu",
        ))
    if batch_normalize:
        nldas1D.add(BatchNormalization())
    # return_sequences set to False here to collapse a dimension
    nldas1D.add(LSTM(units=32, return_sequences=False))
    if batch_normalize:
        nldas1D.add(BatchNormalization())
    #nldas1D.add(Dense(units=8, activation='relu'))
    nldas1D.add(Dense(units=16))
    if batch_normalize:
        nldas1D.add(BatchNormalization())
    nldas1D.add(Dense(units=output_dims, activation='linear'))

    return nldas1D

if __name__=="__main__":
    model = one_shot_multi_horizon(
            window_steps=24,
            horizon_steps=12,
            feature_count=8,
            lstm_layers=[64,64],
            dist_nodes=16,
            #dense_layers=[64,32],
            dense_layers=[64,32],
            dist_activation="relu",
            dense_activation="relu",
            bidirectional=False,
            batch_normalize=True,
            dropout_rate=0,
            lstm_kwargs={}
            )
    '''

    model = direct_seq_to_seq(
            window_steps=24,
            horizon_steps=12,
            window_feature_count=8,
            horizon_feature_count=7,
            dist_nodes=1,
            enc_lstm_layers=[64,64,24],
            dec_lstm_layers=[64,64],
            bidirectional=False,
            batch_normalize=True,
            dropout_rate=0,
            lstm_kwargs={}
            )
    '''

    print(model.summary())
    exit(0)

    # First cycle only
    #training_pkl = Path("data/model_data/silty-loam_set1_training.pkl")
    #validation_pkl = Path("data/model_data/silty-loam_set1_validation.pkl")
    model_dir = Path("models/set004")

    # All cycles
    t_pkl = model_dir.joinpath("input/silty-loam_set4_training.pkl")
    v_pkl = model_dir.joinpath("input/silty-loam_set4_validation.pkl")
    s_pkl = model_dir.joinpath("input/silty-loam_set4_testing.pkl")

    #checkpoint_file = Path("data/model_check/set001")
    checkpoint_file = model_dir.joinpath("checkpoint")

    t_feat,t_truth,t_times = pkl.load(t_pkl.open("rb"))
    v_feat,v_truth,v_times = pkl.load(v_pkl.open("rb"))
    s_feat,s_truth,s_times = pkl.load(s_pkl.open("rb"))

    #'''
    # set1: 5 epochs, first cycle
    # set2: 200 epochs, first cycle
    # set3: 600 epochs, all 4 cycles
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.metrics import RootMeanSquaredError
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import load_model
    EPOCHS = 600
    model = basic_deep_lstm(
            window_size=48,
            feature_dims=t_feat.shape[2],
            output_dims=t_truth.shape[1],
            )
    check = ModelCheckpoint(checkpoint_file.as_posix(), save_best_only=True)
    model.compile(
            loss=MeanSquaredError(),
            optimizer=Adam(learning_rate=1e-4),
            metrics=[RootMeanSquaredError()],
            )
    model.fit(
            t_feat,
            t_truth,
            validation_data=(v_feat, v_truth),
            #epochs=30,
            epochs=EPOCHS,
            callbacks=[check],
            )
    #'''

    """
    Re-load the model and generate predictions for training, validation, and
    testing data.
    """
    print(f"Loading {checkpoint_file.as_posix()}")
    model = load_model(checkpoint_file.as_posix())
    model.compile(optimizer='adam')

    t_out = model.predict(t_feat)
    v_out = model.predict(v_feat)
    s_out = model.predict(s_feat)
    print("features:",t_feat.shape,v_feat.shape,s_feat.shape)
    print("outputs:",t_out.shape, v_out.shape, s_out.shape)
    pkl.dump((t_out, v_out, s_out),
             model_dir.joinpath("output/silty-loam_set003_out.pkl").open("wb"))
