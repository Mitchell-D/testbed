import os
from pathlib import Path
import pickle as pkl

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from models import lstm_static_bidir, basic_deep_lstm, lstm_bidir_3

""" Sanity check tensorflow options to stdout """
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print(f"Tensorflow version: {tf.__version__}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())

""" Load training and validation data from the pkl """
#data_dict = pkl.load(Path("/rstor/mdodson/lstm-1.pkl").open("rb"))
#data_dict = pkl.load(Path("/rstor/mdodson/lstm-2.pkl").open("rb"))
data_dict = pkl.load(Path("/rstor/mdodson/lstm-3.pkl").open("rb"))

csv_path = Path("/rhome/mdodson/lstm-3/lstm-3_prog.csv")
checkpoint_path = Path(
    "/rhome/mdodson/lstm-3/lstm-3_{epoch:02d}_{val_loss:.2f}.hdf5")
model_path = Path("/rhome/mdodson/lstm-3/lstm-3_model.keras")

t_feats, t_static, t_truth = data_dict["training"]
v_feats, v_static, v_truth = data_dict["validation"]
#print(t_feats.shape, t_static.shape, t_truth.shape)

'''
lstm = lstm_static_bidir(
        window_size=t_feats.shape[1],
        feature_dims=t_feats.shape[2],
        static_dims=t_static.shape[1]
        )
'''
'''
lstm = basic_deep_lstm(
        window_size=t_feats.shape[1],
        feature_dims=t_feats.shape[2],
        output_dims=1
        )
'''
lstm = lstm_bidir_3(
        window_size=t_feats.shape[1],
        feature_dims=t_feats.shape[2],
        batch_normalize=True,
        )

lstm.compile(
        loss="mean_squared_error",
        optimizer="adam",
        #metrics=["val_loss"]
        )

hist = lstm.fit(
        #x=[t_feats, t_static],
        x=t_feats,
        y=t_truth,
        epochs=600,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=4),
            CSVLogger(csv_path.as_posix()),
            ModelCheckpoint(
                monitor="val_loss",
                save_best_only=True,
                filepath=checkpoint_path,
                )
            ],
        #validation_data=([v_feats, v_static], v_truth),
        validation_data=(v_feats, v_truth),
        verbose=2,
        )

lstm.save(model_path)
