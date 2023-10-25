import pickle as pkl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

def plot_forecast():
    return

def plot_keras_loss(csv_path:Path):
    """ Plot the training and validation loss from keras CSVLogger output """
    _,tloss,vloss = zip(*map(
        lambda l:map(float,l.split(",")),
        csv_path.open("r").readlines()[1:]))
    plt.plot(range(len(tloss)), tloss)
    plt.plot(range(len(vloss)), vloss)
    plt.show()
    return

def sample_from_data(X, Y, model, count:int=1):
    """
    Runs the model on a random set of samples from the provided datasets,
    and returns a list of 3-tuples like (sIDX, sX, sP) for sample s,
    where sIDX is the batch (first dimension) index of s, sX is the input
    sequence, and sP is the predicted sequence.
    """
    samples = []
    idxs = [b for b in np.random.choice(X.shape[0], size=count)]
    tmp_X = X[idxs]
    tmp_P = model(tmp_X)
    #print(X.shape, X[b].shape)
    #samples.append((b, X[b], model.predict(X[b], verbose=0)))
    #samples.append((b, X[b], model(X[b])))
    #return samples
    return tuple(zip(idxs, tmp_X, tmp_P))

def plot_keras_prediction(prior, truth, prediction, times=None,
                          title:str="", ymean=0, ystdev=1):
    """
    """
    fig,ax = plt.subplots()
    #print(X.shape, Y.shape, P.shape, len(times))
    rescale = lambda d:d*ystdev+ymean
    prior, truth, prediction = map(rescale, (prior, truth, prediction))
    ax.plot(range(len(prior)),
            prior, color="blue")
    ax.plot(range(len(prior), len(prior)+len(truth)),
            truth, color="blue")
    ax.plot(range(len(prior),len(prior)+len(prediction)),
            prediction, color="red")
    #ax.plot(times[:prior.shape[0]], prior, color="blue")
    #ax.plot(times[-truth.shape[0]:], truth, color="blue")
    #ax.plot(times[-prediction.shape[0]:], prediction, color="red")
    ax.set_xticklabels(times, rotation=25)
    plt.ylabel("0-10cm Soil Moisture (kg/m^2)")
    plt.title(title)
    plt.show()

def get_mae(model, X, Y):
    return np.sum(np.abs(model.predict(X)-Y))/X.shape[0]

def get_rmse(model, X, Y):
    return (np.sum((model.predict(X)-Y)**2)/X.shape[0])**(1/2)

# TEMPORARY!! this function shouldn't be stored here. Need to find a way
# to cleanly import it from krttdkit or testbed on matrix.
# check out tf.keras.saving.custom_object_scope
def distance_loss_fn(y_true, y_pred):
    """
    Custom loss function that weights error by the forecast horizon distance
    """
    gain = tf.range(1,tf.keras.backend.int_shape(y_pred)[-1]+1,delta=1,
            dtype=tf.float32)
    return tf.math.reduce_mean(tf.math.divide(
        tf.math.square(y_true-y_pred), gain))

if __name__=="__main__":
    '''
    data_path = Path("data/models/osmh_1/osmh_1_curated.pkl")
    model_path = Path("data/models/osmh_1/run_1/osmh_1_model.keras")
    csv_path = Path("data/models/osmh_1/run_1/osmh_1_prog.csv")
    '''
    data_path = Path("data/models/osmh_1/osmh_1_curated.pkl")
    #model_path = Path("data/models/osmh_1/run_1/osmh_1_model.keras")
    #model_path = Path("data/models/osmh_1/run_2/osmh_1_64_0.01.hdf5")
    #model_path = Path("data/models/osmh_1/run_3/osmh_1_56_0.01.hdf5")
    #model_path = Path("data/models/osmh_1/run_1/osmh_1_37_0.00.hdf5")
    #model_path = Path("data/models/osmh_1/run_4/osmh_1_model.keras")
    model_path = Path("data/models/osmh_1/run_4/osmh_1_50_0.00.hdf5")
    #csv_path = Path("data/models/osmh_1/run_1/osmh_1_prog.csv")
    csv_path = Path("data/models/osmh_1/run_4/osmh_1_prog.csv")

    model = load_model(
            model_path,
            custom_objects={
                "distance_loss_fn":distance_loss_fn,
                }
            )
    data_dict = pkl.load(data_path.open("rb"))

    train_or_validate = "validate"
    X = data_dict[train_or_validate]["X"]
    Y = data_dict[train_or_validate]["Y"]
    times = data_dict[train_or_validate]["t"]

    ymean = data_dict["means"][-1]
    ystdev = data_dict["stdevs"][-1]

    mae = get_mae(model, X, np.squeeze(Y)) * ystdev
    #plot_keras_loss(csv_path)
    print(f"{train_or_validate} MAE: {mae}")
    #print(f"{train_or_validate} RMSE: {get_rmse(model, X, np.squeeze(Y))}")
    exit(0)

    samples = sample_from_data(X, Y, model, count=5)
    for tmp_idx,tmp_X,tmp_P in samples:
        tmp_Y = Y[tmp_idx]
        # Use the window and horizon counts to determine time ranges from the
        # prediction time encoded by times (last non-forecast step)
        tmp_times = [t.strftime("%Y-%m-%d %H") for t in
                     times[tmp_idx-tmp_X.shape[0]:tmp_idx+tmp_Y.shape[0]]]
        plot_keras_prediction(
                prior=tmp_X[:,-1], # Only take the soil moisture output
                truth=np.squeeze(tmp_Y),
                prediction=tmp_P,
                times=tmp_times,
                ymean=ymean,
                ystdev=ystdev,
                #title=f"0-10cm osmh1 training r1 w24 h12 idx{tmp_idx}"
                title=f"0-10cm osmh1 validation r3 w24 h12 idx{tmp_idx}"
                )
