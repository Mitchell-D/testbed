import pickle as pkl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

import model_methods as mm

from tensorflow.keras.models import load_model

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

def load_config(model_dir):
    """
    Load the configuration JSON associated contained in a model directory
    """
    model_name = model_dir.name
    return json.load(model_dir.joinpath(f"{model_name}_config.json").open("r"))

def get_generator(sample_h5s:Path, model_dir, as_tensor=False):
    """
    Returns a data generator for the provided sample h5 according to the model
    configuration associated with model_dir. The provided h5 must conform to
    the format used by curate_samples.py
    """
    cfg = load_config(model_dir)
    return mm.gen_sample(
            h5_paths=list(sample_h5s),
            window_size=cfg["window_size"],
            horizon_size=cfg["horizon_size"],
            window_feats=cfg["window_feats"],
            horizon_feats=cfg["horizon_feats"],
            pred_feats=cfg["pred_feats"],
            static_feats=cfg["static_feats"],
            as_tensor=False,
            )

def load_csv_prog(model_dir):
    """
    Load the per-epoch metrics from a tensorflow CSVLogger file as a dict.
    """
    cfg = load_config(model_dir)
    csv_path = model_dir.joinpath(f"{cfg['model_name']}_prog.csv")
    csv_lines = csv_path.open("r").readlines()
    csv_lines = list(map(lambda l:l.strip().split(","), csv_lines))
    csv_labels = csv_lines.pop(0)
    csv_cols = list(map(
        lambda l:np.asarray([float(v) for v in l]),
        zip(*csv_lines)))
    return dict(zip(csv_labels, csv_cols))

def plot_csv_prog(model_dir, fields=None, show=False, save_path=None,
                  plot_spec={}):
    cfg = load_config(model_dir)
    csv = load_csv_prog(model_dir)
    epoch = csv["epoch"]
    del csv["epoch"]
    if fields is None:
        fields = list(csv.keys())
    fig,ax = plt.subplots()
    for f in fields:
        ax.plot(epoch,csv[f],label=f,)
    ax.set_title(plot_spec.get("title"))
    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    plt.legend()
    if show:
        plt.show()
    if not save_path is None:
        plt.savefig(save_path, bbox_inches="tight")

def plot_keras_prediction(prior, truth, prediction, times=None,
                          title:str="", ymean=0, ystdev=1):
    """ """
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

def model_evaluate(model_path, data_gen, num_samples=1000):
    """ """
    model = load_model(model_path)
    W,H,S,T = [],[],[],[]
    for i in range(num_samples):
        X,Y = next(data_gen)
        W.append(X["window"])
        H.append(X["horizon"])
        S.append(X["static"])
        T.append(Y)
    return model({
        "window":np.stack(W, axis=0),
        "horizon":np.stack(H, axis=0),
        "static":np.stack(S, axis=0)
        }), np.stack(T)

def model_roll(model_path, sample_h5s, horizon_size=12, num_samples=1000):
    """ """
    model = load_model(model_path)
    cfg = load_config(model_path.parent)
    ## Make sure the last features in the window are the predicted ones
    assert tuple(cfg["window_feats"][-len(cfg["pred_feats"]):]) \
            == tuple(cfg["pred_feats"])
    ## Get a sample generator for the full requested horizon size
    sample_gen = mm.gen_sample(
            h5_paths=list(sample_h5s),
            window_size=cfg["window_size"],
            horizon_size=horizon_size,
            window_feats=cfg["window_feats"],
            horizon_feats=cfg["horizon_feats"],
            pred_feats=cfg["pred_feats"],
            static_feats=cfg["static_feats"],
            as_tensor=False,
            )
    ## Consolidate all the sample components
    W,H,S,P,T = [],[],[],[],[]
    for i in range(num_samples):
        X,Y = next(sample_gen)
        W.append(X["window"])
        H.append(X["horizon"])
        S.append(X["static"])
        T.append(Y)
    W = np.stack(W, axis=0)
    H = np.stack(H, axis=0)
    S = np.stack(S, axis=0)
    T = np.stack(T, axis=0)
    ## Progressively calculate the next horizon step for all samples.
    P = np.zeros((num_samples,horizon_size,len(cfg["pred_feats"])))
    for j in range(horizon_size):
        step_horizon = np.expand_dims(H[:,j], axis=1)
        new_pred = model({"window":W, "horizon":step_horizon, "static":S})
        new_horizon = np.expand_dims(H[:,j], axis=1)
        new_window = np.concatenate((new_horizon,new_pred), axis=-1)
        P[:,j,:] = np.squeeze(new_pred)
        W = np.concatenate((W,new_window), axis=1)[:,1:]
    return P,T
    '''
    for i in range(num_samples):
        X,Y = next(sample_gen)
        W = X["window"]
        H = X["horizon"]
        S = X["static"]
        new_window = (H, np.zeros(shape=(horizon_size,len(cfg["pred_feats"]))))
        window_extend = np.concatenate(
                (W,np.concatenate(new_window,axis=-1)), axis=0)
        for j in range(horizon_size):
            P[i,j] = model({
                "window":np.expand_dims(
                    window_extend[j:j+cfg["window_size"]], axis=0),
                "horizon":np.expand_dims(
                    H[j:j+cfg["horizon_size"]], axis=0), ## size 1
                "static":np.expand_dims(S, axis=0)
                })
            window_extend[j+cfg["window_size"],-len(cfg["pred_feats"]):] \
                    = P[i,j]
            print(window_extend)
        print(i)
    print(P.shape)
    return P
    '''

if __name__=="__main__":
    #model_dir = Path("data/models/dense-1")
    model_dir = Path("data/models/lstm-s2s-2")
    #model_path = model_dir.joinpath("dense-1_38_0.07.hdf5")
    model_path = model_dir.joinpath("lstm-s2s-2_015_0.24.hdf5")
    cfg = load_config(model_dir)
    sample_h5s = [Path("data/shuffle_2018.h5")]

    '''
    plot_csv_prog(
            model_dir=model_dir,
            fields=["mae","val_mae"],
            show=True,
            plot_spec={
                "title":f"{cfg['model_name']} training metrics",
                "xlabel":"Epoch",
                "ylabel":"Metric loss value"
                }
            )
    '''

    gen = get_generator(sample_h5s=sample_h5s, model_dir=model_dir)
    #P = model_evaluate(model_path, gen, num_samples=10000)
    P = model_roll(model_path, sample_h5s, )
    print(P.shape)
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
