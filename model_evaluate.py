import pickle as pkl
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

import model_methods as mm

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

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

if __name__=="__main__":
    #model_dir = Path(f"data/models/dense-1")
    #model_path = model_dir.joinpath(f"dense-1_38_0.07.hdf5")
    #model_dir = Path(f"data/models/lstm-s2s-2")
    #model_path = model_dir.joinpath(f"lstm-s2s-2_015_0.24.hdf5")
    #model_dir = Path(f"data/models/lstm-s2s-5")
    #model_path = model_dir.joinpath(f"lstm-s2s-5_002_0.55.hdf5")
    #model_dir = Path(f"data/models/tcn-1")
    #model_path = model_dir.joinpath(f"tcn-1_092_0.03.hdf5")

    #sample_h5 = Path("data/shuffle_2018.h5")
    sample_h5 = Path("/rstor/mdodson/thesis/shuffle_2018.h5")

    cfg = load_config(model_dir)
    gen = get_generator(sample_h5s=[sample_h5], model_dir=model_dir)

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
