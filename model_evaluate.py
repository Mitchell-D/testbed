import pickle as pkl
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

import model_methods as mm
from list_feats import static_coeffs

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

def get_generator(sample_h5s:Path, model_dir, as_tensor=False):
    """
    Returns a data generator for the provided sample h5 according to the model
    configuration associated with model_dir. The provided h5 must conform to
    the format used by curate_samples.py
    """
    cfg = mm.load_config(model_dir)
    if "window_and_horizon_size" in cfg.keys():
        cfg["window_size"] = cfg["window_and_horizon_size"]
        cfg["horizon_size"] = cfg["window_and_horizon_size"]
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
    cfg = mm.load_config(model_dir)
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
    cfg = mm.load_config(model_dir)
    csv = mm.load_csv_prog(model_dir)
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

def gen_sequences(sample_h5, pred_h5, pred_feats, window_size=12):
    """ """
    pG = h5py.File(pred_h5, "r")["data"]
    sG = h5py.File(sample_h5, "r")["data"]

    feats = sG["dynamic"]
    static = sG["static"]
    flabels = list(sG.attrs["flabels"])
    slabels = list(sG.attrs["slabels"])

    pred_idxs = tuple(flabels.index(p) for p in pred_feats)

    P = pG["prediction"]
    Y = pG["truth"]
    sidx = np.array(pG["sample_idx"]).astype(int)
    pidx = np.array(pG["pivot_idx"]).astype(int)

    for i in range(sidx.shape[0]):
        yield {
                "window":feats[sidx[i]][pidx[i]-window_size:pidx[i],pred_idxs],
                "prediction":P[i],
                "true":Y[i],
                "static":static[i],
                }

def get_histograms(pred_h5, nbins=512):
    """ """
    F = h5py.File(pred_h5, "r")
    P = np.array(F["/data/prediction"])
    T = np.array(F["/data/truth"])

    ## Get the value extrema for truth and predictions
    pmin = np.amin(np.amin(P,axis=0), axis=0)
    pmax = np.amax(np.amax(P,axis=0), axis=0)
    tmin = np.amin(np.amin(T,axis=0), axis=0)
    tmax = np.amax(np.amax(T,axis=0), axis=0)
    all_min = np.amin(np.stack([pmin, tmin],axis=0),axis=0)
    all_max = np.amax(np.stack([pmax, tmax],axis=0),axis=0)

    ## Rescale the arrays and quantize them into bins
    P -= all_min
    T -= all_min
    P /= (all_max-all_min)
    T /= (all_max-all_min)
    P *= nbins-1
    T *= nbins-1
    P = np.rint(P).astype(np.uint32)
    T = np.rint(T).astype(np.uint32)

    phist = np.zeros((all_min.size, nbins), dtype=np.uint32)
    thist = np.zeros((all_min.size, nbins), dtype=np.uint32)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for k in range(P.shape[2]):
                phist[k,P[i,j,k]] += 1
                thist[k,T[i,j,k]] += 1
    return phist,thist,all_min,all_max

def get_mae(pred_h5, keep_seqs=True):
    F = h5py.File(pred_h5, "r")
    P = np.array(F["/data/prediction"])
    T = np.array(F["/data/truth"])
    if not keep_seqs:
        P.reshape
    E = mae(P,T)
    return E

def mae(X, Y):
    return np.sum(np.abs(X-Y), axis=0)/X.shape[0]

def rmse(X, Y):
    return (np.sum((X-Y)**2, axis=0)/X.shape[0])**(1/2)

def get_grid_mae(sample_h5, pred_h5):
    """  """
    pG = h5py.File(pred_h5, "r")["data"]
    sG = h5py.File(sample_h5, "r")["data"]

    ## Load and scale the grid indeces
    static = np.asarray(sG["static"])
    slabels = list(sG.attrs["slabels"])
    vidx = np.rint(static[:,slabels.index("vidx")]).astype(int)
    hidx = np.rint(static[:,slabels.index("hidx")]).astype(int)

    ## Calculate mean absolute error for all samples
    P = np.array(pG["/data/prediction"])
    T = np.array(pG["/data/truth"])
    E = np.average(np.abs(P-T), axis=1)
    idxs = np.array(pG["sample_idx"]).astype(int)

    grid_shape = (np.amax(vidx)+1,np.amax(hidx)+1,P.shape[-1])
    err = np.zeros(grid_shape,dtype=np.float64)
    count = np.zeros(grid_shape[:2], dtype=int)
    print(grid_shape)
    for i in range(E.shape[0]):
        v = vidx[idxs[i]]
        h = hidx[idxs[i]]
        err[v,h] += E[i]
        count[v,h] += 1
    ## count should be uniform in valid grid cells
    return err/np.expand_dims(count, axis=-1)

if __name__=="__main__":
    data_dir = Path("/rstor/mdodson/thesis")

    sample_h5 = data_dir.joinpath("shuffle_2018.h5")

    model_parent_dir = Path("data/models")
    #pred_h5 = data_dir.joinpath("pred_2018_dense-1.h5")
    #pred_h5 = data_dir.joinpath("pred_2018_lstm-rec-1.h5")
    #pred_h5 = data_dir.joinpath("pred_2018_lstm-s2s-2.h5")
    #pred_h5 = data_dir.joinpath("pred_2018_lstm-s2s-5.h5")
    pred_h5 = data_dir.joinpath("pred_2018_tcn-1.h5")

    '''
    pred_h5 = data_dir.joinpath("pred_2018_SEUS_dense-seus-0.h5")
    pred_h5 = data_dir.joinpath("pred_2018_SEUS_lstm-rec-seus-0.h5")
    pred_h5 = data_dir.joinpath("pred_2018_SEUS_lstm-s2s-seus-0.h5")
    pred_h5 = data_dir.joinpath("pred_2018_SEUS_lstm-s2s-seus-1.h5")
    pred_h5 = data_dir.joinpath("pred_2018_SEUS_tcn-seus-0.h5")
    '''

    ## Get the model directory using the model name field, and parse the config
    model_dir = model_parent_dir.joinpath(
            pred_h5.name.split(".")[0].split("_")[-1])
    cfg = mm.load_config(model_dir)

    '''
    g = gen_sequences(sample_h5, pred_h5, cfg["pred_feats"])
    for i in range(1000):
        tmp = next(g)
        print([(k,v.shape) for k,v in tmp.items()])
    '''

    '''
    E = get_mae(pred_h5)
    print(E)
    print(E.shape)
    '''

    grid_path = data_dir.joinpath(f"grid_mae_{cfg['model_name']}.npy")
    np.save(grid_path, get_grid_mae(sample_h5, pred_h5))

    #hist_path = data_dir.joinpath(f"hist_2018_{cfg['model_name']}.pkl")
    #pkl.dump(get_histograms(pred_h5), hist_path.open("wb"))

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
