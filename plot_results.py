
import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pprint import pprint
from datetime import timedelta,datetime

#from krttdkit.visualize import geoplot as gp
from model_evaluate import gen_pred_seqs

## labels shared by multiple plotting methods
depth_labels = ["0-10 cm", "10-40 cm", "40-100 cm", "100-200 cm"]

def plot_hist(hist_pkl:Path, fig_dir:Path=None, show:bool=False, title:str=""):
    """ """
    phist,thist,all_min,all_max = pkl.load(hist_pkl.open("rb"))
    domain = np.linspace(all_min,all_max,phist.shape[1])
    cmap = ["red", "blue", "green", "orange"]

    fig,ax = plt.subplots(2,2)

    fig.suptitle(title,y=.96)
    fig.supxlabel("Soil Moisture ($kg / m^3$)",y=.04)
    fig.supylabel("Frequency")
    axids = [(0,0), (0,1), (1,0), (1,1)]
    hist_sum = np.sum(thist[0])
    for i in range(len(axids)):
        c = cmap[i]
        ax[axids[i][0],axids[i][1]].plot(
                domain[:,i], thist[i]/hist_sum, color=c, linewidth=1,
                label="true")
        ax[axids[i][0],axids[i][1]].plot(
                domain[:,i], phist[i]/hist_sum, color="black",
                linestyle="dashed", linewidth=2, label="model")
        ax[axids[i][0],axids[i][1]].set_title("SOILM " + depth_labels[i])
        ax[axids[i][0],axids[i][1]].tick_params(
                axis="y",left=False, right=False,labelleft=False)
        ax[axids[i][0],axids[i][1]].legend()
    fig.tight_layout()
    if fig_dir:
        plt.savefig(fig_dir.joinpath(hist_pkl.stem+".png"), dpi=800)
    if show:
        plt.show()

def plot_grid_error(
        error_file:Path, fig_dir:Path=None, show:bool=False, title:str=""):
    """ """
    error = np.load(error_file)

    fig,ax = plt.subplots(2,2)

    fig.suptitle(title,y=.96)
    axids = [(0,0), (0,1), (1,0), (1,1)]
    for i in range(len(axids)):
        cmap = matplotlib.cm.gist_rainbow
        cmap.set_bad("black",1.)
        arr = np.ma.array(error[...,i], mask=np.isnan(error[...,i]))
        im = ax[axids[i][0],axids[i][1]].imshow(arr, cmap=cmap)
        div = make_axes_locatable(ax[axids[i][0],axids[i][1]])
        cax = div.append_axes("bottom",size="5%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="horizontal")
        ax[axids[i][0],axids[i][1]].set_title("SOILM " + depth_labels[i])
        ax[axids[i][0],axids[i][1]].axes.xaxis.set_visible(False)
        ax[axids[i][0],axids[i][1]].axes.yaxis.set_visible(False)
    fig.tight_layout()
    #plt.subplot
    if fig_dir:
        plt.savefig(fig_dir.joinpath(error_file.stem+".png"), dpi=800)
    if show:
        plt.show()

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

def plot_error_horizons(error_dict, title="", fig_dir:Path=None, show=False):
    """
    Plot the error rates wrt horizon distance for each model given a

    :@param: dict mapping a string model name to a (H,D) shaped numpy array
        for H horizons at D depth levels.
    :@param title: String for the plot title; "{key}" fstring is exposed to
        the keys in the corresponding dictionary.
    """
    for k in error_dict.keys():
        fig,ax = plt.subplots()
        X = error_dict[k].T
        for i in range(X.shape[0]):
            ax.plot(range(X.shape[1]), X[i], label=depth_labels[i])
        ax.legend()
        ax.set_title(title.format(key=k))
        ax.set_xlabel("Horizon Distance (Hours)")
        ax.set_ylabel("Mean Absolute Error ($kg/m^3$)")
        if show:
            plt.show()
        if not fig_dir is None:
            assert fig_dir.exists()
            plt.savefig(fig_dir.joinpath(f"error_horizons_{k}.png"), dpi=800)
        plt.clf()

def show_prediction_sequences(seq_gen, fig_dir=None, show=False, num_seqs=100):
    """
    """
    for i in range(num_seqs):
        sdict = next(seq_gen)
        ## Get a time domain with window values depicted as negative
        full_range = np.arange(-sdict["window"].shape[0],
                               sdict["true"].shape[0])
        fig,ax = plt.subplots(2,2)
        axids = [(0,0), (0,1), (1,0), (1,1)]
        for j in range(len(axids)):
            ## Plot the window, true, and predicted features
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[:sdict["window"].shape[0]],
                    sdict["window"][:,j],
                    color="blue",
                    )
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[sdict["window"].shape[0]-1:],
                    [sdict["window"][-1,j]]+list(sdict["true"][:,j]),
                    color="blue",
                    linestyle="dashed",
                    label="actual",
                    )
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[sdict["window"].shape[0]-1:],
                    [sdict["window"][-1,j]]+list(sdict["prediction"][:,j]),
                    color="red",
                    label="predicted",
                    )
            ax[axids[j][0],axids[j][1]].set_title("SOILM " + depth_labels[j])
            ax[axids[j][0],axids[j][1]].legend()
        fig.tight_layout()
        fig.suptitle("".format(time=sdict["time"]))
        fig.supxlabel("Forecast Hour")
        fig.supylabel("Soil Moisture ($kg\,m^{-3}$)")
        if not fig_dir is None:
            fig.savefig(fig_dir.joinpath(f"sequence_{i}.png"))
        if show:
            plt.show()

def quad_plot_sample_features(sample_h5, features:list, num_samples=10,
        fig_dir=None, show=False, title="", xlabel="sample time", ylabel="",
        pivot_idx=36, dt=timedelta(hours=1), feat_labels=None):
    """ """
    G = h5py.File(sample_h5, "r")["data"]
    feats = G["dynamic"]
    flabels = list(G.attrs["flabels"])
    time = G["time"]
    feat_idxs = tuple(flabels.index(f) for f in features)
    sample_idxs = np.arange(feats.shape[0])
    np.random.shuffle(sample_idxs)
    feat_labels = features if feat_labels is None else feat_labels
    for i in range(num_samples):
        tmp_idx = sample_idxs[i]
        tmp_feats = feats[tmp_idx,...][...,feat_idxs].T
        fig, ax = plt.subplots()
        ## the time is labeled at the pivot index; modify it so that
        ## the initial time is at t0
        t0 = datetime.fromtimestamp(time[tmp_idx]) - dt * pivot_idx
        tmp_times = [t0+n*dt for n in range(tmp_feats.shape[1])]
        for j in range(len(features)):
            ax.plot(tmp_times, tmp_feats[j], label=feat_labels[j])
        ax.set_title(title.format(time=tmp_times[-1]))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        '''
        fig,ax = plt.subplots(2,2)
        axids = [(0,0), (0,1), (1,0), (1,1)]
        for j in range(len(axids)):
            ## Plot the window, true, and predicted features
            ax[axids[j][0],axids[j][1]].plot(
                    range(tmp_feats.shape[1])],
                    tmp_feats[j],
                    )
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[sdict["window"].shape[0]-1:],
                    [sdict["window"][-1,j]]+list(sdict["true"][:,j]),
                    color="blue",
                    linestyle="dashed",
                    )
            ax[axids[j][0],axids[j][1]].plot(
                    full_range[sdict["window"].shape[0]-1:],
                    [sdict["window"][-1,j]]+list(sdict["prediction"][:,j]),
                    color="red",
                    )
            ax[axids[j][0],axids[j][1]].set_title("SOILM " + depth_labels[j])
        fig.tight_layout()
        fig.suptitle("".format(time=sdict["time"]))
        fig.supxlabel("Forecast Hour",y=.04)
        fig.supylabel("Soil Moisture ($kg\,m^{-3}$$",y=.04)
        '''
        if not fig_dir is None:
            fig.savefig(fig_dir.joinpath(f"sequence_{i}.png"))
        if show:
            plt.show()

def plot_static(static_pkl_path:Path, print_info=True, img_dir:Path=None):
    """

    """
    slabels,sdata = pkl.load(path.open("rb"))
    for l,s in zip(slabels, sdata):
        if print_info:
            print(l,s.shape,s.dtype)
        if img_dir:
            nanmask = np.isnan(s)
            s[nanmask] = np.nanmean(s)
            rgb = enh.norm_to_uint(
                    gt.scal_to_rgb(enh.linear_gamma_stretch(
                        s.astype(np.float32))),
                    256, np.uint8)
            rgb[nanmask] = 0
            img_path = img_dir.joinpath(f"{l}.png")
            if img_path.exists():
                print(f"Skipping {img_path.as_posix()}; already exists.")
                continue
            gp.generate_raw_image(rgb, img_path)
    return slabels,sdata

if __name__ == "__main__":
    data_dir = Path("data")
    #data_dir = Path("/rstor/mdodson/thesis")
    fig_dir = Path("figures")

    '''
    """ Plot basic RGBs of the static arrays """
    plot_static(
            static_pkl_path=Path("data/static/nldas_static.pkl"),
            img_dir=Path("figures/static"),
            )
    '''

    '''
    """
    Plot bulk error rates wrt horizon distance from a pkl
    """
    mae_pkl = data_dir.joinpath("mae.pkl")
    mae_bulk,mae_horizons = pkl.load(mae_pkl.open("rb"))
    plot_error_horizons(
            error_dict=mae_horizons,
            title="{key} MAE wrt Prediction Horizon (test data)",
            fig_dir=fig_dir.joinpath("horizon_error"),
            )
    pprint(mae_bulk)
    '''

    '''
    quad_plot_sample_features(
            sample_h5=data_dir.joinpath("shuffle_2018.h5"),
            features=["soilm-10", "soilm-40", "soilm-100", "soilm-200"],
            num_samples=10,
            title="Depth-wise Volumetric Water Content",
            xlabel="Sample Time",
            ylabel="Soil Moisture ($kg\,m^{-3}$)",
            pivot_idx=36,
            dt=timedelta(hours=1),
            show=True,
            feat_labels=["0-10cm", "10-40cm", "40-100cm", "100-200cm"]
            )
    '''

    #'''
    """ Plot individual sequence outputs """
    sgen = gen_pred_seqs(
            sample_h5=data_dir.joinpath("shuffle_2018.h5"),
            #pred_h5=data_dir.joinpath("pred/pred_2018_V2_dense-1.h5"),
            #pred_h5=data_dir.joinpath("pred/pred_2018_V2_lstm-rec-1.h5"),
            #pred_h5=data_dir.joinpath("pred/pred_2018_V2_lstm-s2s-2.h5"),
            #pred_h5=data_dir.joinpath("pred/pred_2018_V2_lstm-s2s-5.h5"),
            pred_h5=data_dir.joinpath("pred/pred_2018_V2_tcn-1.h5"),

            pred_feats=['soilm-10', 'soilm-40', 'soilm-100', 'soilm-200'],
            window_size=12
            )
    show_prediction_sequences(
            seq_gen=sgen,
            num_seqs=10,
            show=True
            #fig_dir=fig_dir.joinpath(f"sequences"),
            )
    #'''


    '''
    """
    Plot gridded error files at 4 depth level from npy files generated by
    model_evaluate.get_grid_mae using a consecutive sample and prediction file
    """
    #error_file = data_dir.joinpath("grid_error/grid_mae_dense-1.npy")
    #error_file = data_dir.joinpath("grid_error/grid_mae_lstm-rec-1.npy")
    #error_file = data_dir.joinpath("grid_error/grid_mae_lstm-s2s-2.npy")
    #error_file = data_dir.joinpath("grid_error/grid_mae_lstm-s2s-5.npy")
    error_file = data_dir.joinpath("grid_error/grid_mae_tcn-1.npy")
    model_name = error_file.stem.split("_")[-1]
    plot_grid_error(
            error_file=error_file,
            title=f"{model_name} Mean Absolute Error (test data)",
            fig_dir=fig_dir.joinpath("grid_error"),
            show=False,
            )
    '''

    '''
    """
    Plot depth-wise value histograms from pkl files generated by
    model_evaluate.get_histograms using a prediction file
    """
    hist_pkl = data_dir.joinpath("hists/hist_2018_dense-1.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_lstm-rec-1.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_lstm-s2s-2.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_lstm-s2s-5.pkl")
    #hist_pkl = data_dir.joinpath("hists/hist_2018_tcn-1.pkl")

    model_name = hist_pkl.stem.split("_")[-1]
    plot_hist(
            hist_pkl=hist_pkl,
            title=f"{model_name} normalized value frequency (test data)",
            fig_dir=fig_dir.joinpath("hists"),
            )
    '''
