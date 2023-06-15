import pickle as pkl
from pathlib import Path
import numpy as np

from aes670hw2 import enhance as enh
from aes670hw2 import geo_plot as gp

def wrap_pixels(X, cycle_num, cycle_size, num_px):
    """
    Based on the cycle size, the number of concatenated "cycle split" cycles,
    and the number of appended pixels per cycle in a 1D dataset, split the
    dataset back into the original continuous time series per pixel
    """
    X = X[cycle_num*cycle_size:cycle_num*cycle_size+cycle_size]
    return np.split(X, num_px, axis=0)


if __name__=="__main__":
    model_dir = Path("models/set003")

    set_label = "silty-loam_set003"

    """
    Parameters for 'wrapping' a dataset generated with the
    cycle-split technique datasets
    """
    # Size of the
    cycle_size = 8064 # set003 training data
    #cycle_size = 2016 # set003 validation/testing data
    num_px = 12 # for unraveling pixels in each dataset
    #CNUM = 2 # cycle number
    #PNUM = 10
    for CNUM in range(4):
        for PNUM in range(12):
            TITLE = f"Set 3 model on training data, C{CNUM}/P{PNUM}"

            fig_dir = Path(f"figures/output_curves/")
            fig_path = fig_dir.joinpath(
                    Path(f"{set_label}_train_C{CNUM}_P{PNUM}.png"))

            #checkpoint_file = Path("data/model_check/set001")
            checkpoint_file = model_dir.joinpath("checkpoint")

            t_out, v_out, s_out = pkl.load(model_dir.joinpath(
                f"output/{set_label}_out.pkl").open("rb"))

            """ Load the pickles of model input data """
            t_pkl = model_dir.joinpath(f"input/{set_label}_training.pkl")
            v_pkl = model_dir.joinpath(f"input/{set_label}_validation.pkl")
            s_pkl = model_dir.joinpath(f"input/{set_label}_testing.pkl")

            t_feat,t_truth,t_times = pkl.load(t_pkl.open("rb"))
            v_feat,v_truth,v_times = pkl.load(v_pkl.open("rb"))
            s_feat,s_truth,s_times = pkl.load(s_pkl.open("rb"))

            """ Configure the dataset to model and plot """
            TRUTH = t_truth
            OUT = t_out
            TIMES = t_times

            print("TRUTH:",TRUTH.shape)
            print("MODEL:",OUT.shape)

            # Extract the truth basis sample and split it into curves for each
            # independent soil depth level
            TRUTH = wrap_pixels(TRUTH, CNUM, cycle_size, num_px)[PNUM]
            TRUTH = [TRUTH[:,i] for i in  range(TRUTH.shape[1])]
            OUT = wrap_pixels(OUT, CNUM, cycle_size, num_px)[PNUM]
            OUT = [OUT[:,i] for i in range(OUT.shape[1])]
            ipos = CNUM*cycle_size
            TIMES = [ t.strftime("%j") for t in TIMES[CNUM] ]

            gp.plot_lines(
                    TIMES,
                    TRUTH+OUT,
                    show=False,
                    plot_spec={
                        "yrange":(0,400),
                        "title":TITLE,
                        "xlabel":"Day of the year",
                        "ylabel":"Liquid soil moisture ($\\frac{kg}{m^2}$)",
                        "line_width":1.2,
                        "colors":["blue" for i in range(len(TRUTH))] + \
                                ["red" for i in range(len(OUT))],
                        #"legend_ncols":2,
                        "dpi":200,
                        },
                    image_path=fig_path
                    )

