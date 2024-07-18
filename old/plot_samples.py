import pickle as pkl
import numpy as np
from pathlib import Path
from datetime import datetime

from krttdkit.visualize import geoplot as gp

def plot_samples(sample_dict, pivot_idx, sample_count=10):
    time_idx = np.arange(-pivot_idx,sample_dict["dynamic"].shape[1]-pivot_idx)

    soilt_labels = ("pct_sand","pct_silt","pct_clay")
    soilt = [sample_dict["static"][...,list(sample_dict["slabels"]).index(k)]
             for k in soilt_labels]
    soilt = np.stack(soilt).T

    soilm_labels = ("soilm-10","soilm-40","soilm-100","soilm-200")
    soilm = [sample_dict["dynamic"][...,list(sample_dict["flabels"]).index(k)]
             for k in soilm_labels]
    soilm = np.stack(soilm).transpose((1,0,2))

    static = sample_dict["static"]
    yx = static[...,-2:].astype(np.uint8)
    print(yx.shape)

    print(soilt.shape, soilm.shape)

    idxs = np.random.randint(0,high=static.shape[0]-1,size=sample_count)
    for ix in idxs:
        soil_vec = "({:.2f}, {:.2f}, {:.2f})".format(*list(soilt[ix]))
        t = datetime.fromtimestamp(int(sample_dict["time"][ix]))
        gp.plot_lines(
                domain=time_idx,
                ylines=[soilm[ix,j] for j in range(soilm.shape[1])],
                labels=soilm_labels,
                plot_spec={
                    "title":f"SOILM {t} ({yx[ix,0]},{yx[ix,1]}) {soil_vec}",
                    "xlabel":f"Hours from pivot time",
                    "ylabel":"Soil Moisture (kg/m^2)"
                    },
                show=True,
                )

'''
def plot_batch_sample(batch, sample_count=10):
    X,pred = batch
    window = X["window"]
    horizon = X["horizon"]
    static = X["static"]
    print(static)
    soilm = np.concatenate((window[:,-4:], pred), axis=0)
    temp = np.concatenate((window[:,2], horizon[:,2]))
    print(temp.shape)

    gp.plot_lines(
            domain=list(range(48)),
            ylines=[soilm[:,j] for j in range(soilm.shape[1])],
            show=True,
            )
    gp.plot_lines(
            domain=list(range(48)),
            ylines=[temp],
            show=True,
            )
'''

if __name__=="__main__":
    sample_path = Path("data/sample/shuffled_samples.pkl")
    batch_path = Path("data/sample/batch_samples.pkl")
    #plot_samples(pkl.load(sample_path.open("rb")), 36, 30)
    for i in range(30):
        plot_batch_sample(pkl.load(batch_path.open("rb"))[i])
