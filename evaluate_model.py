
import pickle as pkl
from pathlib import Path
import numpy as np

from aes670hw2 import enhance as enh
from aes670hw2 import geo_plot as gp

model_dir = Path("models/set003")
t_out, v_out, s_out = pkl.load(model_dir.joinpath(
    "output/silty-loam_set003_out.pkl").open("rb"))

t_pkl = model_dir.joinpath("inputs/silty-loam_set3_training.pkl")
v_pkl = model_dir.joinpath("inputs/silty-loam_set3_validation.pkl")
s_pkl = model_dir.joinpath("inputs/silty-loam_set3_testing.pkl")

#checkpoint_file = Path("data/model_check/set001")
checkpoint_file = model_dir.joinpath("checkpoint")

t_feat,t_truth,t_times = pkl.load(t_pkl.open("rb"))
v_feat,v_truth,v_times = pkl.load(v_pkl.open("rb"))
s_feat,s_truth,s_times = pkl.load(s_pkl.open("rb"))

cycle_size = 8064
#cycle_size = 2016
#cycle_size = 2016
num_px = 12 # for unraveling pixels in each dataset
cnum = 2 # cycle number

print(t_feat.shape)
print(t_truth.shape)

def unwrap_pixels(X, cycle_num, cycle_size, num_px):
    X = X[cycle_num*cycle_size:cycle_num*cycle_size+cycle_size]
    return np.split(X, num_px, axis=0)

t_truth = unwrap_pixels(t_truth, cnum, cycle_size, num_px)[0]
t_truth = [t_truth[:,i] for i in  range(t_truth.shape[1])]

t_out = unwrap_pixels(t_out, cnum, cycle_size, num_px)[0]
t_out = [t_out[:,i] for i in range(t_out.shape[1])]

gp.plot_lines(
        range(t_truth[0].size),
        t_truth+t_out,
        show=True
        )
