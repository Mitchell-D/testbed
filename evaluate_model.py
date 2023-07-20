import pickle as pkl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

#model_path = Path("data/models/lstm-1_model.keras")
#data_path = Path("data/model_ready/lstm-1.pkl")

model_a_path = Path("data/models/lstm-2_22_0.00.hdf5")
#model_b_path = Path("data/models/lstm-2_model.keras")
model_b_path = None
#data_path = Path("data/model_ready/lstm-2.pkl")
data_path = Path("data/model_ready/lstm-2-allsoil.pkl")

data_dict = pkl.load(data_path.open("rb"))
t_feats, t_static, t_truth = data_dict["training"]
v_feats, v_static, v_truth = data_dict["validation"]
soilm_mean, soilm_stdev = data_dict["scales"]["SOILM-0-10"]

print(t_feats.shape, t_static.shape, t_truth.shape)

"""
Define the arrays of samples to evaluate. Feature and static arrays
must be (s,w,f) and (s,f) shaped, respectively, for s samples, w lookback
window size, and f features.

Note that concurrent samples may be discontinuous if they overlap pixel
time series cutoffs in the original time series. Nonetheless, predictions
are made on a per-sample basis.
"""
features = v_feats
static = v_static
truth = v_truth
pred_count = 128
num_frames = 24
fig_dir = Path("figures/model_out")
fig_format = "{model_name}_{variable}_{set_label}_{sample_number}"
model_name = "lstm-2"
variable = "0-10cm"
set_label = "ep22-test-allsoil"
model_a_name = "Model (ep 22)"
model_b_name = "Model (ep 26)"

model_a = load_model(model_a_path.as_posix())
if model_b_path:
    model_b = load_model(model_b_path.as_posix())
soilm_scale = lambda sm: sm*soilm_stdev+soilm_mean
#soilm_scale = lambda sm: sm
for i in range(num_frames):
    s = np.random.randint(features.shape[0])
    print(f"\nShowing samples {s} to {s+pred_count}")
    #model_in = [features[s:s+pred_count], static[s:s+pred_count]]
    model_in = features[s:s+pred_count]
    pred_a = soilm_scale(np.squeeze(model_a.predict(model_in)))
    actual = soilm_scale(truth[s:s+pred_count])
    #soilm_in = soilm_scale(features[s:s+pred_count,-1,-1])
    # If model_b_path is defined, predict and plot the same sample with it.
    # Assumed to be same input/output tensor shapes as model_a
    if model_b_path:
        pred_b = soilm_scale(np.squeeze(model_b.predict(model_in)))
        plt.plot(pred_b, label=model_b_name)
    plt.plot(actual, label="truth (t)")
    #plt.plot(soilm_in, label="soilm (t-1)")
    plt.plot(pred_a, label=model_a_name)
    plt.ylim([0,50])
    plt.ylabel(f"Soil Moisture ({variable}; {set_label})")
    plt.xlabel("Hours")
    plt.title(f"Predicted vs Truth Soil Moisture ({set_label})")
    plt.legend()
    #plt.show()
    plt.savefig(fig_dir.joinpath(fig_format.format(
        model_name=model_name, set_label=set_label,
        sample_number=s, variable=variable)))
    plt.clf()

