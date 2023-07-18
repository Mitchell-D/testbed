import pickle as pkl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

#model_path = Path("data/models/lstm-1_model.keras")
#data_path = Path("data/model_ready/lstm-1.pkl")
model_path = Path("data/models/lstm-2_model.keras")
data_path = Path("data/model_ready/lstm-2.pkl")

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
features = t_feats
static = t_static
truth = t_truth
pred_count = 128
num_frames = 8

lstm = load_model(model_path.as_posix())
#soilm_scale = lambda sm: sm*soilm_stdev+soilm_mean
soilm_scale = lambda sm: sm
for i in range(num_frames):
    s = np.random.randint(features.shape[0])
    print(f"\nShowing samples {s} to {s+pred_count}")
    #model_in = [features[s:s+pred_count], static[s:s+pred_count]]
    model_in = features[s:s+pred_count]
    pred = np.squeeze(lstm.predict(model_in))
    pred = soilm_scale(pred)
    print(np.average(pred), np.std(pred))
    actual = soilm_scale(truth[s:s+pred_count])
    soilm_in = soilm_scale(features[s:s+pred_count,-1,-1])
    plt.plot(pred, label="prediction (t)")
    plt.plot(actual, label="truth (t)")
    plt.plot(soilm_in, label="soilm (t-1)")
    plt.legend()
    plt.show()
    plt.clf()

#t_pred = lstm([t_feats[sample,:,:], t_static[sample,:]])
#v_pred = lstm.predict([v_feats, t_static])

print(pred.shape, actual.shape)
