import pickle as pkl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

#from aes670hw2 import enhance as enh
#from aes670hw2 import geo_plot as gp

if __name__=="__main__":
    model_path = Path("data/models/lstm-3_model.keras")
    data_path = Path("data/model_ready/lstm-3.pkl")
    data_dict = pkl.load(data_path.open("rb"))
    t_feats, t_static, t_truth = data_dict["training"]
    v_feats, v_static, v_truth = data_dict["validation"]
    soilm_mean, soilm_stdev = data_dict["scales"]["SOILM-0-10"]
