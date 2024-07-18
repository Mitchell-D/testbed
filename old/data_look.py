import pickle as pkl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    '''
    data_path = Path("data/models/osmh_1/osmh_1_curated.pkl")
    model_path = Path("data/models/osmh_1/run_1/osmh_1_model.keras")
    csv_path = Path("data/models/osmh_1/run_1/osmh_1_prog.csv")
    '''
    data_path = Path("data/models/osmh_1/osmh_1_curated.pkl")
    data_dict = pkl.load(data_path.open("rb"))

    window = 24
    horizon = 12

    t_feats = data_dict["train"]["X"]
    t_truth = data_dict["train"]["Y"]
    v_feats = data_dict["validate"]["X"]
    v_truth = data_dict["validate"]["Y"]

    t_fct = np.stack([
        t_feats[i+1:i+horizon+1 , -1 , :-1]
        for i in range(t_feats.shape[0]-horizon)
        ])
    t_feats = t_feats[:-horizon]

    v_fct = np.stack([
        v_feats[i+1:i+horizon+1 , -1 , :-1]
        for i in range(v_feats.shape[0]-horizon)
        ])
    v_feats = v_feats[:-horizon]

    print(t_feats.shape)
    print(t_fct.shape)
    for i in range(40):
        #wdw = [t_feats[i,:,:j] for j in range(8)]
        #fct = [t_fct[i,:,:j] for j in range(8)]
        wdw = [t_feats[i,:,5]]
        fct = [t_fct[i,:,5]]
        for w in wdw:
            plt.plot(range(t_feats.shape[1]), w)
        for f in fct:
            plt.plot(range(t_feats.shape[1],t_feats.shape[1]+t_fct.shape[1]),f)
        plt.show()
        plt.clf()
