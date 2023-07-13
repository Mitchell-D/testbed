from pathlib import Path
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

data_dict = pkl.load(Path("data/model_ready/lstm-1.pkl").open("rb"))
t_feats, t_static, t_truth = data_dict["training"]
v_feats, v_static, v_truth = data_dict["validation"]

print(t_feats.shape, t_static.shape, t_truth.shape)
feature_labels = ["APCP", "CAPE", "DLWRF", "DSWRF", "PEVAP",
                      "PRES", "SPFH", "TMP", "SOILM-0-10"]

# Number of random samples to observe
num_samples = 5
for j in range(num_samples):
    new_sample = np.random.randint(t_feats.shape[0])
    print(f"Displaying sample {new_sample}")
    for i in range(len(feature_labels)):
        plt.plot(t_feats[new_sample,:,i], label=feature_labels[i])
    plt.legend()
    plt.show()
    plt.clf()
