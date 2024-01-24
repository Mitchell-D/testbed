import pickle as pkl
from pathlib import Path
import numpy as np
import h5py
import json

import model_methods as mm

from tensorflow.keras.models import load_model

def model_predict(model_path, sample_h5, pred_h5, chunk_size=1000):
    """ """
    model = load_model(model_path)
    cfg = mm.load_config(model_path.parent)
    ## Make sure the last features in the window are the predicted ones
    assert tuple(cfg["window_feats"][-len(cfg["pred_feats"]):]) \
            == tuple(cfg["pred_feats"])

    ## For TCN; should've just copied them over in the config parsing
    if "window_and_horizon_size" in cfg.keys():
        cfg["window_size"] = cfg["window_and_horizon_size" ]
        cfg["horizon_size"] = cfg["window_and_horizon_size" ]

    ## Get a sample generator for the full requested horizon size
    sample_gen = mm.gen_sample(
            h5_paths=[sample_h5],
            window_size=cfg["window_size"],
            horizon_size=cfg["horizon_size"],
            window_feats=cfg["window_feats"],
            horizon_feats=cfg["horizon_feats"],
            pred_feats=cfg["pred_feats"],
            static_feats=cfg["static_feats"],
            as_tensor=False,
            return_idx=True,
            shuffle_chunks=False,
            )
    num_smpl = h5py.File(sample_h5.as_posix(), "r")["/data/dynamic"].shape[0]
    ## scuffed way to iterate over requested batches
    chunks = [chunk_size for i in range(num_smpl//chunk_size)]
    chunks += [num_smpl%chunk_size]
    ## Get scaling coefficients to output in data coordinates
    out_mean,out_stdev = mm.get_dynamic_coeffs(cfg["pred_feats"])

    ## Initialize a new hdf5 file for storing predictions
    f_pred = h5py.File(pred_h5.as_posix(), "w-", rdcc_nbytes=200*1024**2)
    g_pred = f_pred.create_group("/data")
    ## Append the config dict as a JSON attribute
    g_pred.attrs["config"] = json.dumps(cfg)
    d_pred = g_pred.create_dataset(
            name="prediction",
            shape=(num_smpl,cfg["horizon_size"],len(cfg["pred_feats"])),
            chunks=(chunk_size,cfg["horizon_size"],len(cfg["pred_feats"])),
            compression="gzip",
            )
    d_truth = g_pred.create_dataset(
            name="truth",
            shape=(num_smpl,cfg["horizon_size"],len(cfg["pred_feats"])),
            chunks=(chunk_size,cfg["horizon_size"],len(cfg["pred_feats"])),
            compression="gzip"
            )
    d_wdw = g_pred.create_dataset(
            name="window",
            shape=(num_smpl,cfg["window_size"],len(cfg["pred_feats"])),
            chunks=(chunk_size,cfg["window_size"],len(cfg["pred_feats"])),
            compression="gzip"
            )
    ## sample index
    d_sidx = g_pred.create_dataset(
            name="sample_idx",
            shape=(num_smpl,),
            chunks=chunk_size,
            )
    ## pivot index
    d_pidx = g_pred.create_dataset(
            name="pivot_idx",
            shape=(num_smpl,),
            chunks=chunk_size,
            )

    pred_idx = 0
    wdw_pred_idxs = tuple(
            cfg["window_feats"].index(f) for f in cfg["pred_feats"])
    print(f"Window indeces: {wdw_pred_idxs}")
    ## Note that chunks is a list of integers representing chunk sizes
    while len(chunks):
        print(chunks)
        W,H,S,P,T,sample_idxs,pivot_idxs = [],[],[],[],[],[],[]
        W_pred = [] ## store a list of prediction feats in the window
        cur_chunk = chunks.pop(0)
        ## Extract and consolidate all the sample components
        for i in range(cur_chunk):
            X,Y,idxs = next(sample_gen)
            W.append(X["window"])
            W_pred.append(W[-1][...,wdw_pred_idxs])
            H.append(X["horizon"])
            S.append(X["static"])
            sample_idxs.append(idxs[1])
            pivot_idxs.append(idxs[2])
            T.append(Y)
        W = np.stack(W, axis=0)
        W_pred = np.stack(W_pred, axis=0)
        H = np.stack(H, axis=0)
        S = np.stack(S, axis=0)
        T = np.stack(T, axis=0)
        P = model({"window":W, "horizon":H, "static":S})
        ## Save predicted features in the window and horizon range
        d_pred[pred_idx:pred_idx+cur_chunk,...] = P * out_mean + out_stdev
        d_truth[pred_idx:pred_idx+cur_chunk,...] = T * out_mean + out_stdev
        d_wdw[pred_idx:pred_idx+cur_chunk,...] = W_pred * out_mean + out_stdev
        #d_stat[pred_idx:pred_idx+cur_chunk,...] = S * out_mean + out_stdev
        d_pidx[pred_idx:pred_idx+cur_chunk] = np.array(pivot_idxs)
        d_sidx[pred_idx:pred_idx+cur_chunk] = np.array(sample_idxs)
        f_pred.flush()
        pred_idx += cur_chunk
    return d_pred

def model_roll_chunks(model_path, sample_h5, pred_h5,
        horizon_size=12, chunk_size=1024):
    """ roll-evaluates a full sample hdf5 by chunking it """
    model = load_model(model_path)
    cfg = mm.load_config(model_path.parent)
    ## Make sure the last features in the window are the predicted ones
    assert tuple(cfg["window_feats"][-len(cfg["pred_feats"]):]) \
            == tuple(cfg["pred_feats"])
    ## Get a sample generator for the full requested horizon size
    sample_gen = mm.gen_sample(
            h5_paths=[sample_h5],
            window_size=cfg["window_size"],
            horizon_size=horizon_size,
            window_feats=cfg["window_feats"],
            horizon_feats=cfg["horizon_feats"],
            pred_feats=cfg["pred_feats"],
            static_feats=cfg["static_feats"],
            as_tensor=False,
            return_idx=True,
            shuffle_chunks=False,
            )
    ## Open the sample file briefly to check the number of samples
    num_smpl = h5py.File(sample_h5.as_posix(), "r")["/data/dynamic"].shape[0]
    ## scuffed way to iterate over requested batches
    chunks = [chunk_size for i in range(num_smpl//chunk_size)]
    chunks += [num_smpl%chunk_size]
    ## Get scaling coefficients to output in data coordinates
    out_mean,out_stdev = mm.get_dynamic_coeffs(cfg["pred_feats"])

    ## Initialize a new hdf5 file for storing predictions
    f_pred = h5py.File(pred_h5.as_posix(), "w-", rdcc_nbytes=200*1024**2)
    g_pred = f_pred.create_group("/data")
    g_pred.attrs["config"] = json.dumps(cfg)
    d_pred = g_pred.create_dataset(
            name="prediction",
            shape=(num_smpl,horizon_size,len(cfg["pred_feats"])),
            chunks=(chunk_size,horizon_size,len(cfg["pred_feats"])),
            compression="gzip"
            )
    d_truth = g_pred.create_dataset(
            name="truth",
            shape=(num_smpl,horizon_size,len(cfg["pred_feats"])),
            chunks=(chunk_size,horizon_size,len(cfg["pred_feats"])),
            compression="gzip"
            )
    d_wdw = g_pred.create_dataset(
            name="window",
            shape=(num_smpl,cfg["window_size"],len(cfg["pred_feats"])),
            chunks=(chunk_size,cfg["window_size"],len(cfg["pred_feats"])),
            compression="gzip"
            )
    ## sample index
    d_sidx = g_pred.create_dataset(
            name="sample_idx",
            shape=(num_smpl,),
            chunks=chunk_size,
            )
    ## pivot index
    d_pidx = g_pred.create_dataset(
            name="pivot_idx",
            shape=(num_smpl,),
            chunks=chunk_size,
            )

    wdw_pred_idxs = tuple(
            cfg["window_feats"].index(f) for f in cfg["pred_feats"])
    pred_idx = 0
    ## For each sample chunk, make predictions and consolidate all predicted
    ## features in the window, horizon truth, and horizon predictions
    while len(chunks):
        print(chunks)
        W,H,S,P,T,sample_idxs,pivot_idxs = [],[],[],[],[],[],[]
        W_pred = []
        cur_chunk = chunks.pop(0)
        ## Extract and consolidate all the sample components
        for i in range(cur_chunk):
            X,Y,idxs = next(sample_gen)
            W.append(X["window"])
            W_pred.append(X["window"][...,wdw_pred_idxs])
            H.append(X["horizon"])
            S.append(X["static"])
            T.append(Y)
            ## Ignore file idxs since only one file is allowed
            sample_idxs.append(idxs[1])
            pivot_idxs.append(idxs[2])
        W = np.stack(W, axis=0)
        W_pred = np.stack(W_pred, axis=0)
        H = np.stack(H, axis=0)
        S = np.stack(S, axis=0)
        T = np.stack(T, axis=0)
        ## Progressively calculate the next horizon step for all samples.
        P = np.zeros((cur_chunk,horizon_size,len(cfg["pred_feats"])))
        for j in range(horizon_size):
            print(f"    horizon {j}")
            ## Get the next horizon feature input
            step_horizon = np.expand_dims(H[:,j], axis=1)
            ## Make a prediction for the current step
            new_pred = model({"window":W, "horizon":step_horizon, "static":S})
            ## Concatenate the prediction with the current step horizon feats
            ## in order to cycle them into the window feats for the next step
            new_window = np.concatenate((step_horizon,new_pred), axis=-1)
            ## Update the predictions with the new model output
            P[:,j,:] = np.squeeze(new_pred)
            ## Cycle the window features to include this step's prediction
            W = np.concatenate((W,new_window), axis=1)[:,1:]
        ## Update all the file fields and save the file
        d_pred[pred_idx:pred_idx+cur_chunk,...] = P * out_mean + out_stdev
        d_truth[pred_idx:pred_idx+cur_chunk,...] = T * out_mean + out_stdev
        d_wdw[pred_idx:pred_idx+cur_chunk,...] = W_pred * out_mean + out_stdev
        d_pidx[pred_idx:pred_idx+cur_chunk] = np.array(pivot_idxs)
        d_sidx[pred_idx:pred_idx+cur_chunk] = np.array(sample_idxs)
        f_pred.flush()
        pred_idx += cur_chunk
    return d_pred

def model_roll(model_path, sample_h5s, horizon_size=12, num_samples=1000):
    """ """
    model = load_model(model_path)
    cfg = mm.load_config(model_path.parent)
    ## Make sure the last features in the window are the predicted ones
    assert tuple(cfg["window_feats"][-len(cfg["pred_feats"]):]) \
            == tuple(cfg["pred_feats"])
    ## Get a sample generator for the full requested horizon size
    sample_gen = mm.gen_sample(
            h5_paths=list(sample_h5s),
            window_size=cfg["window_size"],
            horizon_size=horizon_size,
            window_feats=cfg["window_feats"],
            horizon_feats=cfg["horizon_feats"],
            pred_feats=cfg["pred_feats"],
            static_feats=cfg["static_feats"],
            as_tensor=False,
            )
    ## Consolidate all the sample components
    W,H,S,P,T = [],[],[],[],[]
    for i in range(num_samples):
        X,Y = next(sample_gen)
        W.append(X["window"])
        H.append(X["horizon"])
        S.append(X["static"])
        T.append(Y)
    W = np.stack(W, axis=0)
    H = np.stack(H, axis=0)
    S = np.stack(S, axis=0)
    T = np.stack(T, axis=0)
    ## Progressively calculate the next horizon step for all samples.
    P = np.zeros((num_samples,horizon_size,len(cfg["pred_feats"])))
    for j in range(horizon_size):
        step_horizon = np.expand_dims(H[:,j], axis=1)
        new_pred = model({"window":W, "horizon":step_horizon, "static":S})
        new_horizon = np.expand_dims(H[:,j], axis=1)
        new_window = np.concatenate((new_horizon,new_pred), axis=-1)
        P[:,j,:] = np.squeeze(new_pred)
        W = np.concatenate((W,new_window), axis=1)[:,1:]
    return P,T

if __name__=="__main__":
    #sample_h5 = Path("data/shuffle_2018.h5")
    sample_h5 = Path("/rstor/mdodson/thesis/shuffle_2018.h5")
    #sample_h5 = Path("/rstor/mdodson/thesis/shuffle_SEUS_2018.h5")

    """ --=( Self-cycling models )=-- """
    #model_dir = Path(f"data/models/dense-1")
    #model_path = model_dir.joinpath(f"dense-1_38_0.07.hdf5")
    #model_dir = Path(f"data/models/lstm-s2s-2")
    #model_path = model_dir.joinpath(f"lstm-s2s-2_015_0.24.hdf5")
    model_dir = Path(f"data/models/lstm-rec-1")
    model_path = model_dir.joinpath(f"lstm-rec-1_061_0.08.hdf5")

    #model_dir = Path(f"data/models-seus/lstm-rec-seus-0")
    #model_path = model_dir.joinpath(f"lstm-rec-seus-0_115_0.05.hdf5")
    #model_dir = Path(f"data/models-seus/dense-seus-0")
    #model_path = model_dir.joinpath(f"dense-seus-0_8_0.11.hdf5")
    #model_dir = Path(f"data/models-seus/lstm-s2s-seus-0")
    #model_path = model_dir.joinpath(f"lstm-s2s-seus-0_028_0.18.hdf5")

    #'''
    cfg = mm.load_config(model_dir)
    P = model_roll_chunks(
            model_path=model_path,
            sample_h5=sample_h5,
            pred_h5=Path(
                #f"/rstor/mdodson/thesis/pred_2018_SEUS_{cfg['model_name']}.h5"),
                f"/rstor/mdodson/thesis/pred_2018_{cfg['model_name']}_V2.h5"),
            chunk_size=512**2
            )
    #'''

    """ --=( Multi-horizon models )=-- """
    #model_dir = Path(f"data/models/lstm-s2s-5")
    #model_path = model_dir.joinpath(f"lstm-s2s-5_002_0.55.hdf5")
    #model_dir = Path(f"data/models/tcn-1")
    #model_path = model_dir.joinpath(f"tcn-1_092_0.03.hdf5")

    #model_dir = Path(f"data/models-seus/tcn-seus-0")
    #model_path = model_dir.joinpath(f"tcn-seus-0_099_0.04.hdf5")
    #model_dir = Path(f"data/models-seus/lstm-s2s-seus-1")
    #model_path = model_dir.joinpath(f"lstm-s2s-seus-1_004_0.47.hdf5")

    '''
    cfg = mm.load_config(model_dir)
    P = model_predict(
            model_path=model_path,
            sample_h5=sample_h5,
            pred_h5=Path(
                #f"/rstor/mdodson/thesis/pred_2018_SEUS_{cfg['model_name']}.h5"),
                f"/rstor/mdodson/thesis/pred_2018_{cfg['model_name']}_V2.h5"),
            chunk_size=256**2
            )
    '''
