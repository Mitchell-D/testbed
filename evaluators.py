from abc import ABC,abstractmethod
import numpy as np
import pickle as pkl
import random as rand
import json
import h5py
from datetime import datetime
from typing import Callable
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
import tensorflow as tf
import gc

import model_methods as mm
import tracktrain as tt
import generators

class Evaluator(ABC):
    @abstractmethod
    def __init__(self):
        pass
    @abstractmethod
    def add_batch(self):
        """ Update the partial evaluation data with a new batch of samples """
        pass
    @abstractmethod
    def get_results(self):
        """
        Collect the partial data from supplied batches into a dict of results
        formatted as the complete evaluation data this class produces.
        """
        pass
    @abstractmethod
    def to_pkl(self):
        pass
    @abstractmethod
    def from_pkl():
        pass

class EvalHorizon(Evaluator):
    def __init__(self, attrs={}):
        """ """
        self._counts = None ## Number of samples included in sums
        self._es_sum = None ## Sum of state error wrt horizon
        self._er_sum = None ## Sum of residual error wrt horizon
        self._es_var_sum = None ## State error partial variance sum
        self._er_var_sum = None ## Residual error partial variance sum
        self._attrs = attrs ## additional attributes

    def add_batch(self, inputs, true_state, predicted_residual):
        """ """
        ys,pr = true_state,predicted_residual
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the absolute error in the residual and state predictions
        es_abs = np.abs(ps - ys[:,1:,:])
        er_abs = np.abs(pr - yr)

        if self._counts is None:
            self._counts = es_abs.shape[0]
            self._es_sum = np.sum(es_abs, axis=0)
            self._er_sum = np.sum(er_abs, axis=0)
            self._es_var_sum = np.sum(
                    (es_abs-self._es_sum/self._counts)**2, axis=0)
            self._er_var_sum = np.sum(
                    (er_abs-self._er_sum/self._counts)**2, axis=0)
        else:
            self._counts += es_abs.shape[0]
            self._es_sum += np.sum(es_abs, axis=0)
            self._er_sum += np.sum(er_abs, axis=0)
            self._es_var_sum += np.sum(
                    (es_abs - self._es_sum/self._counts)**2, axis=0)
            self._er_var_sum += np.sum(
                    (er_abs - self._er_sum/self._counts)**2, axis=0)
        return

    def get_results(self):
        """ """
        return {
                "state_avg":self._es_sum/self._counts,
                "state_var":self._es_var_sum/self._counts,
                "residual_avg":self._er_sum/self._counts,
                "residual_var":self._er_var_sum/self._counts,
                "counts":self._counts,
                #"feats":pred_dict["pred_feats"],
                #"pred_coarseness":coarseness,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state horizon error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        r = self.get_results()
        attrs = {**self._attrs, **additional_attributes}
        pkl.dump({**r, "attrs":attrs}, pkl_path.open("wb"))

    def from_pkl(self, pkl_path:Path):
        """ """
        p = pkl.load(pkl_path.open("rb"))
        self._counts = p["counts"]
        self._es_sum = p["state_avg"] * self._counts
        self._er_sum = p["residual_avg"] * self._counts
        self._es_var_sum = p["state_var"] * self._counts
        self._er_var_sum = p["residual_var"] * self._counts

class EvalTemporal(Evaluator):
    def __init__(self, use_absolute_error=False, horizon_limit=None, attrs={}):
        """ """
        self._doy_r = None ## day of year residual error
        self._doy_s = None ## day of year static error
        self._doy_c = None ## day of year counts
        self._tod_r = None ## time of day residual error
        self._tod_s = None ## time of day static error
        self._tod_c = None ## time of day counts
        self.absolute_error = use_absolute_error
        self.horizon_limit = horizon_limit
        self._attrs = attrs

    def add_batch(self, inputs, true_state, predicted_residual):
        (_,_,_,_,th),ys,pr = inputs,true_state,predicted_residual
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        yr = ys[:,1:]-ys[:,:-1]

        ## Once the prediction shape is known declare the derived arrays
        if self._doy_r is None:
            self._doy_r = np.zeros((366, pr.shape[-1]))
            self._doy_s = np.zeros((366, pr.shape[-1]))
            self._doy_c = np.zeros((366, pr.shape[-1]), dtype=np.uint)
            self._tod_r = np.zeros((24, pr.shape[-1]))
            self._tod_s = np.zeros((24, pr.shape[-1]))
            self._tod_c = np.zeros((24, pr.shape[-1]), dtype=np.uint)

        times = list(map(
            datetime.fromtimestamp,
            th.astype(np.uint)[:,:self.horizon_limit].reshape((-1,))
            ))
        ## Times are reported exactly on the hour, but float rounding can cause
        ## some to be above or below. Add a conditional to account for this.
        tmp_tods = np.array([
            (t.hour+1 if t.minute >= 30 else t.hour)%24 for t in times
            ])
        tmp_doys = np.array([t.timetuple().tm_yday-1 for t in times])

        es = ps - ys[:,1:]
        er = pr - yr
        if self.absolute_error:
            es,er = map(np.abs,(es,er))
        es = es[:,:self.horizon_limit].reshape((-1, es.shape[-1]))
        er = er[:,:self.horizon_limit].reshape((-1, er.shape[-1]))

        for i in range(len(times)):
            self._doy_s[tmp_doys[i]] += es[i]
            self._doy_r[tmp_doys[i]] += er[i]
            self._doy_c[tmp_doys[i]] += 1
            self._tod_s[tmp_tods[i]] += es[i]
            self._tod_r[tmp_tods[i]] += er[i]
            self._tod_c[tmp_tods[i]] += 1

    def get_results(self):
        return {
                "doy_state":self._doy_s,
                "doy_residual":self._doy_r,
                "doy_counts":self._doy_c,
                "tod_state":self._tod_s,
                "tod_residual":self._tod_r,
                "tod_counts":self._tod_c,
                #"feats":pred_dict["pred_feats"],
                "absolute_error":self.absolute_error,
                "horizon_limit":self.horizon_limit,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state horizon error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        r = self.get_results()
        attrs = {**self._attrs, **additional_attributes}
        pkl.dump({**r, "attrs":attrs}, pkl_path.open("wb"))

    def from_pkl(self, pkl_path:Path):
        """ """
        p = pkl.load(pkl_path.open("rb"))
        self._doy_s = p["doy_state"]
        self._doy_r = p["doy_residual"]
        self._doy_c = p["doy_counts"]
        self._tod_s = p["tod_state"]
        self._tod_r = p["tod_residual"]
        self._tod_c = p["tod_counts"]
        self.absolute_error = p["absolute_error"]
        self.horizon_limit = p["horizon_limit"]
        self._attrs = p["attrs"]

class EvalStatic(Evaluator):
    def __init__(self, soil_idxs=None, use_absolute_error=False, attrs={}):
        ## Soil components to index mapping. Scuffed and slow, I know, but
        ## unfortunately I didn't store integer types alongside sequences,
        ## and it's too late to turn back now :(
        self._soil_mapping = list(map(
            lambda a:np.array(a, dtype=np.float32),
            [
                [0.,   0.,   0.  ],
                [0.92, 0.05, 0.03],
                [0.82, 0.12, 0.06],
                [0.58, 0.32, 0.1 ],
                [0.17, 0.7 , 0.13],
                [0.1 , 0.85, 0.05],
                [0.43, 0.39, 0.18],
                [0.58, 0.15, 0.27],
                [0.1 , 0.56, 0.34],
                [0.32, 0.34, 0.34],
                [0.52, 0.06, 0.42],
                [0.06, 0.47, 0.47],
                [0.22, 0.2 , 0.58],
                ]
            ))
        self._counts = np.zeros((14,13))
        self._err_res = None
        self._err_state = None
        self.absolute_error = use_absolute_error
        self.soil_idxs = soil_idxs
        self._attrs = attrs

    def add_batch(self, inputs, true_state, predicted_residual):
        """ """
        (_,_,s,si,_),ys,pr = inputs,true_state,predicted_residual
        if self._err_res is None:
            self._err_res = np.zeros((14,13,pr.shape[-1]))
            self._err_state = np.zeros((14,13,pr.shape[-1]))

        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr

        ## Average the error over the full horizon
        if self.absolute_error:
            es = np.abs(es)
            er = np.abs(er)
        es_avg = np.average(es, axis=1)
        er_avg = np.average(er, axis=1)

        soil_texture = s[...,self.soil_idxs]
        for i,soil_array in enumerate(self._soil_mapping):
            ## Get a boolean mask
            m_this_soil = np.isclose(soil_texture, soil_array).all(axis=1)
            if not np.any(m_this_soil):
                continue
            es_avg_subset = es_avg[m_this_soil]
            er_avg_subset = er_avg[m_this_soil]
            si_subset = si[m_this_soil]
            ## Convert the one-hot encoded vegetation vectors to indeces
            si_idxs = np.argwhere(si_subset)[:,1]
            for j in range(si_idxs.shape[0]):
                self._err_res[si_idxs[j],i] += er_avg_subset[j]
                self._err_state[si_idxs[j],i] += es_avg_subset[j]
                self._counts[si_idxs[j],i] += 1

    def get_results(self):
        """ Collect data from batches into a dict """
        return {
            "err_state":self._err_state,
            "err_residual":self._err_res,
            "counts":self._counts,
            "soil_idxs":self.soil_idxs,
            "use_absolute_error":self.absolute_error,
            #"feats":pred_dict["pred_feats"],
            }
    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Serialize the bulk data and attributes of this instance as a pkl
        """
        r = self.get_results()
        attrs = {**self._attrs, **additional_attributes}
        pkl.dump({**r, "attrs":attrs}, pkl_path.open("wb"))

    def from_pkl(self, pkl_path:Path):
        """
        Load the bulk data and attributes of a EvalStatic instance from a pkl
        file that has already been generated
        """
        p = pkl.load(pkl_path.open("rb"))
        self._err_state = p["err_state"]
        self._err_res = p["err_residual"]
        self._counts = p["err_residual"]
        self.soil_idxs = p["soil_idxs"]
        self.absolute_error = p["use_absolute_error"]
        self._attrs = p["attrs"]

class EvalJointHist(ABC):
    def __init__(self, ax1_args:tuple=None, ax2_args:tuple=None,
            use_absolute_error=False, ignore_nan=False, pred_coarseness=1,
            coarse_reduce_func="mean", attrs={}):
        """
        Initialize a histogram evaluator with 2 axes defined by tuples

        Specify a feature axis with a 2-tuple
        (
            (data_source, feat_idx),
            (val_min, val_max, num_bins)
        )

        Or specify a functional axis with 3-tuple (args, func, bounds) like:
        (
            ((data_source, feat_idx), (data_source, feat_idx), ...),
            func_or_lambda_str,
            (val_min, val_max, num_bins),
        )
        such that the first sub-tuple of (data_source, feat_idx) pairs provides
        the arguments for func_or_lambda_str

        data_source must be one of:
        {"horizon", "static", "true_res", "pred_res", "err_res",
         "true_state", "pred_state", "err_state"}

        :@param ax1_args: First (vertical) axis arguments as specified above
        :@param ax2_args: Second (horizontal) axis arguments as specified above
        :@param use_absolute_error: If True, calculate histograms based on
            the absolute value of error rather than the actual magnitude
        :@param ignore_nan: If True, NaN values encountered after a derived
            axis feature calculation will be ignored when histograms are binned
        :@param pred_coarseness: Include the model's coarseness argument so
            that arguments to derived axis feat calculations have the same
            number of elements along the sequence axis.
        :@param coarse_reduce_func: If functions output coarsened predictions,
            and axis arguments implement a function that uses horizon input
            data, a function must be used to reduce the inputs to the coarser
            resolution. Current choices include "min", "mean", and "max"
        """
        self._ax1_args_unevaluated = ax1_args
        self._ax2_args_unevaluated = ax2_args
        ## Validate axis arguments and evaluate any string lambda functions
        self._ax1_args,self._ax1_is_func = self._validate_axis_args(ax1_args)
        self._ax2_args,self._ax2_is_func = self._validate_axis_args(ax2_args)
        self.ignore_nan = ignore_nan
        self.absolute_error = use_absolute_error
        self._attrs = attrs
        self._counts = None
        self._coarse_reduce_str = coarse_reduce_func
        self.pred_coarseness = pred_coarseness
        self._rfuncs = {"min":np.amin, "mean":np.average, "max":np.amax}
        try:
            self._crf = self._rfuncs[coarse_reduce_func]
        except:
            raise ValueError(f"coarse_reduce_func must be in: {rfuncs.keys()}")

    @staticmethod
    def _validate_axis_args(axis_args):
        """ """
        if len(axis_args) == 2:
            is_func = False
        elif len(axis_args) == 3:
            is_func = True
            if type(axis_args[1]) == str:
                axis_args = (axis_args[0], eval(axis_args[1]), axis_args[2])
            else:
                assert isinstance(axis_args[1], Callable)
        else:
            raise ValueError(f"Invalid {axis_args =}")
        return axis_args,is_func

    def add_batch(self, inputs, true_state, predicted_residual):
        """ Update the partial evaluation data with a new batch of samples """
        (_,h,s,_,_),ys,pr = inputs,true_state,predicted_residual
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        ## Calculate the error in the residual and state predictions
        es = ps - ys[:,1:,:]
        er = pr - yr
        if self.absolute_error:
            es = np.abs(es)
            er = np.abs(er)
        ## Make a dict of the data arrays to make extraction easier
        if self.pred_coarseness != 1:
            b,_,f = h.shape
            h = h.reshape(h.shape[0],-1,self.pred_coarseness,h.shape[-1])
            h = self._crf(h, axis=2)
        data = {
                "horizon":h, "static":s,
                "true_res":yr, "pred_res":pr, "err_res":er,
                "true_state":ys[:,1:], "pred_state":ps, "err_state":es
                }
        ## Collect arguments and evaluate the method if ax1 is functional
        if self._ax1_is_func:
            args = [data[s][...,ix] for s,ix in self._ax1_args[0]]
            ax1 = self._ax1_args[1](*args)
        ## Otherwise just extract the data from the proper source array
        else:
            s,ix = self._ax1_args[0]
            ax1 = data[s][ix]
        ## Collect arguments and evaluate the method if ax2 is functional
        if self._ax2_is_func:
            args = [data[s][...,ix] for s,ix in self._ax2_args[0]]
            ax2 = self._ax2_args[1](*args)
        ## Otherwise just extract the data from the proper source array
        else:
            s,ix = self._ax2_args[0]
            ax2 = data[s][ix]

        ## extract bounds from the axis arguments
        ax1_min,ax1_max,ax1_bins = self._ax1_args[-1]
        ax2_min,ax2_max,ax2_bins = self._ax2_args[-1]

        ## declare the counts array if it hasn't already been declared
        if self._counts is None:
            self._counts = np.zeros((ax1_bins,ax2_bins), dtype=np.uint64)
        ## accumulate the predicted state time series
        ax1_idxs = np.reshape(
                self._norm_to_idxs(ax1, ax1_min, ax1_max, ax1_bins), (-1,))
        ax2_idxs = np.reshape(
                self._norm_to_idxs(ax2, ax2_min, ax2_max, ax2_bins), (-1,))

        if self.ignore_nan:
            ax1_idxs = ax1_idxs[np.isfinite(ax1_idxs)]
            ax2_idxs = ax2_idxs[np.isfinite(ax2_idxs)]
        ## Loop since fancy indexing doesn't accumulate repetitions
        for i in range(ax1_idxs.size):
            self._counts[ax1_idxs[i],ax2_idxs[i]] += 1

    @staticmethod
    def _norm_to_idxs(A:np.array, mins, maxs, num_bins):
        A = (np.clip(A, mins, maxs) - mins) / (maxs - mins)
        A = np.clip(np.floor(A * num_bins).astype(int), 0, num_bins-1)
        return A

    def get_results(self):
        """
        Collect the partial data from supplied batches into a dict of results
        formatted as the complete evaluation data this class produces.
        """
        return {
                "ax1_args":self._ax1_args_unevaluated,
                "ax2_args":self._ax2_args_unevaluated,
                "counts":self._counts,
                "use_absolute_error":self.absolute_error,
                "ignore_nan":self.ignore_nan,
                "pred_coarseness":self.pred_coarseness,
                "coarse_reduce_func":self._coarse_reduce_str,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Serialize the bulk data and attributes of this instance as a pkl
        """
        r = self.get_results()
        attrs = {**self._attrs, **additional_attributes}
        pkl.dump({**r, "attrs":attrs}, pkl_path.open("wb"))

    def from_pkl(self, pkl_path:Path):
        p = pkl.load(pkl_path.open("rb"))
        self._ax1_args_unevaluated = p["ax1_args"]
        self._ax1_args,self._ax1_is_func = self._validate_axis_args(
                self._ax1_args_unevaluated)
        self._ax2_args,self._ax2_is_func = self._validate_axis_args(
                self._ax2_args_unevaluated)
        self.absolute_error = p["use_absolute_error"]
        self.ignore_nan = p["ignore_nan"]
        self._counts = p["counts"]
        self.pred_coarseness = p["pred_coarseness"]
        self._coarse_reduce_str = p["coarse_reduce_func"]
        try:
            self._crf = rfuncs[self._coarse_reduce_str]
        except:
            raise ValueError(f"coarse_reduce_func must be in: {rfuncs.keys()}")

if __name__=="__main__":
    from list_feats import dynamic_coeffs,static_coeffs,derived_feats
    sequence_h5_dir = Path("data/sequences/")
    model_parent_dir = Path("data/models/new")
    pred_h5_dir = Path("data/predictions")
    error_horizons_pkl = Path(f"data/performance/error_horizons.pkl")
    temporal_pkl = Path(f"data/performance/temporal_absolute.pkl")
    hists_pkl = Path(f"data/performance/validation_hists_7d.pkl")
    static_error_pkl = Path(f"data/performance/static_error.pkl")

    ## Evaluate a single model over a series of sequence files, storing the
    ## results in new hdf5 files of predictions in the same order as sequences
    #'''
    #model_name = "snow-6"
    #weights_file = "lstm-7_095_0.283.weights.h5"
    #weights_file = "lstm-8_091_0.210.weights.h5"
    #weights_file = "lstm-14_099_0.028.weights.h5"
    #weights_file = "lstm-15_101_0.038.weights.h5"
    #weights_file = "lstm-16_505_0.047.weights.h5"
    #weights_file = "lstm-17_235_0.286.weights.h5"
    #weights_file = "lstm-19_191_0.158.weights.h5"
    #weights_file = "lstm-20_353_0.053.weights.h5"
    #weights_file = "lstm-21_522_0.309.weights.h5"
    #weights_file = "lstm-22_339_2.357.weights.h5"
    #weights_file = "lstm-23_217_0.569.weights.h5"
    #weights_file = "lstm-24_401_4.130.weights.h5"
    #weights_file = "lstm-25_624_3.189.weights.h5"
    #weights_file = "lstm-27_577_4.379.weights.h5"
    #weights_file = "snow-4_005_0.532.weights.h5"
    #weights_file = "snow-6_230_0.064.weights.h5"
    #weights_file = "snow-7_069_0.676.weights.h5"
    #weights_file = "lstm-rsm-1_458_0.001.weights.h5"
    #weights_file = "lstm-rsm-6_083_0.013.weights.h5"
    #weights_file = "lstm-rsm-9_231_0.003.weights.h5"
    #weights_file = None

    weights_file = "acclstm-rsm-4_249_0.002.weights.h5"
    model_name = "acclstm-rsm-4"
    model_label = f"{model_name}-249"

    ## Sequence hdf5s to avoid processing
    seq_h5_ignore = []

    md = tt.ModelDir(
            model_parent_dir.joinpath(model_name),
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                "acclstm":lambda args:mm.get_acclstm(**args),
                })
    ## Get a list of sequence hdf5s which will be independently evaluated
    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("ne", "nc", "nw", "se", "sc", "sw"),
            #region_strs=("nc",),
            season_strs=("warm", "cold"),
            #season_strs=("cold",),
            #time_strs=("2013-2018"),
            #time_strs=("2018-2023"),
            time_strs=("2018-2021", "2021-2024"),
            )

    ## Ignore min,max values prepended to dynamic coefficients in list_feats
    dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs}
    ## Arguments sufficient to initialize a generators.sequence_dataset
    seq_gen_args = {
            #"sequence_hdf5s":[p.as_posix() for p in seq_h5s],
            **md.config["feats"],
            "seed":200007221750,
            "frequency":1,
            "sample_on_frequency":True,
            "num_procs":3,
            "block_size":16,
            "buf_size_mb":128.,
            "deterministic":True,
            "shuffle":False,
            "yield_times":True,
            "dynamic_norm_coeffs":dynamic_norm_coeffs,
            "static_norm_coeffs":dict(static_coeffs),
            "derived_feats":derived_feats,
            }
    for h5_path in seq_h5s:
        if Path(h5_path).name in seq_h5_ignore:
            continue
        seq_gen_args["sequence_hdf5s"] = [h5_path]
        _,region,season,time_range = Path(h5_path).stem.split("_")
        pred_h5_path = pred_h5_dir.joinpath(
                f"pred_{region}_{season}_{time_range}_{model_label}.h5")
        sequence_preds_to_hdf5(
                model_dir=md,
                sequence_generator_args=seq_gen_args,
                pred_h5_path=pred_h5_path,
                chunk_size=128,
                gen_batch_size=128,
                weights_file_name=weights_file,
                pred_norm_coeffs=dynamic_norm_coeffs,
                )
    exit(0)
    #'''

    ## Establish sequence and prediction file pairings based on their
    ## underscore-separated naming scheme, which is expected to adhere to:
    ## (sequences file):   {file_type}_{region}_{season}_{period}.h5
    ## (prediction file):  {file_type}_{region}_{season}_{period}_{model}.h5
    #eval_regions = ("sw", "sc", "se")
    eval_regions = ("ne", "nc", "nw", "se", "sc", "sw")
    eval_seasons = ("warm", "cold")
    #eval_periods = ("2018-2023",)
    eval_periods = ("2018-2021", "2021-2024")
    #eval_models = ("lstm-17-235",)
    #eval_models = ("lstm-16-505",)
    #eval_models = ("lstm-19-191", "lstm-20-353")
    #eval_models = ("lstm-21-522", "lstm-22-339")
    #eval_models = ("lstm-23-217",)
    #eval_models = ("lstm-24-401", "lstm-25-624")
    #eval_models = ("snow-4-005",)
    #eval_models = ("snow-7-069",)
    #eval_models = ("lstm-rsm-6-083",)
    eval_models = ("lstm-rsm-9-231",)
    batch_size=2048
    buf_size_mb=128
    num_procs = 7

    """ Match sequence and prediction files, and parse name fields of both """
    seq_pred_files = [
            (s,p,tuple(pt[1:]))
            for s,st in map(
                lambda f:(f,f.stem.split("_")),
                sequence_h5_dir.iterdir())
            for p,pt in map(
                lambda f:(f,f.stem.split("_")),
                pred_h5_dir.iterdir())
            if st[0] == "sequences"
            and pt[0] == "pred"
            and pt[-1] in eval_models
            and st[1:4] == pt[1:4]
            and st[1] in eval_regions
            and st[2] in eval_seasons
            and st[3] in eval_periods
            ]

    ## Generate joint residual and state error histograms
    '''
    residual_bounds = {
            k[4:]:v[:2]
            for k,v in dynamic_coeffs
            if k[:4] == "res_"}
    state_bounds = {k:v[:2] for k,v in dynamic_coeffs}
    kwargs,id_tuples = zip(*[
        ({
            "sequence_h5":s,
            "prediction_h5":p,
            "pred_state_bounds":state_bounds,
            "pred_residual_bounds":residual_bounds,
            "num_bins":128,
            "batch_size":batch_size,
            "buf_size_mb":buf_size_mb,
            "horizon_limit":24*7,
            }, t)
        for s,p,t in seq_pred_files
        ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_joint_hists,kwargs)):
            ## Update the histograms pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if hists_pkl.exists():
                hists = pkl.load(hists_pkl.open("rb"))
            else:
                hists = {}
            hists[id_tuples[i]] = subdict
            pkl.dump(hists, hists_pkl.open("wb"))
    '''

    ## Evaluate the absolute error wrt static parameters for each pair
    '''
    args,id_tuples = zip(*[
            ((sfile, pfile, batch_size, buf_size_mb),id_tuple)
            for sfile, pfile, id_tuple in seq_pred_files
            ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_static_error,args)):
            ## Update the error horizons pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if static_error_pkl.exists():
                static_error = pkl.load(static_error_pkl.open("rb"))
            else:
                static_error = {}
            static_error[id_tuples[i]] = subdict
            pkl.dump(static_error, static_error_pkl.open("wb"))
    '''

    ## Evaluate the absolute error wrt horizon distance for each file pair
    '''
    args,id_tuples = zip(*[
            ((sfile, pfile, batch_size, buf_size_mb),id_tuple)
            for sfile, pfile, id_tuple in seq_pred_files
            ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_error_horizons,args)):
            ## Update the error horizons pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if error_horizons_pkl.exists():
                error_horizons = pkl.load(error_horizons_pkl.open("rb"))
            else:
                error_horizons = {}
            error_horizons[id_tuples[i]] = subdict
            pkl.dump(error_horizons, error_horizons_pkl.open("wb"))
    '''

    ## Calculate error rates with respect to day of year and time of day
    '''
    kwargs,id_tuples = zip(*[
            ({
                "sequence_h5":s,
                "prediction_h5":p,
                "batch_size":batch_size,
                "buf_size_mb":buf_size_mb,
                "horizon_limit":24*7,
                "absolute_error":True,
                }, t)
            for s,p,t in seq_pred_files
            ])
    with Pool(num_procs) as pool:
        for i,subdict in enumerate(pool.imap(mp_eval_temporal_error,kwargs)):
            ## Update the temporal pkl with the new model/file results,
            ## distinguished by their id_tuple (region,season,time_range,model)
            if temporal_pkl.exists():
                temporal = pkl.load(temporal_pkl.open("rb"))
            else:
                temporal = {}
            temporal[id_tuples[i]] = subdict
            pkl.dump(temporal, temporal_pkl.open("wb"))
    '''

    ## combine regions together for bulk statistics
    '''
    combine_years = ("2018-2021", "2021-2024")
    combine_model = "lstm-rsm-9-231"
    new_key = ("all", "all", "2018-2024", "lstm-rsm-9-231")
    combine_pkl = Path(
            "data/performance/performance-bulk_2018-2024_lstm-rsm-9-231.pkl")

    ## combine histograms
    hists = pkl.load(hists_pkl.open("rb"))
    combine_keys = [k for k in hists.keys()
            if k[3]==combine_model and k[2] in combine_years
            and k[1]!="all" and k[2] !="all"
            ]
    combo_hist = {}
    for k in combine_keys:
        if not combo_hist:
            hist_shape = hists[k]["state_hist"].shape
            combo_hist["state_hist"] = np.zeros(hist_shape, dtype=np.uint64)
            combo_hist["residual_hist"] = np.zeros(hist_shape, dtype=np.uint64)
            combo_hist["state_bounds"] = hists[k]["state_bounds"]
            combo_hist["residual_bounds"] = hists[k]["residual_bounds"]
            combo_hist["feats"] = hists[k]["feats"]
        combo_hist["state_hist"] += hists[k]["state_hist"]
        combo_hist["residual_hist"] += hists[k]["residual_hist"]
    hists[new_key] = combo_hist
    pkl.dump(hists, hists_pkl.open("wb"))

    ## combine static
    static = pkl.load(static_error_pkl.open("rb"))
    combine_keys = [k for k in static.keys()
            if k[3]==combine_model and k[2] in combine_years]
    combo_static = {}
    for k in combine_keys:
        stmp = static[k]
        ctmp = stmp["counts"][:,:,np.newaxis]
        if not combo_static:
            static_shape = stmp["err_state"].shape
            combo_static["err_state"] = np.zeros(static_shape)
            combo_static["err_residual"] = np.zeros(static_shape)
            combo_static["counts"] = np.zeros(
                    static_shape[:-1], dtype=np.uint64)
            combo_static["feats"] = stmp["feats"]
        combo_static["err_state"] += stmp["err_state"] * ctmp
        combo_static["err_residual"] += stmp["err_residual"] * ctmp
        combo_static["counts"] += stmp["counts"].astype(np.uint64)
    combo_static["err_state"] /= combo_static["counts"][:,:,np.newaxis]
    combo_static["err_residual"] /= combo_static["counts"][:,:,np.newaxis]
    m_zero = (combo_static["counts"] == 0)
    combo_static["err_state"][m_zero] = 0
    combo_static["err_residual"][m_zero] = 0
    static[new_key] = combo_static
    pkl.dump(static, static_error_pkl.open("wb"))

    ## combine horizons
    hor = pkl.load(error_horizons_pkl.open("rb"))
    combine_keys = [k for k in hor.keys()
            if k[3]==combine_model and k[2] in combine_years]
    combo_hor = {}
    for k in combine_keys:
        htmp = hor[k]
        if not combo_hor:
            hor_shape = htmp["state_avg"].shape
            combo_hor["state_avg"] = np.zeros(hor_shape)
            combo_hor["residual_avg"] = np.zeros(hor_shape)
            combo_hor["state_var"] = np.zeros(hor_shape)
            combo_hor["residual_var"] = np.zeros(hor_shape)
            combo_hor["counts"] = 0
            combo_hor["feats"] = htmp["feats"]
            combo_hor["pred_coarseness"] = htmp["pred_coarseness"]
        combo_hor["counts"] += htmp["counts"]
        combo_hor["state_avg"] += htmp["state_avg"] * htmp["counts"]
        combo_hor["residual_avg"] += htmp["residual_avg"] * htmp["counts"]
        combo_hor["state_var"] += htmp["state_var"] * htmp["counts"]
        combo_hor["residual_var"] += htmp["residual_var"] * htmp["counts"]
    combo_hor["state_avg"] /= combo_hor["counts"]
    combo_hor["residual_avg"] /= combo_hor["counts"]
    combo_hor["state_var"] /= combo_hor["counts"]
    combo_hor["residual_var"] /= combo_hor["counts"]
    hor[new_key] = combo_hor
    pkl.dump(hor, error_horizons_pkl.open("wb"))
    '''
