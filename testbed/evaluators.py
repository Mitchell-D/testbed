from abc import ABC,abstractmethod
from copy import deepcopy
import numpy as np
import pickle as pkl
from datetime import datetime
from typing import Callable
from pathlib import Path
import matplotlib.pyplot as plt

from testbed.plotting import plot_stats_1d

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

class EvalGridAxes(Evaluator):
    """
    Stores mean and stdev of values along a subset of axes separately for each
    batch. For gridded data, each batch is implied to correspond to a different
    init time. Ultimately, the dataset is (T, P, S, F) shape for T init times,
    P valid pixels, S sequence elements, and F features.

    The provided axis numbers are the ones that will be preserved, while error
    will be reduced along each of the remaining ones.

    This should work for any dataset as long as the sequence axis is always
    the 2nd one per batch.

    The features with respect to which bulk values are collected are defined
    by feat_args, which is a list of 2-tuples specifying either a stored or
    functional feature.

    Stored feature args corresponds to horizon, true, predicted, or error data
    that are explicitly returned by the generator. Stored feat args must be
    a 2-tuple like (data_source, feat_idx).

    Functional feature args recieve 1 or more arguments each defined by a
    stored feature configuration as specified above, and execute an arbitrary
    function according to the provided method.

    Functional feat args must be a 2-tuple like (args, func) where args is a
    list of 2-tuples (data_source, feat_idx) and func is a Callable or a string
    defining a lambda function.

    data_source must be one of:
    {"horizon", "true_res", "pred_res", "err_res",
     "true_state", "pred_state", "err_state"}
    """
    def __init__(self, feat_args=[], axes=tuple(), pred_coarseness=1,
            store_static=False, store_time=False, coarse_reduce_func="mean",
            use_absolute_error=False, attrs={}):
        """ """
        self._pred_coarseness = pred_coarseness
        self._axes = (axes,) if type(axes)==int else tuple(axes)
        self._counts = None
        self._batch_count = 0
        ## keep feat args with un-compiled lambda strings for serializability
        self._feat_args_unevaluated = feat_args
        self._feat_args,self._feat_is_func = zip(
                *map(self._validate_feat_arg,feat_args)
                ) if len(feat_args) else (None,None)
        self._sum = None ## Sum of feature wrt horizon
        self._var_sum = None ## State error partial variance sum
        self._store_static = store_static
        self._static = None
        self._store_time = store_time
        self._time = None
        self._indeces = None
        self._attrs = attrs ## additional attributes
        self._rfuncs = {"min":np.amin, "mean":np.average, "max":np.amax}
        self._absolute_error = use_absolute_error
        self._coarse_reduce_str = coarse_reduce_func
        try:
            self._crf = self._rfuncs[coarse_reduce_func]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")
    @property
    def attrs(self):
        return self._attrs

    @staticmethod
    def _validate_feat_arg(feat_arg):
        """
        Verify the validity of a feature-specifying argument, which may
        identify a horizon, true, predicted, or error data feature, or
        specify a function of one or more of the above.

        :@param feat_arg: feature arg 2-tuple following the format specified
            in the class docstrign
        :@return: 2-tuple (feat_arg, is_callable) where feat_arg has compiled
            function strings and is_callable specified whether the feature
            is functional.
        """
        sources = ("horizon", "true_res", "pred_res", "err_res",
                "true_state", "pred_state", "err_state")
        assert len(feat_arg)==2, feat_arg
        ## feat args are for stored feats iff they have type profile (str, int)
        if type(feat_arg[0])==str and type(feat_arg[1])==int:
            assert feat_arg[0] in sources,f"{feat_arg[0]} must be in {sources}"
            return feat_arg,False
        ## Otherwise it is a functional arg or invalid
        for arg in feat_arg[0]:
            _,is_func = EvalGridAxes._validate_feat_arg(arg)
            assert is_func, "Functional feat arg must itself be a stored " + \
                    "feat, not {arg}"
        if isinstance(feat_arg[1], str):
            axis_args = (axis_args[0], eval(axis_args[1]))
        else:
            assert isinstance(axis_args[1], Callable)
        return axis_args,True

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        (_,h,s,_,t),ys,pr = inputs,true_state,predicted_residual
        ## store grid indeces if requested, provided, and not done already
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
        ## the predicted state time series
        ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
        ## Calculate the label residual from labels
        yr = ys[:,1:]-ys[:,:-1]
        if self._static is None and self._store_static:
            self._static = s
        if self._store_time:
            if self._time is None:
                self._time = t[np.newaxis, -ys.shape[1]::self._pred_coarseness]
            else:
                tmpt = t[np.newaxis, -h.shape[1]::self._pred_coarseness]
                self._time = np.concatenate([self._time, tmpt], axis=0)
        es = ps - ys[:,1:]
        er = pr - yr
        if self._absolute_error:
            es = np.abs(es)
            er = np.abs(er)
        ## Make a dict of the data arrays to make extraction easier
        if self._pred_coarseness != 1:
            b,_,f = h.shape
            h = h.reshape(h.shape[0],-1,self._pred_coarseness,h.shape[-1])
            h = self._crf(h, axis=2)

        data = {"horizon":h, "true_res":yr, "pred_res":pr, "err_res":er,
                "true_state":ys[:,1:], "pred_state":ps, "err_state":es}
        feats = []
        for f,is_func in zip(self._feat_args, self._feat_is_func):
            ## Collect arguments and evaluate the method if feat is functional
            if is_func:
                args = [data[s][...,ix] for s,ix in f[0]]
                feats.append(f[1](*args))
            ## Otherwise just extract the data from the proper source array
            else:
                s,ix = f
                feats.append(data[s][...,ix])

        feats = np.stack(feats, axis=-1)

        ## Keep requested axes, and never reduce along the feature axis. Also
        ## ignore the first axis for now since it is only implied.
        r_axes = tuple([
                a+1 for a in range(len(feats.shape)-1)
                if a+1 not in self._axes
                ])
        ## set the counts for the sum/var arrays, which is the product of the
        ## number of elements along each marginalized axis
        self._batch_count += 1
        self._counts = np.prod([feats.shape[a] for a in r_axes]) \
                * [self._batch_count, 1][0 in self._axes]

        ## Create new init time axis
        feats = feats[None]
        tmp_sum = np.sum(feats, keepdims=True, axis=r_axes, dtype=np.float64)

        ## Case where batch axis is kept
        if 0 in self._axes:
            ## Only calculate variance within this timestep if not
            ## marginalizing over the first axis
            tmp_var = np.sum(
                    (feats - tmp_sum/self._counts)**2,
                    axis=r_axes, keepdims=True, dtype=np.float64)
            if self._sum is None:
                self._sum = tmp_sum
                self._var_sum = tmp_var
            else:
                self._sum = np.concatenate([self._sum, tmp_sum], axis=0)
                self._var_sum = np.concatenate([self._var_sum,tmp_var], axis=0)
        ## Case where batch axis is marginalized over
        else:
            ## For averaging over first axis, use mean values that gradually
            ## update over multiple batches to calculate variance
            if self._sum is None:
                self._sum = tmp_sum
                tmp_var = np.sum(
                        (feats - self._sum/self._counts)**2,
                        axis=r_axes, keepdims=True, dtype=np.float64)
                self._var_sum = tmp_var
            else:
                self._sum += tmp_sum
                tmp_var = np.sum(
                        (feats - self._sum/self._counts)**2,
                        axis=r_axes, keepdims=True, dtype=np.float64)
                self._var_sum += tmp_var

    def get_results(self):
        """ """
        return {
                "avg":self._sum,
                "var":self._var_sum,
                "static":self._static,
                "time":self._time,
                "counts":self._counts,
                "axes":self._axes,
                "indeces":self._indeces,
                "batch_count":self._batch_count,
                "feat_args":self._feat_args_unevaluated,
                "pred_coarseness":self._pred_coarseness,
                "coarse_reduce_func":self._coarse_reduce_str,
                "use_absolute_error":self._absolute_error,
                "attrs":self._attrs,
                }

    def concatenate(self, other:"EvalGridAxes", axis):
        """
        Concatenate a EvalGridAxes object with another one along an axis.
        I'm only really going to use it for the spatial axis, but I tried to
        write it to be more general. No promises it works though.
        """
        assert axis in self._axes, f"Concatenation axis {axis} must be one" + \
                f" of the preserved ones ({self._axes})"
        evr1 = self.get_results()
        evr2 = other.get_results()
        ## Assume by default all config comes from this object
        conc_data = deepcopy(evr1)
        conc_data.update({
                "avg":np.concatenate(
                    [evr1["avg"], evr2["avg"]], axis=axis),
                "var":np.concatenate(
                    [evr1["var"], evr2["var"]], axis=axis),
                })
        if all(not ix is None for ix in [evr1["indeces"],evr2["indeces"]]):
            conc_data["indeces"] = np.concatenate(
                    [evr1["indeces"], evr2["indeces"]], axis=0)
        if all(not ix is None for ix in [evr1["static"],evr2["static"]]):
            conc_data["static"] = np.concatenate(
                    [evr1["static"], evr2["static"]], axis=0)
        new_ev = EvalGridAxes()
        return new_ev.from_dict(conc_data)

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Write the residual and state horizon error results to a pkl file

        :@param pkl_path: Path to a non-existing pkl path to dump results to.
        :@param additional_attributes: Dict of additional information to
            include alongside the horizon error distribution data. If any of
            the keys match existing auxillary attributes the new ones provided
            here will replace them.
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        p = config_dict
        self._counts = p["counts"]
        self._sum = p["avg"]
        self._var_sum = p["var"]
        self._static = p["static"]
        self._time = p["time"]
        self._store_static = not self._static is None
        self._store_time = not self._time is None
        self._axes = p["axes"]
        self._feat_args_unevaluated = p["feat_args"]
        self._feat_args,self._feat_is_func = zip(
                *map(self._validate_feat_arg,p["feat_args"]))
        self._batch_count = p["batch_count"]
        self._indeces = p["indeces"]
        self._pred_coarseness = p.get("pred_coarseness", 1)
        self._coarse_reduce_str = p["coarse_reduce_func"]
        self._attrs = p["attrs"]
        self._absolute_error = p["use_absolute_error"]
        try:
            self._crf = self._rfuncs[self._coarse_reduce_str]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

class EvalHorizon(Evaluator):
    def __init__(self, pred_coarseness=1, attrs={}):
        """ """
        self._pred_coarseness = pred_coarseness
        self._counts = None ## Number of samples included in sums
        self._es_sum = None ## Sum of state error wrt horizon
        self._er_sum = None ## Sum of residual error wrt horizon
        self._es_var_sum = None ## State error partial variance sum
        self._er_var_sum = None ## Residual error partial variance sum
        self._attrs = attrs ## additional attributes
        self._indeces = None

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
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
            self._es_sum = np.sum(es_abs, axis=0, dtype=np.float64)
            self._er_sum = np.sum(er_abs, axis=0, dtype=np.float64)
            self._es_var_sum = np.sum(
                    (es_abs-self._es_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
            self._er_var_sum = np.sum(
                    (er_abs-self._er_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
        else:
            self._counts += es_abs.shape[0]
            self._es_sum += np.sum(es_abs, axis=0, dtype=np.float64)
            self._er_sum += np.sum(er_abs, axis=0, dtype=np.float64)
            self._es_var_sum += np.sum(
                    (es_abs - self._es_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
            self._er_var_sum += np.sum(
                    (er_abs - self._er_sum/self._counts)**2,
                    axis=0, dtype=np.float64)
        return

    def add(self, other:"EvalHorizon"):
        """
        Add the state and residual error sums and counts of multiple
        EvalHorizon instances
        """
        hor1 = self.get_results()
        hor2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(hor1)
        sum_fields = [ "state_avg", "state_var", "residual_avg",
                "residual_var", "counts"]
        ## Update the added data with the summed field
        new_data.update({f:hor1[f]+hor2[f] for f in sum_fields})
        if all(not ix is None for ix in [hor1["indeces"], hor2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [hor1["indeces"], hor2["indeces"]], axis=0)
        return EvalHorizon().from_dict(new_data)

    def get_results(self):
        """ """
        return {
                "state_avg":self._es_sum,
                "state_var":self._es_var_sum,
                "residual_avg":self._er_sum,
                "residual_var":self._er_var_sum,
                "counts":self._counts,
                "indeces":self._indeces,
                "pred_coarseness":self._pred_coarseness,
                "attrs":self._attrs,
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
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._counts = p["counts"]
        self._es_sum = p["state_avg"]
        self._er_sum = p["residual_avg"]
        self._es_var_sum = p["state_var"]
        self._er_var_sum = p["residual_var"]
        self._indeces = p["indeces"]
        self._pred_coarseness = p.get("pred_coarseness", 1)
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

    def plot(self, fig_path, feat_labels:list, state_or_res, plot_spec={},
            fill_between=True, fill_sigma=1., bar_sigma=1., class_space=1,
            use_stdev=True, yscale="linear", show=False):
        """  """
        domain = np.arange(self._es_sum.shape[0]) * self._pred_coarseness
        ps_def = {"line_width":2, "error_line_width":.5,"error_every":6,
                "fill_alpha":.25, "xticks":domain}
        ps = {**ps_def, **plot_spec}
        if state_or_res=="state":
            avg = self._es_sum / self._counts
            var = self._es_var_sum / self._counts
        elif state_or_res=="res":
            avg = self._er_sum / self._counts
            var = self._er_var_sum / self._counts
        else:
            raise ValueError(
                    f"{state_or_res = } must be one of 'state' or 'res'")
        if use_stdev:
            var = var**(1/2)
        plot_stats_1d(
                x_labels=domain,
                data_dict={
                    f:{"means":avg[...,i], "stdevs":var[...,i]}
                    for i,f in enumerate(feat_labels)
                    },
                fig_path=fig_path,
                fill_between=fill_between,
                fill_sigma=fill_sigma,
                class_space=class_space,
                bar_sigma=bar_sigma,
                yscale=yscale,
                plot_spec=ps,
                show=show,
                )

class EvalTemporal(Evaluator):
    def __init__(self, use_absolute_error=False, horizon_limit=None, attrs={}):
        """ """
        self._doy_r = None ## day of year residual error
        self._doy_s = None ## day of year static error
        self._doy_c = None ## day of year counts
        self._tod_r = None ## time of day residual error
        self._tod_s = None ## time of day static error
        self._tod_c = None ## time of day counts
        self._indeces = None
        self.absolute_error = use_absolute_error
        self.horizon_limit = horizon_limit
        self._attrs = attrs

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
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

    def add(self, other:"EvalTemporal"):
        """
        Add the state and residual error sums and counts of multiple
        EvalTemporal instances
        """
        hor1 = self.get_results()
        hor2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(hor1)
        sum_fields = [ "doy_state", "doy_residual", "doy_counts",
                "tod_state", "tod_residual", "tod_counts", ]
        ## Update the added data with the summed field
        new_data.update({f:hor1[f]+hor2[f] for f in sum_fields})
        if all(not ix is None for ix in [hor1["indeces"], hor2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [hor1["indeces"], hor2["indeces"]], axis=0)
        return EvalTemporal().from_dict(new_data)

    def get_results(self):
        return {
                "doy_state":self._doy_s,
                "doy_residual":self._doy_r,
                "doy_counts":self._doy_c,
                "tod_state":self._tod_s,
                "tod_residual":self._tod_r,
                "tod_counts":self._tod_c,
                #"feats":pred_dict["pred_feats"],
                "indeces":self._indeces,
                "absolute_error":self.absolute_error,
                "horizon_limit":self.horizon_limit,
                "attrs":self._attrs,
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
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._doy_s = p["doy_state"]
        self._doy_r = p["doy_residual"]
        self._doy_c = p["doy_counts"]
        self._tod_s = p["tod_state"]
        self._indeces = p["indeces"]
        self._tod_r = p["tod_residual"]
        self._tod_c = p["tod_counts"]
        self.absolute_error = p["absolute_error"]
        self.horizon_limit = p["horizon_limit"]
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

class EvalStatic(Evaluator):
    def __init__(self, soil_idxs=None, use_absolute_error=False, attrs={}):
        """"
        Extracts a combination matrix of surface types and soil textures
        for state and residual bias or residual error

        :@param soil_idxs: feature indeces for the (sand, silt, clay)
            components of the static array (in the above order of decreasing
            particle size).
        """
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
        self._indeces = None
        self.soil_idxs = soil_idxs
        self._attrs = attrs

    @property
    def attrs(self):
        return self._attrs

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ """
        (_,_,s,si,_),ys,pr = inputs,true_state,predicted_residual
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
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

    def add(self, other:"EvalStatic"):
        """
        Add the state and residual error sums and counts of multiple
        EvalHorizon instances
        """
        stat1 = self.get_results()
        stat2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(stat1)
        sum_fields = ["err_state", "err_residual", "counts"]
        ## Update the added data with the summed field
        new_data.update({f:stat1[f]+stat2[f] for f in sum_fields})
        if all(not ix is None for ix in [stat1["indeces"], stat2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [stat1["indeces"], stat2["indeces"]], axis=0)
        return EvalStatic().from_dict(new_data)

    def get_results(self):
        """ Collect data from batches into a dict """
        return {
            "err_state":self._err_state,
            "err_residual":self._err_res,
            "counts":self._counts,
            "soil_idxs":self.soil_idxs,
            "indeces":self._indeces,
            "use_absolute_error":self.absolute_error,
            "attrs":self._attrs,
            #"feats":pred_dict["pred_feats"],
            }
    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Serialize the bulk data and attributes of this instance as a pkl
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._err_state = p["err_state"]
        self._err_res = p["err_residual"]
        self._counts = p["counts"]
        self._indeces = p["indeces"]
        self.soil_idxs = p["soil_idxs"]
        self.absolute_error = p["use_absolute_error"]
        self._attrs = p["attrs"]
        return self

    def from_pkl(self, pkl_path:Path):
        """
        Load the bulk data and attributes of a EvalStatic instance from a pkl
        file that has already been generated
        """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

    def plot(self, plot_index:int, state_or_res="res", fig_path=None,
            show=False, plot_spec={}):
        """
        Generate a matrix plot of each soil and vegetation type combination
        as calculated by this object.
        """
        old_ps = {"cmap":"magma", "norm":"linear", "xlabel":"Soil type",
                "ylabel":"Surface type", "vmax":None}
        old_ps.update(plot_spec)
        plot_spec = old_ps

        soils = ["other", "sand", "loamy-sand", "sandy-loam", "silty-loam",
                "silt", "loam", "sandy-clay-loam", "silty-clay-loam",
                "clay-loam", "sandy-clay", "silty-clay", "clay"]
        vegs = ["water", "evergreen-needleleaf", "evergreen_broadleaf",
                "deciduous-needleleaf", "deciduous-broadleaf", "mixed-cover",
                "woodland", "wooded-grassland", "closed-shrubland",
                "open-shrubland", "grassland", "cropland", "bare", "urban"]

        static_error = {
                "state":self._err_state[...,plot_index],
                "res":self._err_res[...,plot_index],
                }[state_or_res] / self._counts

        fig,ax = plt.subplots()
        cb = ax.imshow(static_error, cmap=plot_spec.get("cmap"),
                vmax=plot_spec.get("vmax"), norm=plot_spec.get("norm"))
        fig.colorbar(cb)
        ax.set_xlabel(plot_spec.get("xlabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_ylabel(plot_spec.get("ylabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_title(plot_spec.get("title"),
                fontsize=plot_spec.get("title_size"))

        # Adding labels to the matrix
        ax.set_yticks(range(len(vegs)), vegs)
        ax.set_xticks(range(len(soils)), soils, rotation=45, ha='right',)
        if not fig_path is None:
            fig.savefig(fig_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
        return fig,ax

class EvalJointHist(ABC):
    def __init__(self, ax1_args:tuple=None, ax2_args:tuple=None,
            covariate_feature:tuple=None, use_absolute_error=False,
            ignore_nan=False, pred_coarseness=1, coarse_reduce_func="mean",
            attrs={}):
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
        :@param covariate_feature: Optional 2-tuple identifying
            (data_source, feat_idx) feature for which to capture an average
            value corresponding to each 2D value bin described by the axes
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
        self._cov_feat = covariate_feature
        self.ignore_nan = ignore_nan
        self.absolute_error = use_absolute_error
        self._attrs = attrs
        self._counts = None
        self._cov_sum = None
        self._indeces = None
        self._coarse_reduce_str = coarse_reduce_func
        self._pred_coarseness = pred_coarseness
        self._rfuncs = {"min":np.amin, "mean":np.average, "max":np.amax}
        try:
            self._crf = self._rfuncs[coarse_reduce_func]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")

    @property
    def attrs(self):
        return self._attrs

    @staticmethod
    def _validate_axis_args(axis_args):
        """
        """
        if axis_args is None:
            return (None, None)
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

    def add_batch(self, inputs, true_state, predicted_residual, indeces=None):
        """ Update the partial evaluation data with a new batch of samples """
        (_,h,s,_,_),ys,pr = inputs,true_state,predicted_residual
        if not indeces is None and self._indeces is None:
            self._indeces = indeces
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
        if self._pred_coarseness != 1:
            b,_,f = h.shape
            h = h.reshape(h.shape[0],-1,self._pred_coarseness,h.shape[-1])
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
            ax1 = data[s][...,ix]
        ## Collect arguments and evaluate the method if ax2 is functional
        if self._ax2_is_func:
            args = [data[s][...,ix] for s,ix in self._ax2_args[0]]
            ax2 = self._ax2_args[1](*args)
        ## Otherwise just extract the data from the proper source array
        else:
            s,ix = self._ax2_args[0]
            ax2 = data[s][...,ix]
        if self._cov_feat != None:
            s,ix = self._cov_feat
            cov = data[s][...,ix]
        else:
            cov = None

        ## extract bounds from the axis arguments
        ax1_min,ax1_max,ax1_bins = self._ax1_args[-1]
        ax2_min,ax2_max,ax2_bins = self._ax2_args[-1]

        ## declare the counts array if it hasn't already been declared
        if self._counts is None:
            self._counts = np.zeros((ax1_bins,ax2_bins), dtype=np.uint64)
            if self._cov_feat != None:
                self._cov_sum = np.zeros((ax1_bins,ax2_bins), dtype=np.float64)
        ## Cast the (batch,sequence) arrays for this feature as integer indeces
        ## corresponding to their value bin, and flatten them into a 1d array.
        ax1_idxs = np.reshape(
                self._norm_to_idxs(ax1, ax1_min, ax1_max, ax1_bins), (-1,))
        ax2_idxs = np.reshape(
                self._norm_to_idxs(ax2, ax2_min, ax2_max, ax2_bins), (-1,))

        m_valid = None
        if self.ignore_nan:
            m_valid = np.logical_and(
                    np.isfinite(ax1_idxs),
                    np.isfinite(ax2_idxs)
                    )
            ax1_idxs = ax1_idxs[m_valid]
            ax2_idxs = ax2_idxs[m_valid]
        if self._cov_feat != None:
            cov = np.reshape(cov, (-1,))
            if self.ignore_nan:
                cov = cov[m_valid]
        ## Loop since fancy indexing doesn't accumulate repetitions
        for i in range(ax1_idxs.size):
            self._counts[ax1_idxs[i],ax2_idxs[i]] += 1
            if self._cov_feat != None:
                self._cov_sum[ax1_idxs[i],ax2_idxs[i]] += cov[i]

    @staticmethod
    def _norm_to_idxs(A:np.array, mins, maxs, num_bins):
        A = (np.clip(A, mins, maxs) - mins) / (maxs - mins)
        A = np.clip(np.floor(A * num_bins).astype(int), 0, num_bins-1)
        return A

    def add(self, other:"EvalJointHist"):
        """
        Add the state and residual error sums and counts of multiple
        EvalHorizon instances
        """
        ejh1 = self.get_results()
        ejh2 = other.get_results()
        ## Assume by default all config comes from this object
        new_data = deepcopy(ejh1)
        sum_fields = ["covariate_sum", "counts"]
        ## Update the added data with the summed field
        new_data.update(
                {f:ejh1[f]+ejh2[f] \
                        if (not ejh1[f] is None and not ejh2[f] is None) \
                        else None for f in sum_fields}
                )
        if all(not ix is None for ix in [ejh1["indeces"], ejh2["indeces"]]):
            new_data["indeces"] = np.concatenate(
                    [ejh1["indeces"], ejh2["indeces"]], axis=0)
        return EvalJointHist().from_dict(new_data)

    def get_results(self):
        """
        Collect the partial data from supplied batches into a dict of results
        formatted as the complete evaluation data this class produces.
        """
        return {
                "ax1_args":self._ax1_args_unevaluated,
                "ax2_args":self._ax2_args_unevaluated,
                "covariate_feature":self._cov_feat,
                "covariate_sum":self._cov_sum,
                "counts":self._counts,
                "use_absolute_error":self.absolute_error,
                "indeces":self._indeces,
                "ignore_nan":self.ignore_nan,
                "pred_coarseness":self._pred_coarseness,
                "coarse_reduce_func":self._coarse_reduce_str,
                "attrs":self._attrs,
                }

    def to_pkl(self, pkl_path:Path, additional_attributes:dict={}):
        """
        Serialize the bulk data and attributes of this instance as a pkl
        """
        pkl.dump(self.get_results(), pkl_path.open("wb"))

    def from_dict(self, config_dict):
        """ """
        p = config_dict
        self._ax1_args_unevaluated = p["ax1_args"]
        self._ax2_args_unevaluated = p["ax2_args"]
        self._ax1_args,self._ax1_is_func = self._validate_axis_args(
                self._ax1_args_unevaluated)
        self._ax2_args,self._ax2_is_func = self._validate_axis_args(
                self._ax2_args_unevaluated)
        self.absolute_error = p["use_absolute_error"]
        self.ignore_nan = p["ignore_nan"]
        self._counts = p["counts"]
        self._indeces = p["indeces"]
        self._cov_sum = p["covariate_sum"]
        self._pred_coarseness = p["pred_coarseness"]
        self._coarse_reduce_str = p["coarse_reduce_func"]
        self._cov_feat = p["covariate_feature"]
        self._attrs = p["attrs"]
        try:
            self._crf = self._rfuncs[self._coarse_reduce_str]
        except:
            raise ValueError(f"coarse_reduce_func must be in: " + \
                    "{self._rfuncs.keys()}")
        return self

    def from_pkl(self, pkl_path:Path):
        """ """
        return self.from_dict(pkl.load(pkl_path.open("rb")))

    def plot(self, show_ticks=True, plot_covariate=False,
            separate_covariate_axes=False, plot_diagonal=False,
            normalize_counts=False, fig_path=None, nan_to_value=np.nan,
            cov_contour_levels=None, show=False, use_imshow=False,
            plot_spec={}):
        """ """
        # Merge provided plot_spec with un-provided default values
        old_ps = {
                "cmap":"nipy_spectral", "cb_size":1, "cb_orient":"vertical",
                "norm":"linear", "cov_levels":8, "cov_colors":None,
                "cov_linewidth":2, "cov_linestyles":"solid",
                "cov_cmap":"plasma", "cov_negative_linestyles":None,
                "xscale":"linear", "yscale":"linear", "cov_fontsize":"medium",
                **self.attrs.get("plot_spec", {})
                }
        old_ps.update(plot_spec)
        plot_spec = old_ps

        if self._cov_sum is None or not separate_covariate_axes:
            fig, ax = plt.subplots()
            cov_ax = None
        else:
            fig, (ax,cov_ax) = plt.subplots(1,2)

        if normalize_counts:
            heatmap = self._counts / np.sum(self._counts)
        else:
            heatmap = self._counts.astype(np.float64)

        if not self._cov_sum is None:
            cov = self._cov_sum / self._counts
            cov[np.logical_not(np.isfinite(cov))] = nan_to_value

        heatmap[np.logical_not(np.isfinite(heatmap))] = nan_to_value

        if plot_diagonal:
            ax.plot((0,heatmap.shape[1]-1), (0,heatmap.shape[0]-1),
                    linewidth=plot_spec.get("line_width"))
        y,x = np.meshgrid(
                np.linspace(*self._ax1_args[-1]),
                np.linspace(*self._ax2_args[-1])
                )
        if use_imshow:
            extent = (*self._ax2_args[-1][:2], *self._ax1_args[-1][:2])
            im = ax.imshow(
                    heatmap,
                    cmap=plot_spec.get("cmap"),
                    vmax=plot_spec.get("vmax"),
                    extent=extent,
                    norm=plot_spec.get("norm"),
                    origin="lower",
                    aspect=plot_spec.get("aspect")
                    )
            if plot_covariate \
                    and not self._cov_sum is None \
                    and separate_covariate_axes:
                cov_plot = ax.imshow(
                        cov,
                        cmap=plot_spec.get("cov_cmap"),
                        extent=extent,
                        origin="lower",
                        norm=plot_spec.get("cov_norm", "linear"),
                        aspect=plot_spec.get("aspect"),
                        vmax=plot_spec.get("cov_vmax"),
                        vmin=plot_spec.get("cov_vmin"),
                        )
                cov_cbar = fig.colorbar(
                        cov_plot,
                        orientation=plot_spec.get("cb_orient"),
                        label=plot_spec.get("cov_cb_label", ""),
                        shrink=plot_spec.get("cb_size", None),
                        )
        else:
            im = ax.pcolormesh(
                    x, y, heatmap.T,
                    cmap=plot_spec.get("cmap"),
                    vmax=plot_spec.get("vmax"),
                    norm=plot_spec.get("norm"),
                    )
            if plot_covariate \
                    and not self._cov_sum is None \
                    and separate_covariate_axes:
                cov_plot = cov_ax.pcolormesh(
                        x, y, cov.T,
                        cmap=plot_spec.get("cov_cmap"),
                        norm=plot_spec.get("cov_norm", "linear"),
                        vmax=plot_spec.get("cov_vmax"),
                        vmin=plot_spec.get("cov_vmin"),
                        )
                cov_cbar = fig.colorbar(
                        cov_plot,
                        orientation=plot_spec.get("cb_orient"),
                        label=plot_spec.get("cov_cb_label", ""),
                        shrink=plot_spec.get("cb_size", None),
                        )
        cbar = fig.colorbar(
                im, orientation=plot_spec.get("cb_orient"),
                label=plot_spec.get("cb_label"),
                shrink=plot_spec.get("cb_size")
                )
        if plot_covariate \
                and not self._cov_sum is None \
                and not separate_covariate_axes:
            cov_plot = ax.contour(
                    x, y, cov.T,
                    levels=plot_spec.get("cov_levels"),
                    colors=plot_spec.get("cov_colors"),
                    cmap=plot_spec.get("cov_cmap"),
                    linewidths=plot_spec.get("cov_linewidth"),
                    negative_linestyles=plot_spec.get(
                        "cov_negative_linestyles"),
                    )
            ax.clabel(con, fontsize=plot_spec.get("cov_fontsize"))
        if not show_ticks:
            plt.tick_params(axis="x", which="both", bottom=False,
                            top=False, labelbottom=False)
            plt.tick_params(axis="y", which="both", bottom=False,
                            top=False, labelbottom=False)

        plt.xlim(self._ax2_args[-1][:2])
        plt.ylim(self._ax1_args[-1][:2])

        #fig.suptitle(plot_spec.get("title"))
        fig.suptitle(plot_spec.get("title"))
        ax.set_xlabel(plot_spec.get("xlabel"))
        ax.set_ylabel(plot_spec.get("ylabel"))
        ax.set_yscale(plot_spec.get("yscale"))
        ax.set_xscale(plot_spec.get("xscale"))
        if plot_spec.get("aspect"):
            ax.set_box_aspect(plot_spec["aspect"])
        if plot_covariate \
                and not self._cov_sum is None \
                and separate_covariate_axes:
            #cov_ax.set_title(plot_spec.get("cov_title", ""))
            cov_ax.set_xlabel(plot_spec.get("cov_xlabel", ""))
            cov_ax.set_ylabel(plot_spec.get("cov_ylabel", ""))
            if plot_spec.get("aspect"):
                cov_ax.set_box_aspect(plot_spec["aspect"])

        if not plot_spec.get("x_ticks") is None:
            ax.set_xticks(plot_spec.get("x_ticks"))
        if not plot_spec.get("y_ticks") is None:
            ax.set_yticks(plot_spec.get("y_ticks"))
        if show:
            plt.show()
        if not fig_path is None:
            if plot_spec.get("fig_size"):
                fig.set_size_inches(plot_spec["fig_size"])
            fig.savefig(
                    fig_path.as_posix(),
                    dpi=plot_spec.get("dpi", 200),
                    bbox_inches=plot_spec.get("bbox_inches")
                    )
            print(f"Generated {fig_path.as_posix()}")
        plt.close()
        return fig,ax

if __name__=="__main__":
    pass
