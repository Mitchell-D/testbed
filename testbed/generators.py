import pickle as pkl
import numpy as np
import os
import sys
import h5py
import json
import random as rand
from datetime import datetime,timedelta
from pathlib import Path
from pprint import pprint as ppt

import tensorflow as tf

def parse_sequence_params(sequence_h5:Path):
    """ Simple method to extract the parameter dict from a sequence h5 """
    with h5py.File(sequence_h5, "r") as tmpf:
        params = json.loads(tmpf["data"].attrs["gen_params"])
    return params

def parse_prediction_params(prediction_h5:Path):
    """ Simple method to extract the parameter dict from a prediction h5 """
    with h5py.File(prediction_h5, "r") as tmpf:
        params = json.loads(tmpf["data"].attrs["gen_args"])
    return params

## TODO: finish transforms abstraction (see stub in list_feats)
def _resolve_transforms(out_feats:list, source_arrays:dict, transforms:dict):
    """
    :@param out_feats: List of string features or transforms to resolve
    :@param source_arrays: dict mapping a data source name to a 2-tuple
        (data_array, feat_labels)
    :@param transforms: dict mapping transform names to a 2-tuple (args, func)
        where args is a tuple of 2-tuples (source, feat) corresponding to the
        positional arguments to func, and func is a lambda string or function
        object taking the same number of positional arguments as args' elements
    """
    pass

def _parse_feat_idxs(out_feats, src_feats, static_feats, derived_feats,
        alt_feats:list=[]):
    """
    Helper for determining the Sequence indeces of stored features,
    and the output array indeces of derived features.

    :@param out_feats: Full ordered list of output features including
        dynamic stored and dynamic derived features
    :@param src_feats: Full ordered list of the features available in
        the main source array, which will be reordered and sub-set
        as needed to supply the ingredients for derived feats
    :@param static_feats: List of labels for static array features
    :@param alt_feats: If stored features can be retrieved from a
        different source array, provide a list of that array's feat
        labels here, and a third element will be included in the
        returned tuple listing the indeces of stored features with
        respect to alt_feats. These indeces will correspond in order
        to the None values in the stored feature index list
    :@return: 2-tuple (stored_feature_idxs, derived_data) where
        stored_feature_idxs is a list of integers indexing the
        array corresponding to src_feats, and derived_data is a
        4-tuple (out_idx,dynamic_arg_idxs,static_arg_idxs,lambda_func).
        If alt_feats are provided, a 3-tuple is returned instead
        with the third element being the indeces of features available
        only in the alternative array wrt the alternative feature list.
    """
    tmp_sf_idxs = [] ## stored feature idxs wrt src feats
    tmp_derived_data = [] ## derived feature idxs wrt output arrays
    tmp_alt_sf_idxs = [] ## alt stored feature idxs wrt alt_feats
    tmp_alt_out_idxs = [] ## alt stored feature idxs wrt out array
    for ix,l in enumerate(out_feats):
        if l not in src_feats:
            if l in alt_feats:
                tmp_alt_sf_idxs.append(alt_feats.index(l))
                tmp_alt_out_idxs.append(ix)
                tmp_sf_idxs.append(0)
            elif l in derived_feats.keys():
                assert l in derived_feats.keys()
                ## make a place for the derived features in the output
                ## array by temporarily indexing the first feature,
                ## to be overwritten when derived values are calc'd.
                tmp_sf_idxs.append(0)
                ## parse the derived feat arguments and function
                tmp_in_flabels,tmp_in_slabels,tmp_func = \
                        derived_feats[l]
                ## get derived func arg idxs wrt stored static/dynamic
                ## data; cannot yet support nested derived feats
                tmp_in_fidxs = tuple(
                    src_feats.index(q) for q in tmp_in_flabels)
                tmp_in_sidxs = tuple(
                        static_feats.index(q) for q in tmp_in_slabels)
                ## store (output_idx, dynamic_input_idxs,
                ##          static_input_idxs, derived_lambda_func)
                ## as 4-tuple corresponding to this single derived feat
                tmp_derived_data.append(
                        (ix,tmp_in_fidxs,tmp_in_sidxs,eval(tmp_func)))
            else:
                raise ValueError(
                        f"{l} not a stored, derived, or alt feature")
        else:
            tmp_sf_idxs.append(src_feats.index(l))

    alt_info = (tmp_alt_sf_idxs, tmp_alt_out_idxs)
    return tuple(tmp_sf_idxs),tmp_derived_data,alt_info

def _calc_feat_array(src_array, static_array,
        stored_feat_idxs:tuple, derived_data:list,
        alt_info=None, alt_array=None,
        alt_to_src_shape_slices:tuple=tuple()):
    """
    Compute a feature array including derived features and stored
    features from an alternative source array. This includes
    extracting and re-ordering a subset of source and alternative
    data features, as well as extracting ingredients for and
    computing derived data.

    Both stored_feat_idxs and derived_data, and optionally
    alt_info are outputs of _parse_feat_idxs

    stored_feat_idxs must include placeholder indeces where derived
    or alternative data is substituted. derived_data is a list
    of 4-tuples: (out_idx, dynamic_arg_idxs, static_arg_idxs, func)
    where out_idx specifies each derived output's location in the
    output array, *_arg_idxs are the indeces of the function inputs
    with respect to the source array, and func is the initialized
    lambda object associated with the transform.

    The optional alternative array of dynamic features may be the
    same shape or larger than the source array, As long as it can
    be adapted to the proper size.

    The alternative array ability is mainly used to provide a
    feature stored in the "pred" array of a sequence time series
    as an output in the "horizon" sequence array. The prediction
    array has samples covering the same time range as the horizon
    array, but including the timestep just prior to the first
    output. With the alternative functionality, features predicted
    by a different model (ie snow, runoff, canopy evaporation) may
    be substituted for the actual outputs.

    :@param src_array: Array-like main source of input data for
        the derived feature. The output shape will match this
        array's shape, except for the final (feature) axis.
    :@param static_array: Array like source of static data for
        derived features, which must contain a superset of all
        their ingredient features.
    :@param stored_feat_idxs: Ordered indeces of stored feats with
        respect to the source array, including placeholder values
        (typically 0) where derived/alternative feats are placed.
        This is an output of _parse_feat_idxs
    :@param derived_data: List of 4-tuples (see above) containing
        derived feature info and functions. This is an output of
        _parse_feat_idxs
    :@param alt_info: Optional 2-tuple of lists for alt feature
        indeces wrt the alt array and output array, respectively.
        This is also an output of _parse_feat_idxs.
    :@param alt_array: Alternative source array containing a
        superset of any alt feats requested in the output array.
    :@param alt_to_src_shape_slices: tuple of slice objects that
        correspond to the axes of alt_array, which reshape
        alt_array to the shape of src_array (except the feat dim).
    """
    ## Unfortunately, the Tensor API doesn't support fancy index gathering.
    src_is_tf = isinstance(src_array, tf.Tensor)
    static_is_tf = isinstance(static_array, tf.Tensor)
    alt_is_tf = isinstance(alt_array, tf.Tensor)

    ## All arrays are assumed to be uniformly numpy or tensors
    args_are_tf = [
            isinstance(a,tf.Tensor)
            for a in (src_array, static_array, alt_array)
            if not a is None
            ]
    if any(args_are_tf):
        if not all(args_are_tf):
            raise ValueError(
                    f"All array arguments must be either "
                    "numpy arrays or Tensors")
        src_array = src_array.numpy()
        static_array = static_array.numpy()
        alt_array = alt_array.numpy() if not alt_array is None else None

    ## Extract a numpy array around stored feature indeces, which
    ## should include placeholders for alt and derived feats
    sf_subset = src_array[...,stored_feat_idxs]

    ## Extract and substitute alternative features
    if not alt_array is None:
        ## 2 empty lists will extracts zero-element arrays that
        ## don't affect the stored feature subset
        if alt_info is None:
            alt_info = ([], [])
        ## alt array slc should be a tuple of slices
        slc = alt_to_src_shape_slices
        if type(slc) is slice:
            slc = (slc,)
        alt_sf_idxs,alt_out_idxs = alt_info
        ## slice the alt array to match the source array,
        ## and replace features sourced from alt data
        sf_subset[...,alt_out_idxs] = alt_array[*slc][...,alt_sf_idxs]

    ## Calculate and substitute derived features
    for (ix,dd_idxs,sd_idxs,fun) in derived_data:
        try:
            sf_subset[...,ix] = fun(
                    tuple(src_array[...,f] for f in dd_idxs),
                    tuple(static_array[...,f] for f in sd_idxs),
                    )
        except Exception as e:
            print(f"Error getting derived feat in position {ix}:")
            print(e)
            raise e
    if args_are_tf[0]:
        sf_subset = tf.convert_to_tensor(sf_subset)
    return sf_subset

def timegrid_sequence_dataset(
        timegrid_paths, window_size, horizon_size,
        window_feats, horizon_feats, pred_feats,
        static_feats, static_int_feats, static_conditions=[], derived_feats={},
        num_procs=1, deterministic=False, block_size=16, buf_size_mb=128,
        samples_per_timegrid=256, max_offset=0, sample_separation=1,
        include_init_state_in_predictors=False, load_full_grid=False,
        seed=None):
    """
    Versatile dataset generator for providing data samples consisting of
    a window, a horizon, a static vector, and a label (truth) vector
    using a timegrid hdf5 file with a superset of features.

    This method returns a tensorflow dataset object that generates the
    samples saved by make_sequence_hdf5. Data are returned in the same
    ("sequence") format as the sequence_dataset generator.

    Feature types:
    window : feats used to initialize the model prior to the first prediction
    horizon : covariate feats used to inform predictions
    pred : feats to be predicted by the model
    static : time-invariant feats used to parameterize dynamics
    static_int : 2-TUPLE like (feature_name, embed_size) of time-invariant
        integer features to be one-hot encoded in embed_size vector.

    :@param *_size: Number of timesteps from the pivot in the window or horizon
    :@param *_feats: String feature labels in order of appearence
    :@param static_conditions: Provide a list of 2-tuples with the first
        element containing a static feature label, and the second containing a
        string-encoded lambda function taking the corresponding array as an
        argument, and returning a boolean mask with the same shape, setting
        valid values to True. Use this to select a single soil type, or to
        encode threshold conditions for valid pixels, etc.
    :@param derived_feats: Provide a dict mapping NEW feature labels to a
        3-tuple (dynamic_args, static_args, lambda_str) where the args are
        each tuples of existing dynamic/static labels, and lambda_str contains
        a string-encoded function taking 2 arguments (dynamic,static) of tuples
        containing the corresponding arrays, and returns the subsequent new
        feature after calculating it based on the arguments. These will be
        invoked if the new derived feature label appears in one of the window,
        horizon, or pred feature lists.
    :@param num_procs: Number of generators to multithread over
    :@param deterministic: If True, always yields block_size samples per
        timegrid at a time; if False, multiple timegrids may inconsistently
        interleaved so block_size isn't always honored.
    :@param block_size: Number of consecutive samples drawn per timegrid
    :@param buf_size_mb: hdf5 buffer size for each timegrid file in MB
    :@param samples_per_timegrid: Maximum number of samples to yield from a
        single timegrid. If timegrids have fewer than this number that's okay.
    :@param max_offset: Maximum offset from the initial time in # timesteps.
        If nonzero, each pixel is given a random offset in [0,max_offset] that
        ensures all samples aren't observed at the same interval.
    :@param include_init_state_in_predictors: if True, the horizon features for
        the last observed state are prepended to the horizon array, which is
        necessary for forward-differencing if the network is predicting
        residual changes rather than absolute magnitudes.
    :@param load_full_grid: If True, loads the full timegrid to memory at once
        instead of paging from the memory-mapped hdf5 file.
    :@return: multiprocessed tensorflow generator dataset over the provided
        timegrid data.
    """
    ## establish the output signature for this generator as a 2-tuple like
    ## ((window, horizon, static, int_static, times), predictors)
    window_shape = (window_size, len(window_feats))
    horizon_shape = (horizon_size, len(horizon_feats))
    ## lengthen the prediction sequence by 1 to include the last observed
    ## (window) states as well, if requested. Used for forward-difference
    ## increment predictions evaluated in the loss function.
    pred_size = horizon_size + int(include_init_state_in_predictors)
    pred_shape = (pred_size, len(pred_feats))
    static_shape = (len(static_feats),)
    static_int_shape = (sum(tuple(zip(*static_int_feats))[1]),)
    time_shape = (window_size+horizon_size,)
    out_sig = ((
        tf.TensorSpec(shape=window_shape, dtype=tf.float32),
        tf.TensorSpec(shape=horizon_shape, dtype=tf.float32),
        tf.TensorSpec(shape=static_shape, dtype=tf.float32),
        tf.TensorSpec(shape=static_int_shape, dtype=tf.float32,),
        tf.TensorSpec(shape=time_shape, dtype=np.uint32),
        ), tf.TensorSpec(shape=pred_shape, dtype=tf.float32))

    def _gen_timegrid(h5_path):
        """
        Generator for a single timegrid hdf5

        1. Open the file and extract static data. Apply static constraints to
           develop a mask setting valid values in the spatial domain to True.
        2. One-hot encode and concatenate integer static values.
        3. Determine the latest valid initial time in the window.
        4. Get a random offset count of timesteps (up to stagger_offset)
           for each valid pixel's first initialization time.
        5. Generate a collection of initial times for each pixel's samples
           separated by sample_separation timesteps.
        6. Optionally shuffle the initial times, then extract, format, and
           yield subsequent samples one at a time.
        """
        h5_path = Path(h5_path) if type(h5_path)==str \
                else Path(h5_path.decode('ASCII')) if type(h5_path)==bytes \
                else h5_path
        print(f"\n{h5_path.name} ")
        ## Get a random number generator hashed with this file's name to
        ## prevent files with similar valid cells from having the same offsets.
        if not seed is None:
            proc_seed = abs(seed+hash(Path(h5_path).name))
        rng = np.random.default_rng(seed=proc_seed)

        F = h5py.File(
                h5_path,
                mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15,
                )

        ## (P,Q,F_s) static data grid
        S = F["/data/static"][...]

        ## (T,P,Q,F_d) dynamic data grid, optionally up-front loaded to memory
        if load_full_grid:
            D = F["/data/dynamic"][...]
        else:
            D = F["/data/dynamic"]

        ## Load the timesteps array
        T = F["/data/time"][...]

        ## static and dynamic information dictionaries
        sdict = json.loads(F["/data"].attrs["static"])
        ddict = json.loads(F["/data"].attrs["dynamic"])

        w_sf_idxs,w_derived,_ = _parse_feat_idxs(
                out_feats=window_feats,
                src_feats=ddict["flabels"],
                derived_feats=derived_feats,
                )
        h_sf_idxs,h_derived,_ = _parse_feat_idxs(
                out_feats=horizon_feats,
                src_feats=ddict["flabels"],
                derived_feats=derived_feats,
                )
        p_sf_idxs,p_derived,_ = _parse_feat_idxs(
                out_feats=pred_feats,
                src_feats=ddict["flabels"],
                derived_feats=derived_feats,
                )

        static_idxs = tuple(sdict["flabels"].index(l) for l in static_feats)

        ## Make a (P,Q) boolean mask setting grid points that meet the
        ## provided static conditions to True. The conditions should each
        ## be a lambda function stored as a string
        if static_conditions:
            valid = np.stack(
                    [eval(f)(S[...,sdict["flabels"].index(l)])
                        for l,f in static_conditions],
                    axis=-1)
            valid = np.logical_and.reduce(valid, axis=-1)
            if np.count_nonzero(valid) == 0:
                print(f"Warning: no valid pixels identified")
                print(valid.shape)
        else:
            valid = np.full(S.shape[:-1], True, dtype=bool)

        ## One-hot encode integer static features
        static_oh = np.concatenate([
                (np.arange(embed_size) == x[...,None]).astype(int)
                for x,embed_size in [
                    (S[...,sdict["flabels"].index(l)].astype(int), embed_size)
                    for l,embed_size in static_int_feats
                    ]
                ], axis=-1)

        ## Get a random offset for each valid grid point
        num_valid = np.count_nonzero(valid)
        if max_offset > 0:
            offsets = rng.integers(
                    low=0,
                    high=max_offset,
                    size=num_valid,
                    dtype=np.uint16
                    )[...,None]
        else:
            offsets = np.zeros((num_valid,1), dtype=np.uint16)

        ## full size of the valid initialization range
        min_init_range = D.shape[0]-window_size-horizon_size-max_offset
        ## Get a shuffled array of valid initialization indeces along the time
        ## axis, independently for each valid grid point
        start_idxs = np.broadcast_to(
                np.arange(min_init_range//sample_separation, dtype=np.uint16),
                shape=(num_valid, min_init_range//sample_separation),
                )
        start_idxs = rng.permuted(start_idxs, axis=1)
        ## (V,S) array of V valid spatial points at S initialization times
        start_idxs = start_idxs * sample_separation + offsets
        ## (V,2) array of V valid spatial points' 2D indeces
        grid_idxs = np.stack(np.where(valid), axis=1)
        ## (V,) counter array indicating the number of samples drawn from each
        ## valid grid point (keeps track of when to stop iterating).
        counter = np.zeros(start_idxs.shape[0], dtype=np.uint16)

        ## Loop over the total number of samples to return
        for i in range(min((samples_per_timegrid, start_idxs.size))):
            ## Get the index of the next sample in the valid array
            if start_idxs.shape[0] > 1:
                tmp_vidx = rng.integers(0, start_idxs.shape[0]-1)
            else:
                tmp_vidx = 0
            ## Get the 2d spatial index corresponding to the new valid index
            tmp_gidx = tuple(grid_idxs[tmp_vidx])
            ## Get the next start time index for the selected pixel
            tmp_sidx = start_idxs[tmp_vidx, counter[tmp_vidx]]
            counter[tmp_vidx] += 1
            ## Once a valid grid point's samples have been exhausted,
            ## remove it from the arrays.
            if counter[tmp_vidx] == start_idxs.shape[1]:
                start_idxs = np.delete(start_idxs, tmp_vidx, axis=0)
                counter = np.delete(counter, tmp_vidx, axis=0)
                grid_idxs = np.delete(grid_idxs, tmp_vidx, axis=0)

            ## collect static features
            tmp_static = S[*tmp_gidx,...]
            tmp_static_int = static_oh[*tmp_gidx]

            ## extract and separate dynamic features, and calculate any
            ## derived features for each sequence type
            seq_length = window_size+horizon_size
            tmp_dynamic = D[tmp_sidx:tmp_sidx+seq_length,
                    tmp_gidx[0],tmp_gidx[1],:]
            tmp_window = tmp_dynamic[:window_size, w_sf_idxs]
            tmp_horizon = tmp_dynamic[-horizon_size:, h_sf_idxs]
            tmp_pred = tmp_dynamic[-pred_size:, p_sf_idxs]

            tmp_window = _calc_feat_array(
                    src_array=tmp_window,
                    static_array=tmp_static,
                    stored_feat_idxs=w_sf_idxs,
                    derived_data=w_derived,
                    )
            tmp_horizon = _calc_feat_array(
                    src_array=tmp_horizon,
                    static_array=tmp_static,
                    stored_feat_idxs=h_sf_idxs,
                    derived_data=h_derived,
                    )
            tmp_pred = _calc_feat_array(
                    src_array=tmp_pred,
                    static_array=tmp_static,
                    stored_feat_idxs=p_sf_idxs,
                    derived_data=p_derived,
                    )

            '''
            for (ix,dd_idxs,sd_idxs,fun) in w_derived:
                tmp_window[...,ix] = fun(
                        tuple(tmp_dynamic[:window_size,f] for f in dd_idxs),
                        tuple(tmp_static[...,f] for f in sd_idxs),
                        )
            for (ix,dd_idxs,sd_idxs,fun) in h_derived:
                tmp_horizon[...,ix] = fun(
                        tuple(tmp_dynamic[-horizon_size:,f] for f in dd_idxs),
                        tuple(tmp_static[...,f] for f in sd_idxs),
                        )
            for (ix,dd_idxs,sd_idxs,fun) in p_derived:
                tmp_pred[...,ix] = fun(
                        tuple(tmp_dynamic[-pred_size:,f] for f in dd_idxs),
                        tuple(tmp_static[...,f] for f in sd_idxs),
                        )
            '''

            ## restrict static data to only what is requested
            tmp_static = tmp_static[..., static_idxs]

            ## collect the time segment
            tmp_time = T[tmp_sidx:tmp_sidx+seq_length]

            x = (tmp_window, tmp_horizon, tmp_static, tmp_static_int, tmp_time)
            y = tmp_pred
            yield (x,y)

    h5s = tf.data.Dataset.from_tensor_slices(
            list(map(lambda p:p.as_posix(), map(Path, timegrid_paths))))
    dataset = h5s.interleave(
            lambda fpath: tf.data.Dataset.from_generator(
                generator=_gen_timegrid,
                args=(fpath,),
                output_signature=out_sig,
                ),
            cycle_length=num_procs,
            num_parallel_calls=num_procs,
            block_length=block_size,
            deterministic=deterministic,
            )
    return dataset

def parse_timegrid_attrs(timegrid_path):
    """
    Parse the attributes attached to a timegrid hdf5 file as a 3-tuple.

    :@param timegrid_path: Path to a valid timegrid file
    :@return: (timestamp_array, static_attr_dict, dynamic_attr_dict)
    """
    with h5py.File(timegrid_path, mode="r") as tmpf:
        time = tmpf["/data/time"][...]
        static_attrs = json.loads(tmpf["/data"].attrs["static"])
        dynamic_attrs = json.loads(tmpf["/data"].attrs["dynamic"])
        return (time,dynamic_attrs, static_attrs)

def gen_timegrid_subgrids(
        timegrid_paths, window_size, horizon_size,
        window_feats, horizon_feats, pred_feats,
        static_feats, static_int_feats, init_pivot_epoch:float,
        final_pivot_epoch:float=None, pred_coarseness=1,
        derived_feats:dict={}, frequency=1,
        vidx_min=None, vidx_max=None, hidx_min=None, hidx_max=None,
        buf_size_mb=128, load_full_grid=False, max_delta_hours=2,
        include_init_state_in_predictors=False, seed=None,
        total_static_int_input_size=None, **kwargs):
    """
    Extracts gridded sequence samples at regular time intervals from timegrid
    hdf5 files populated by extract_timegrid, and yields them in chronological
    order.

    yields gridded samples in chronological order as a 2-tuple like:

    ( (W, H, S, SI, T), Y )

    W  : (T_w, Y, X, F_w)   F_w window features over window range T_w
    H  : (T_h, Y, X, F_h)   F_h horizon features over horizon range T_h
    S  : (Y, X, F_s)        F_s time-invariant static features
    SI : (Y, X, F_si)       F_si One-hot encoded static features
    T  : (T_w + T_h,)       Epoch float values over the full time range
    Y  : (T_y, Y, X, F_y)   F_y predicted features over predicted range T_y,
                            which is 1+T_h if include_init_state_in_predictors

    :@param *_size: Number of timesteps from the pivot in the window or horizon
    :@param *_feats: String feature labels in order of appearence
    :@param static_int_feats: list of 2-tuples like (feature_name,embed_size)
        of the static integer features to be one-hot encoded each with their
        respective size. These are returned concatenated and in order.
    :@param pred_coarseness: Frequency at which to return true values.
    :@param init_pivot_epoch: Inclusive initial valid time of the first
        prediction step. Note that this is the first step AFTER the window, so
        files must include enough time prior to this point for the window.
    :@param derived_feats: Provide a dict mapping unique feature labels to a
        3-tuple (dynamic_args, static_args, lambda_str) where the args are
        each tuples of existing dynamic/static labels, and lambda_str contains
        a string-encoded function taking 2 arguments (dynamic,static) of tuples
        containing the corresponding arrays, and returns the subsequent new
        feature after calculating it based on the arguments. These will be
        invoked if the new derived feature label appears in one of the window,
        horizon, or pred feature lists.
    :@param final_pivot_epoch: Exclusive final valid time of the first pred.
    :@param frequency: Number of timesteps in between the first prediction
        steps of consecutive yielded grids.
    :@param vidx_min: Minimum vertical index (downward from the North)
    :@param vidx_max: Maximum vertical index (downward from the North)
    :@param hidx_min: Minimum horizontal index (rightward from the West)
    :@param hidx_max: Maximum horizontal index (rightward from the West)
    :@param buf_size_mb: hdf5 buffer size for each timegrid file in MB
    :@param load_full_grid: If True, each full timegrid hdf5 is loaded up-front
        instead if being buffered from the hdf5. This has a much higher memory
        cost, but much faster access speed since way fewer IO requests.
    :@param max_delta_hours: Maximum amount of time in hours that samples may
        be separated before an error is raised. This is used for making sure
        consecutive timegrid files are close enough to be sampled across.
    :@param include_init_state_in_predictors: if True, the horizon features for
        the last observed state are prepended to the horizon array, which is
        necessary for forward-differencing if the network is predicting
        residual changes rather than absolute magnitudes.
    :@param total_static_int_input_size: see todo note below... :(
    """
    timegrid_paths = list(map(Path, timegrid_paths))
    assert all(p.exists() for p in timegrid_paths)
    times = []
    static_dicts = []
    dynamic_dicts = []
    ## Parse info dicts and timestamps from each file
    for p in timegrid_paths:
        assert p.exists(), p
        tmp_time,tmp_dynamic,tmp_static = parse_timegrid_attrs(p)
        times.append(tmp_time)
        static_dicts.append(tmp_static)
        dynamic_dicts.append(tmp_dynamic)

    ## Establish a prediction size based on whether to prepend an initial state
    pred_size = horizon_size + int(include_init_state_in_predictors)

    ## All of the timegrids' feature ordering must be uniform
    dynamic_labels = tuple(dynamic_dicts[0]["flabels"])
    static_labels = tuple(static_dicts[0]["flabels"])
    assert all(tuple(d["flabels"])==dynamic_labels for d in dynamic_dicts[1:])
    assert all(tuple(d["flabels"])==static_labels for d in static_dicts[1:])

    ## Make slices for the requested spatial bounds
    hslice = slice(hidx_min,hidx_max)
    vslice = slice(vidx_min,vidx_max)

    ## Determine the index ordering of requested features in the timegrids
    ## and get derived feature information
    w_fidx,w_derived,_ = _parse_feat_idxs(
        out_feats=window_feats,
        src_feats=dynamic_labels,
        static_feats=static_labels,
        derived_feats=derived_feats,
        )
    ## Get a list of horizon feats to extract from the pred array
    ## Only pred feats can be substituted for horizon feats
    ## (not vice-versa) since pred feats contain an extra initial value
    h_fidx,h_derived,_ = _parse_feat_idxs(
        out_feats=horizon_feats,
        src_feats=dynamic_labels,
        static_feats=static_labels,
        derived_feats=derived_feats,
        )
    p_fidx,p_derived,_ = _parse_feat_idxs(
        out_feats=pred_feats,
        src_feats=dynamic_labels,
        static_feats=static_labels,
        derived_feats=derived_feats,
        )

    s_idxs = tuple(static_labels.index(f) for f in static_feats)
    if len(static_int_feats) > 1:
        print(f"WARNING: generators.gen_timegrid_subgrids is not able to"
                "handle multiple static integer embeddings due to a config "
                "shortcoming that needs to be fixed. Currently the total "
                "embed size is assumed to be the size of every sint feat")
    ## TODO: For static ints, each feature's embed size should really be
    ## provided alongside its name in the "feat" ModelDir config subdict,
    ## but that isn't currently how things are set up. I need to change this
    ## in the future but don't want to cause side effects right now.
    si_idxs_embed = tuple(
            (static_labels.index(f), total_static_int_input_size)
            #for f,e in static_int_feats
            for f in static_int_feats
            )

    ## Collect each path in order with its valid range
    time_ranges = [(p,t[0],t[-1]) for t,p in zip(times,timegrid_paths)]
    conc_times = np.concatenate(times, axis=0)

    ## Make sure provided files are appropriately chronological
    file_diffs = list(zip(time_ranges[:-1], time_ranges[1:]))
    for p0,pf,dt in [(p0,pf,(i-f)) for (p0,_,f),(pf,i,_) in file_diffs]:
        if dt<0:
            raise ValueError(
                    "timegrid files must be ordered chronologically;",
                    f"currently {p0} followed by {pf}")
        if dt>max_delta_hours*60*60:
            raise ValueError(
                    "timegrid files must be adjacent in time; currently",
                    f"{pf} starts {dt} seconds the last time in {p0}")

    ## Only include files with time ranges intersecting the requested bounds.
    ## The user provides the initial and final 'pivot' times between the window
    ## and horizon, so the actual bounds are larger since they include the
    ## initial window and final horizon.
    init_pivot_idx = np.argmin(np.abs(conc_times-init_pivot_epoch))
    init_window_idx = init_pivot_idx - window_size
    final_pivot_idx = np.argmin(np.abs(conc_times-final_pivot_epoch))
    final_horizon_idx = final_pivot_idx + horizon_size
    if init_window_idx < 0:
        raise ValueError(
                "Timegrids must include data before the initial provided time",
                datetime.fromtimestamp(int(conc_times[0])),
                " for the first window."
                )
    if final_horizon_idx >= conc_times.size:
        raise ValueError(
                "Timegrids must include data after the final provided time",
                datetime.fromtimestamp(int(conc_times[-1])),
                " for the last horizon."
                )
    init_window_epoch = conc_times[init_window_idx]
    final_horizon_epoch = conc_times[final_horizon_idx]

    ## Reassign the times arrays to only include valid files
    valid_files,times = zip(*[
            (p,t) for (p,t0,tf),t in zip(time_ranges,times)
            if not tf < init_window_epoch and not t0 >= final_horizon_epoch
            ])
    conc_times = np.concatenate(times, axis=0)
    init_pivot_idx = np.argmin(np.abs(conc_times - init_pivot_epoch))
    init_window_idx = init_pivot_idx - window_size
    final_pivot_idx = np.argmin(np.abs(conc_times - final_pivot_epoch))

    ## Determine the index boundaries of each sample in the domain of times
    ## only including files overlapping the requested period
    d_pivot_idx = (final_pivot_idx-init_pivot_idx)
    if d_pivot_idx <= frequency:
        num_samples = 1
    else:
        num_samples = d_pivot_idx // int(frequency)
    init_idxs = np.arange(num_samples) * int(frequency) + init_window_idx
    final_idxs = init_idxs + window_size + horizon_size

    ## Get a list of files and their index bounds in the broader time domain.
    idx_accum = 0
    files_idx_bounds = []
    for f,t in zip(valid_files, times):
        files_idx_bounds.append((f,idx_accum,idx_accum+t.size))
        idx_accum += t.size

    ## Make a list of slices and corresponding files for each of the samples;
    ## some samples may span multiple files.
    grid_slices = []
    cur_file_idx = 0
    for idx0,idxf in zip(*map(list,(init_idxs,final_idxs))):
        cur_slices = []
        for f,fidx0,fidxf in files_idx_bounds:
            ## Start index within this file's index range
            if fidx0<=idx0<fidxf:
                ## end index also within this file's range
                if idxf<=fidxf:
                    cur_slices.append((f,slice(idx0-fidx0,idxf-fidx0)))
                ## end index beyond this file's range
                else:
                    cur_slices.append((f,slice(idx0-fidx0,fidxf-fidx0)))
                continue
            ## Start and end index ranges surround the entire file range
            elif idx0<=fidx0<idxf and idx0<=fidxf<idxf:
                cur_slices.append((f,slice(0,fidxf-fidx0)))
            ## End index within this file's index range, but not start index
            elif fidx0<idxf<=fidxf:
                cur_slices.append((f,slice(0,idxf-fidx0)))
        grid_slices.append(cur_slices)

    ## Extract subgrids from the timegrid files in chronological order,
    ## according to the sample format, concatenating across files if needed
    open_files = {}
    static_grid = None
    static_int_grid = None
    for sample in grid_slices:
        ## Close files that are no longer in use and remove from the dict
        del_keys = []
        for k in open_files.keys():
            if k not in [p for p,_ in sample]:
                open_files[k].close()
                del_keys.append(k)
        for k in del_keys:
            del open_files[k]
        ## Open new files and add them to the dict
        open_files.update({
            tmp_path:h5py.File(
                tmp_path, mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15
                )
            for tmp_path,_ in sample
            if tmp_path not in open_files.keys()
            })

        ## Extract the full dynamic grid associated with this sample
        tmp_dynamic_grid = np.concatenate(
                [open_files[f]["/data/dynamic"][s,vslice,hslice]
                    for f,s in sample],
                axis=0
                )
        t = np.concatenate(
                [open_files[f]["/data/time"][s] for f,s in sample],
                axis=0
                )

        if static_grid is None:
            tmp_static = open_files[sample[0][0]]["/data/static"]
            tmp_static = tmp_static[vslice,hslice]
            ## extract numeric static values
            s = tmp_static[...,s_idxs]
            ## one-hot encode static integers
            si = np.concatenate([
                    np.arange(embed_size) == \
                            tmp_static[...,idx].astype(int)[...,None]
                    for idx,embed_size in si_idxs_embed
                    ], axis=-1).astype(int)

        w = tmp_dynamic_grid[:window_size]
        h = tmp_dynamic_grid[-horizon_size:]
        p = tmp_dynamic_grid[-pred_size:]
        p = p[::pred_coarseness]

        w = _calc_feat_array(
                src_array=w,
                static_array=tmp_static,
                stored_feat_idxs=w_fidx,
                derived_data=w_derived,
                )
        h = _calc_feat_array(
                src_array=h,
                static_array=tmp_static,
                stored_feat_idxs=h_fidx,
                derived_data=h_derived,
                )
        p = _calc_feat_array(
                src_array=p,
                static_array=tmp_static,
                stored_feat_idxs=p_fidx,
                derived_data=p_derived,
                )

        yield (w,h,s,si,t),p

def make_sequence_hdf5(
        seq_h5_path, timegrid_paths, window_size, horizon_size,
        window_feats, horizon_feats, pred_feats,
        static_feats, static_int_feats, static_conditions=[], derived_feats={},
        num_procs=1, deterministic=False, block_size=16, buf_size_mb=128,
        samples_per_timegrid=256, max_offset=0, sample_separation=1,
        include_init_state_in_predictors=False, load_full_grid=False,
        seed=None, prefetch_count=1, batch_size=64, max_batches=None,
        samples_per_chunk=64, debug=False):
    """
    Create a new hdf5 file of training-ready samples extracted using a
    dataset generator from timegrid_sequence_dataset

    --(old parameters )--

    The majority of parameters are used to initialize a
    timegrid_sequence_dataset dataset generator, and are identical
    to those documented above

    --( new parameters )--

    :@param seq_h5_path: path to a new hdf5 file for extracted sequence samples
    :@param prefetch_count: Number of batches to concurrently prefetch
        dynamically during training.
    :@param batch_size: Size of batches drawn from the generator. This should
        be balanced with the volume of write operations to the new file and
        the memory required to prefetch batches in the background.
    :@param max_batches: Largest number of batches to draw from the generator.
        If None, continues to add to the new file until StopIteration.
    :@param samples_per_chunk: Chunks in the new hdf5 file are divided only
        along the sample axis, and modulated by setting this int argument.
    """
    ## establish the output signature for this generator as a 2-tuple like
    ## ((window, horizon, static, int_static), predictors)
    window_shape = (window_size, len(window_feats))
    horizon_shape = (horizon_size, len(horizon_feats))
    ## lengthen the prediction sequence by 1 to include the last observed
    ## (window) states as well, if requested. Used for forward-difference
    ## increment predictions evaluated in the loss function.
    pred_size = horizon_size + int(include_init_state_in_predictors)
    pred_shape = (pred_size, len(pred_feats))
    static_shape = (len(static_feats),)
    static_int_shape = (sum(tuple(zip(*static_int_feats))[1]),)
    time_shape = (window_size+horizon_size,)

    timegrid_dataset_params = {
        "timegrid_paths":timegrid_paths,
        "window_size":window_size,
        "horizon_size":horizon_size,
        "window_feats":window_feats,
        "horizon_feats":horizon_feats,
        "pred_feats":pred_feats,
        "static_feats":static_feats,
        "static_int_feats":tuple(map(tuple,static_int_feats)),
        "static_conditions":static_conditions,
        "derived_feats":derived_feats,
        "num_procs":num_procs,
        "deterministic":deterministic,
        "block_size":block_size,
        "buf_size_mb":buf_size_mb,
        "samples_per_timegrid":samples_per_timegrid,
        "max_offset":max_offset,
        "sample_separation":sample_separation,
        "include_init_state_in_predictors":include_init_state_in_predictors,
        "load_full_grid":load_full_grid,
        "seed":seed,
        }

    with h5py.File(seq_h5_path, "w") as f:
        ## Declare batched datasets for each input/output type
        W = f.create_dataset(
                name="/data/window",
                shape=(0, *window_shape),
                maxshape=(None, *window_shape),
                chunks=(samples_per_chunk, *window_shape),
                compression="gzip",
                )
        H = f.create_dataset(
                name="/data/horizon",
                shape=(0, *horizon_shape),
                maxshape=(None, *horizon_shape),
                chunks=(samples_per_chunk, *horizon_shape),
                compression="gzip",
                )
        Y = f.create_dataset(
                name="/data/pred",
                shape=(0, *pred_shape),
                maxshape=(None, *pred_shape),
                chunks=(samples_per_chunk, *pred_shape),
                compression="gzip",
                )
        S = f.create_dataset(
                name="/data/static",
                shape=(0, *static_shape),
                maxshape=(None, *static_shape),
                chunks=(samples_per_chunk, *static_shape),
                )
        SI = f.create_dataset(
                name="/data/static_int",
                shape=(0, *static_int_shape),
                maxshape=(None, *static_int_shape),
                chunks=(samples_per_chunk, *static_int_shape),
                )
        T = f.create_dataset(
                name="/data/time",
                shape=(0, *time_shape),
                maxshape=(None, *time_shape),
                chunks=(samples_per_chunk, *time_shape),
                )


        ## Serialize and include the params used to initialize the generator
        timegrid_dataset_params["timegrid_paths"] = \
                tuple(p.as_posix() for p in timegrid_paths)
        f["data"].attrs.update({
            "gen_params":json.dumps(timegrid_dataset_params),
            })

        ## Create a dataset generator using the parameters
        gen = timegrid_sequence_dataset(**timegrid_dataset_params)

        ## Use the generator to populate the new file with model-ready samples
        h5idx = 0
        batch_counter  = 0
        max_batches = (max_batches, -1)[max_batches is None]
        for (w,h,s,si,t),p in gen.batch(batch_size).prefetch(prefetch_count):
            ## First axis slice for the new batch
            sample_slice = slice(h5idx, h5idx+w.shape[0])
            if debug:
                print(f"Loading batch spanning ({sample_slice.start}, " + \
                        f"{sample_slice.stop})")
            h5idx += w.shape[0]

            ## Expand the memory bounds to fit the new batch
            W.resize((h5idx, *W.shape[1:]))
            H.resize((h5idx, *H.shape[1:]))
            Y.resize((h5idx, *Y.shape[1:]))
            S.resize((h5idx, *S.shape[1:]))
            SI.resize((h5idx, *SI.shape[1:]))
            T.resize((h5idx, *T.shape[1:]))

            ## Load the batch into the new file
            W[sample_slice,...] = w.numpy()
            H[sample_slice,...] = h.numpy()
            Y[sample_slice,...] = p.numpy()
            S[sample_slice,...] = s.numpy()
            SI[sample_slice,...] = si.numpy()
            T[sample_slice,...] = t.numpy()
            f.flush()

            batch_counter += 1
            if batch_counter == max_batches:
                break
        f.close()

def sequence_dataset(sequence_hdf5s:list, window_feats, horizon_feats,
        pred_feats, static_feats, static_int_feats, derived_feats:dict={},
        seed=None, shuffle=False, frequency=1, sample_on_frequency=True,
        num_procs=1, block_size=64, buf_size_mb=128., deterministic=False,
        yield_times:bool=False, pred_coarseness=1, dynamic_norm_coeffs:dict={},
        static_norm_coeffs:dict={}, static_conditions:list=[],
        horizon_conditions:list=[], pred_conditions:list=[],
        max_samples_per_file=None, debug=False, **kwargs):
        #use_residual_pred_coeffs:bool=False,  **kwargs):
    """
    get a tensorflow dataset that generates samples from sequence hdf5s,
    which must have been created by make_sequence_hdf5

    Note: Unlike make_sequence_hdf5 and timegrid_sequence_dataset, static feat
        labels provided as the static_int_feats argument should not be paired
        with a embedding size. All provided sequence hdf5s should already
        have uniform pre-embedding size (total number of categories).

    Yields 2-tuples (x, y) such that x is a tuple of Tensors and y is a Tensor
    with the shapes specified below

    x := (window:(B,S_w,F_w), horizon:(B,S_h,F_h),
          static:(B,F_s), static_int:(B,F_si))
    y := labels:(B, S_h, F_p)

    :@param sequence_hdf5s: List of Paths for sequence-style hdf5s created by
        generators.make_sequence_hdf5, which will be interleaved and returned
    :@param *_feats: Ordered list of features to return for each data category;
        must be a subset of the features present in every dataset here
    :@param derived_feats: Provide a dict mapping NEW feature labels to a
        3-tuple (dynamic_args, static_args, lambda_str) where the args are
        each tuples of existing dynamic/static labels, and lambda_str contains
        a string-encoded function taking 2 arguments (dynamic,static) of tuples
        containing the corresponding arrays, and returns the subsequent new
        feature after calculating it based on the arguments. These will be
        invoked if the new derived feature label appears in one of the window,
        horizon, or pred feature lists.

    --( shuffling and frequency-splitting )--

    :@param seed: Optional random seed determining the shuffle order of chunks
        returned from the sequence hdf5s
    :@param shuffle: If True, chunks and samples within each chunk are both
        separately shuffled. Otherwise, samples are returned in the order they
        appear in the original sequence file.
    :@param frequency: Integer frequency of chunks to randomly sample. For
        example, set to 3 to only return samples from floor(num_chunks/3)
        chunks. The specific chunks returned depend on the seed, because the
        frequency here determines the number of elements skipped when iterating
        over a shuffled index array.
    :@param sample_on_frequency: When True, samples are selected when the
        mod-frequency of the index is zero ; Inversely, when False, samples
        are selected when it is NON-zero. Therefore, if the seed is the same,
        sequences generated by two separate dataset instances acting on
        the same file (and using the same frequency) but with opposite values
        for sample_on_frequency will be mutually exclusive. This is useful for
        generating fairly uniform training and validation datasets.
    :@param normalize_preds: Boolean determining whether predicted features
        are linearly normalized along with the inputs. Since predictions are
        residuals, their mean should already be

    --( performance properties )--

    :@param num_procs: Number of generators to multithread over
    :@param block_size: Number of consecutive samples drawn per sequence hdf5
    :@param buf_size_mb: hdf5 buffer size for each timegrid file in MB
    :@param deterministic: Determines whether samples returned from concurrent
        dataset generators will always preserve their ordering. This value
        being set to False does not affect the chunks selected when frequency-
        -splitting data
    :@param yield_times: If True, yields the epoch timestep of each sequence
        step as a new (B,S_w+S_h) 5th element of the input tuple
    :@param pred_coarseness: Integer determining the frequency of predictions
        such that the first prediction output is the pred_coarseness'th step
        in the horizon, and there are (horizon_size // pred_coarseness) total
        outputs. In other words, this slices and returns output predictions
        like slice(pred_coarseness-1, horizon_size, pred_coarseness)

    --( sample selection conditions )--

    :@param static_conditions: optional way of restricting returned samples
        by providing 'OR' conditions as a list of 2-tuples (args, func) where
        args is a size F list of stored static feature labels and func is a
        string-encoded lambda function that takes a size F list of 1D arrays of
        static data corresponding to the arguments (each with size N) to a
        boolean arry with size N.
    :@param horizon_conditions: optional way of restricting returned samples
        by providing 'AND' conditions as a list of 2-tuples (args, func) where
        args is a size F list of stored horizon feature labels and func is a
        string-encoded lambda function that takes a size F list of 2D arrays of
        horizon data corresponding to the arguments (with shape (N,S_h)) to a
        boolean arry with size N.
    :@param horizon_conditions: optional way of restricting returned samples
        by providing 'AND' conditions as a list of 2-tuples (args, func) where
        args is a size F list of stored target feature labels and func is a
        string-encoded lambda function that takes a size F list of 2D arrays of
        target data corresponding to the arguments (with shape (N,S_h)) to a
        boolean arry with size N.
    """
    ## Make a pass over all the files to make sure the data is valid
    assert len(sequence_hdf5s), "There must be at least one sequence hdf5"
    window_size = None
    for tmp_path in sequence_hdf5s:
        tmp_file = h5py.File(tmp_path, "r")
        tmp_params = json.loads(tmp_file["data"].attrs["gen_params"])

        ## Verify that all requested features are present in their
        ## respective sample arrays
        output_feats = {
                "window_feats":window_feats,
                "horizon_feats":horizon_feats,
                "pred_feats":pred_feats,
                "static_feats":static_feats,
                #"static_int_feats":[s[0] for s in static_int_feats],
                }
        try:
            ## make sure all requested feature labels appear in the array,
            ## or are provided as a derived feature.
            for feat_type,feat_labels in output_feats.items():
                for l in feat_labels:
                    ## If a provided label is a derived feature, make sure all
                    ## the dynamic and static ingredients exist.
                    if l in derived_feats.keys():
                        df_labels = (*tmp_params[feat_type],
                                *derived_feats.keys())
                        sf_labels = (*tmp_params["static_feats"],
                                *derived_feats.keys())
                        assert all(
                                k in df_labels
                                for k in derived_feats[l][0]
                                ), "Not all dynamic ingredients exist for "+\
                                    f"derived feat {l}."+\
                                    f"\ningredients: {derived_feats[l][0]}"+\
                                    f"\nstored: {tmp_params[feat_type]}"

                        assert all(
                                k in sf_labels
                                for k in derived_feats[l][1]
                                ), "Not all static ingredients exist for"+\
                                    f"derived feat {l}."+\
                                    f"\ningredients: {derived_feats[l][1]}"+\
                                    f"\nstored: {tmp_params['static_feats']}"
                    ## Otherwise if not derived feature make sure it exists.
                    elif l not in tmp_params[feat_type]:
                        ## exception for only horizon features since prediction
                        ## feats can be substituded as an "alt feat" option
                        if feat_type=="horizon_feats" \
                                and l in tmp_params["pred_feats"]:
                            if debug:
                                print(f"Using {l} pred array for horizon feat")
                            continue
                        raise ValueError(
                            f"{l} not a derived feat or member of {feat_type}"
                            f"\n{derived_feats.keys() = }")
            ## Establish the sequence sizes or make sure they are uniform
            if window_size is None:
                window_size = tmp_params["window_size"]
                horizon_size = tmp_params["horizon_size"]
                include_init_state_in_predictors = \
                        tmp_params["include_init_state_in_predictors"]
                static_int_size = tmp_file["/data/static_int"].shape[-1]
                tmp_int_feats = [
                        s for s,_ in tmp_params["static_int_feats"]]
            else:
                assert window_size == tmp_params["window_size"]
                assert horizon_size == tmp_params["horizon_size"]
                assert include_init_state_in_predictors == \
                        tmp_params["include_init_state_in_predictors"]
                ## static int subsetting not currently supported
                assert tmp_file["/data/static_int"].shape[-1] == \
                            static_int_size
                assert tuple(tmp_int_feats) == tuple(static_int_feats)
        except Exception as e:
            raise e
        finally:
            tmp_file.close()

    ## establish the output signature for this generator as a 2-tuple like
    ## ((window, horizon, static, int_static, times), predictors)
    window_shape = (window_size, len(window_feats))
    horizon_shape = (horizon_size, len(horizon_feats))

    ## lengthen the prediction sequence by 1 to include the last observed
    ## (window) states as well, if requested. Used for forward-difference
    ## increment predictions evaluated in the loss function.
    pred_size = horizon_size // pred_coarseness \
            + int(include_init_state_in_predictors)
    pred_shape = (pred_size, len(pred_feats))
    static_shape = (len(static_feats),)
    static_int_shape = (static_int_size,)
    time_shape = (window_size+horizon_size,)

    in_sig = [
        tf.TensorSpec(shape=window_shape, dtype=tf.float32),
        tf.TensorSpec(shape=horizon_shape, dtype=tf.float32),
        tf.TensorSpec(shape=static_shape, dtype=tf.float32),
        tf.TensorSpec(shape=static_int_shape, dtype=tf.float32,),
        ]
    if yield_times:
        in_sig.append(tf.TensorSpec(shape=time_shape, dtype=np.uint32))
    out_sig = tf.TensorSpec(shape=pred_shape, dtype=tf.float32)
    sig = (tuple(in_sig), out_sig)

    ## (mean,stdev) values for each array feature
    norm_window = np.array([
        dynamic_norm_coeffs[k] if k in dynamic_norm_coeffs.keys() else [0,1]
        for k in window_feats
        ])[np.newaxis,np.newaxis,...]
    norm_horizon = np.array([
        dynamic_norm_coeffs[k] if k in dynamic_norm_coeffs.keys() else [0,1]
        for k in horizon_feats
        ])[np.newaxis,np.newaxis,...]
    norm_static = np.array([
        static_norm_coeffs[k] if k in static_norm_coeffs.keys() else [0,1]
        for k in static_feats
        ])[np.newaxis,...]

    ## Select dynamic coefficients with feature labels prepended "res_"
    ## if residual predictor coefficients are requested
    norm_pred = np.array([
        #dynamic_norm_coeffs[ (k, f"res_{k}")[use_residual_pred_coeffs] ]
        dynamic_norm_coeffs[k]
        if k in dynamic_norm_coeffs.keys() else [0,1]
        for k in pred_feats
        ])[np.newaxis,np.newaxis,...]

    generic_seed = seed
    def _gen_samples(seq_h5):
        seq_h5 = Path(seq_h5) if type(seq_h5)==str \
                else Path(seq_h5.decode('ASCII')) if type(seq_h5)==bytes \
                else seq_h5
        if debug:
            print(f"\nGenerating from {seq_h5.name}")
        F = h5py.File(
                seq_h5,
                mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15,
                )

        ## Seed with the file hash so that if the same file is passed to a
        ## different generator, chunks will be shuffled the same way, but
        ## different files will have independently random chunk shuffling.
        if not generic_seed is None:
            tmp_seed = generic_seed+abs(int(hash(seq_h5.name)))
        else:
            tmp_seed = generic_seed
        rng = np.random.default_rng(seed=tmp_seed)

        ## Get index tuples mapping file features to the requested order
        tmp_params = json.loads(F["data"].attrs["gen_params"])
        inc_init = tmp_params["include_init_state_in_predictors"]

        ## Parse the stored feat indeces and derived feature indeces,
        ## ingredient indeces, and recipes.
        w_fidx,w_derived,_ = _parse_feat_idxs(
            out_feats=window_feats,
            src_feats=tmp_params["window_feats"],
            static_feats=tmp_params["static_feats"],
            derived_feats=derived_feats,
            )
        ## Get a list of horizon feats to extract from the pred array
        ## Only pred feats can be substituted for horizon feats
        ## (not vice-versa) since pred feats contain an extra initial value
        h_fidx,h_derived,h_alt = _parse_feat_idxs(
            out_feats=horizon_feats,
            src_feats=tmp_params["horizon_feats"],
            static_feats=tmp_params["static_feats"],
            derived_feats=derived_feats,
            alt_feats=tmp_params["pred_feats"],
            )
        p_fidx,p_derived,_ = _parse_feat_idxs(
            out_feats=pred_feats,
            src_feats=tmp_params["pred_feats"],
            static_feats=tmp_params["static_feats"],
            derived_feats=derived_feats,
            )
        s_fidx = tuple(tmp_params["static_feats"].index(l)
                for l in static_feats)

        ## Get the functions needed to evaluate sample validity
        sc_idxs_funcs = []
        for args,func in static_conditions:
            sc_idxs = tuple(
                    tmp_params["static_feats"].index(l)
                    for l in args
                    )
            sc_idxs_funcs.append((sc_idxs,eval(func)))
        hc_idxs_funcs = []
        for args,func in horizon_conditions:
            hc_idxs = tuple(
                    tmp_params["horizon_feats"].index(l)
                    for l in args
                    )
            hc_idxs_funcs.append((hc_idxs,eval(func)))
        pc_idxs_funcs = []
        for args,func in pred_conditions:
            pc_idxs = tuple(
                    tmp_params["pred_feats"].index(l)
                    for l in args
                    )
            pc_idxs_funcs.append((pc_idxs,eval(func)))

        ## Get slices along the batch axis identifying each chunk and shuffle
        ## an index array to homogenize the data.
        batch_chunk_slices = sorted([
            s[0] for s in F["/data/window"].iter_chunks()
            ])
        chunk_idxs = np.arange(len(batch_chunk_slices))
        ## Shuffle chunks in the file
        if shuffle:
            rng.shuffle(chunk_idxs)

        ## Make a bool mask with indeces divisible by frequency set to True
        on_frequency = ((np.arange(len(batch_chunk_slices)) % frequency) == 0)
        ## If sampling on frequency preserve chunk indeces with the mask True
        ## Otherwisee preserve indeces with mask set to False. This allows
        ## sample_off_mod_frequency to act as a switch for training/validation
        ## data as long as separate instances of this generator are seeded same
        if sample_on_frequency:
            chunk_idxs = chunk_idxs[on_frequency]
        else:
            chunk_idxs = chunk_idxs[np.logical_not(on_frequency)]

        ## Establish the number of chunks associated with the maximum number
        ## of samples (optionally specified by the user)
        chunk_size = batch_chunk_slices[0].stop-batch_chunk_slices[0].start
        if max_samples_per_file:
            last_chunk = max_samples_per_file // chunk_size
            last_chunk_samples = max_samples_per_file % chunk_size
            chunk_idxs = chunk_idxs[:last_chunk+1]

        ## Extract valid chunks one at a time.
        for i in range(chunk_idxs.size):
            tmp_slice = batch_chunk_slices[chunk_idxs[i]]

            ## Extract data within the current slice
            m_valid = np.full(tmp_slice.stop-tmp_slice.start, True)
            tmp_static = F["/data/static"][tmp_slice,...]
            tmp_window = F["/data/window"][tmp_slice,...]
            tmp_horizon = F["/data/horizon"][tmp_slice,...]
            tmp_pred = F["/data/pred"][tmp_slice,...]

            ## evaluate static conditions and restrict returned pixels
            ## to only those meeting ONE of the conditions.
            tmp_masks = []
            for sidxs,func in sc_idxs_funcs:
                args = [ tmp_static[...,ix] for ix in sidxs ]
                tmp_masks.append(func(args))
            if len(tmp_masks)>0:
                m_tmp = np.any(np.stack(tmp_masks, axis=-1), axis=-1)
                m_valid = m_valid & m_tmp

            ## evaluate horizon conditions and restrict returned samples to
            ## only those meeting ALL of the conditions. Functions should
            ## return a boolean array along the first (sample) axis.
            tmp_masks = []
            for hidxs,func in hc_idxs_funcs:
                args = [ tmp_horizon[...,ix] for ix in hidxs ]
                tmp_masks.append(func(args))
            if len(tmp_masks)>0:
                m_tmp = np.all(np.stack(tmp_masks, axis=-1), axis=-1)
                m_valid = m_valid & m_tmp

            ## same with prediction array conditions
            tmp_masks = []
            for hidxs,func in pc_idxs_funcs:
                args = [ tmp_pred[...,ix] for ix in pidxs ]
                tmp_masks.append(func(args))
            if len(tmp_masks)>0:
                m_tmp = np.all(np.stack(tmp_masks, axis=-1), axis=-1)
                m_valid = m_valid & m_tmp

            if np.count_nonzero(m_valid) == 0:
                if debug:
                    print(f"Skipping invalid chunk")
                continue

            ## use the valid mask to shuffle and subset samples from the chunk
            cidxs = np.arange(m_valid.size)[m_valid]
            ## Shuffle data within an extracted chunk
            if shuffle:
                rng.shuffle(cidxs)
            ## Check if chunk reaches the maximum number of samples allowed
            if max_samples_per_file:
                if i == last_chunk:
                    cidxs = cidxs[:last_chunk_samples]
                if i > last_chunk:
                    break
            tmp_static = tmp_static[cidxs]
            tmp_window = tmp_window[cidxs]
            tmp_horizon = tmp_horizon[cidxs]
            tmp_pred = tmp_pred[cidxs]

            ## Extract stored and derived feats for each dynamic component of
            ## the sequence generator output
            tmp_window = _calc_feat_array(
                    src_array=tmp_window,
                    static_array=tmp_static[:,np.newaxis],
                    stored_feat_idxs=w_fidx,
                    derived_data=w_derived,
                    )
            tmp_horizon = _calc_feat_array(
                    src_array=tmp_horizon,
                    static_array=tmp_static[:,np.newaxis],
                    stored_feat_idxs=h_fidx,
                    derived_data=h_derived,
                    alt_info=h_alt,
                    alt_array=tmp_pred,
                    alt_to_src_shape_slices=(slice(0,None),slice(1,None)),
                    )
            tmp_pred = _calc_feat_array(
                    src_array=tmp_pred,
                    static_array=tmp_static[:,np.newaxis],
                    stored_feat_idxs=p_fidx,
                    derived_data=p_derived,
                    )

            ## Coarsen prediction steps to the requested resolution.
            ## If prediction sequences include the prepended initial state
            ## prior to the first prediction state, the first element index in
            ## the sliced sequence would be 0 instead of pred_coarseness-1.
            if inc_init:
                tmp_pred = tmp_pred[:,::pred_coarseness]
            else:
                tmp_pred = tmp_pred[:,pred_coarseness-1::pred_coarseness]

            ## subset the static features to only the requested ones
            tmp_static = tmp_static[...,s_fidx]
            ## static int subsetting not currently supported
            tmp_static_int = F["/data/static_int"][tmp_slice,...][cidxs]
            tmp_time = F["/data/time"][tmp_slice,...][cidxs]

            ## scale by normalization coefficients
            tmp_window = (tmp_window-norm_window[...,0])/norm_window[...,1]
            tmp_horizon = (tmp_horizon-norm_horizon[...,0])/norm_horizon[...,1]
            tmp_static = (tmp_static-norm_static[...,0])/norm_static[...,1]
            tmp_pred = (tmp_pred-norm_pred[...,0])/norm_pred[...,1]

            ## yield chunk samples one at a time
            for j in range(tmp_window.shape[0]):
                x = [tmp_window[j], tmp_horizon[j],
                        tmp_static[j], tmp_static_int[j]]
                if yield_times:
                    x.append(tmp_time[j])
                y = tmp_pred[j]
                yield (tuple(x),y)

    h5s = tf.data.Dataset.from_tensor_slices(
            list(map(lambda p:p.as_posix(), map(Path, sequence_hdf5s))))
    dataset = h5s.interleave(
            lambda fpath: tf.data.Dataset.from_generator(
                generator=_gen_samples,
                args=(fpath,),
                output_signature=sig,
                ),
            cycle_length=num_procs,
            num_parallel_calls=num_procs,
            block_length=block_size,
            deterministic=deterministic,
            )
    return dataset

def gen_timegrid_series(
        timegrid_paths, dynamic_feats, static_feats,
        init_epoch:float, final_epoch:float, frequency:int,
        steps_per_batch:int, m_valid=None, include_residual=True,
        derived_feats:dict={}, buf_size_mb=128, max_delta_hours=2, **kwargs):
    """
    Extracts a time series for a set of pixels given contiguous timegrid-style
    hdf5s, time bounds and sequence length, and optionally a user-provided
    valid pixel mask.

    yields gridded samples in chronological order as a 2-tuple like:

    (D, S, T, IX)

    D  : (N, P, F_d)    Dynamic array with N timesteps for P pixels, F_d feats
    S  : (P, F_s)       Static array with P pixels having F_s feats
    T  : (N,)           N epoch float values over the current time range
    IX : (P, 2)         2d integer pixel indeces of each of the P points

    :@param timegrid_paths: List of paths constituting temporally contiguous
        timegrid files uniformly covering the same domain
    :@param dynamic_feats: list of dynamic features to extract, in order.
    :@param static_feats: list of static features to extract, in order.
    :@param init_epoch: First inclusive timestep to extract.
    :@param final_epoch: Exclusive timestep after the last one extracted.
    :@param frequency: Number of timesteps between the initial times of batches
    :@param steps_per_batch: Number of timesteps to include per batch returned.
        If include_residual, this number may be greater than the final batch.
    :@param include_residual: If True, every timestep in [initial, final) will
        be returned even if that means the final batch has fewer elements.
    :@param buf_size_mb: hdf5 buffer size for each timegrid file in MB
    :@param max_delta_hours: Maximum amount of time in hours that samples may
        be separated before an error is raised. This is used for making sure
        consecutive timegrid files are close enough to be sampled across.
    """
    timegrid_paths = list(map(Path, timegrid_paths))
    assert all(p.exists() for p in timegrid_paths)
    times = []
    static_dicts = []
    dynamic_dicts = []
    yslice,xslice = None,None
    ## Parse info dicts and timestamps from each file
    for p in timegrid_paths:
        assert p.exists(), p
        with h5py.File(p, mode="r") as tmpf:
            tmp_time = tmpf["/data/time"][...]
            tmp_static = json.loads(tmpf["/data"].attrs["static"])
            tmp_dynamic = json.loads(tmpf["/data"].attrs["dynamic"])
            dshape = tmpf["/data/dynamic"].shape
            if m_valid is None:
                m_valid = np.full(dshape[1:3], True)
            else:
                ## m_valid will be sliced to match the maximum bounds of yslice
                ## and xslice, so only check size on first pass
                if yslice is None:
                    assert m_valid.shape == dshape[1:3]
            if yslice is None:
                ix = np.stack(np.where(m_valid), axis=1)
                ymin,xmin = np.amin(ix, axis=0)
                ymax,xmax = np.amax(ix, axis=0)
                yslice = slice(ymin, ymax+1)
                xslice = slice(xmin, xmax+1)
                m_valid = m_valid[yslice,xslice]
        times.append(tmp_time)
        static_dicts.append(tmp_static)
        dynamic_dicts.append(tmp_dynamic)

    ## All of the timegrids' feature ordering must be uniform
    dynamic_labels = tuple(dynamic_dicts[0]["flabels"])
    static_labels = tuple(static_dicts[0]["flabels"])
    assert all(tuple(d["flabels"])==dynamic_labels for d in dynamic_dicts[1:])
    assert all(tuple(d["flabels"])==static_labels for d in static_dicts[1:])
    if static_feats is None:
        static_feats = static_labels
    if dynamic_feats is None:
        dynamic_feats = dynamic_labels

    ## Determine the index ordering of requested features in the timegrids
    ## and get derived feature information
    fidx,derived,_ = _parse_feat_idxs(
        out_feats=dynamic_feats,
        src_feats=dynamic_labels,
        static_feats=static_labels,
        derived_feats=derived_feats,
        )

    sidxs = tuple(static_labels.index(f) for f in static_feats)

    ## Collect each path in order with its valid range, sorted by init time
    time_ranges,times = list(zip(*sorted(
            [((p,t[0],t[-1]),t) for t,p in zip(times,timegrid_paths)],
            key=lambda tr:tr[0][1],
            )))
    conc_times = np.concatenate(times, axis=0)

    ## Make sure provided files are appropriately chronological
    file_diffs = list(zip(time_ranges[:-1], time_ranges[1:]))
    for p0,pf,dt in [(p0,pf,(i-f)) for (p0,_,f),(pf,i,_) in file_diffs]:
        if dt<0:
            raise ValueError(
                    "timegrid files must be ordered chronologically;",
                    f"currently {p0} followed by {pf}")
        if dt>max_delta_hours*60*60:
            raise ValueError(
                    "timegrid files must be adjacent in time; currently",
                    f"{pf} starts {dt} seconds the last time in {p0}")

    ## Only include files with time ranges intersecting the requested bounds.
    init_idx = np.argmin(np.abs(conc_times-init_epoch))
    final_idx = np.argmin(np.abs(conc_times-final_epoch))
    print(conc_times[-1], final_epoch, final_idx, conc_times.shape)
    if init_idx < 0:
        raise ValueError(
                "Timegrids must include data before the initial provided time",
                datetime.fromtimestamp(int(conc_times[0])),
                " for the first window."
                )
    if final_idx >= conc_times.size:
        raise ValueError(
                "Timegrids must include data after the final provided time",
                datetime.fromtimestamp(int(conc_times[-1])),
                " for the last horizon."
                )
    init_epoch = conc_times[init_idx]
    final_epoch = conc_times[final_idx]

    ## Reassign the times arrays to only include valid files
    valid_files,times = zip(*[
            (p,t) for (p,t0,tf),t in zip(time_ranges,times)
            if not tf < init_epoch and not t0 >= final_epoch
            ])
    conc_times = np.concatenate(times, axis=0)
    init_idx = np.argmin(np.abs(conc_times - init_epoch))
    final_idx = np.argmin(np.abs(conc_times - final_epoch))
    print(final_idx, conc_times.shape)

    ## Determine the index boundaries of each sample in the domain of times
    ## only including files overlapping the requested period
    total_size = (final_idx-init_idx) + 1
    if total_size <= frequency:
        num_samples = 1
    else:
        num_samples = total_size // int(frequency)
    ## includes partial steps
    init_step_idxs = np.arange(num_samples) * int(frequency) + init_idx
    ## unlike final_idx, this array is non-inclusive
    final_step_idxs = np.clip(init_step_idxs + steps_per_batch, 0, final_idx+1)
    if not include_residual:
        m_res = (final_step_idxs-init_step_idxs) < steps_per_batch
        init_step_idxs = init_step_idxs[~m_res]
        final_step_idxs = final_step_idxs[~m_res]

    ## Get a list of files and their index bounds in terms of full time array
    idx_accum = 0
    files_idx_bounds = []
    for f,t in zip(valid_files, times):
        files_idx_bounds.append((f,idx_accum,idx_accum+t.size))
        idx_accum += t.size

    ## Make a list of slices and corresponding files for each of the samples;
    ## some samples may span multiple files.
    grid_slices = []
    cur_file_idx = 0
    for idx0,idxf in zip(*map(list,(init_step_idxs,final_step_idxs))):
        cur_slices = []
        for f,fidx0,fidxf in files_idx_bounds:
            ## Start index within this file's index range
            if fidx0<=idx0<fidxf:
                ## end index also within this file's range
                if idxf<=fidxf:
                    cur_slices.append((f,slice(idx0-fidx0,idxf-fidx0)))
                ## end index beyond this file's range
                else:
                    cur_slices.append((f,slice(idx0-fidx0,fidxf-fidx0)))
                continue
            ## Start and end index ranges surround the entire file range
            elif idx0<=fidx0<idxf and idx0<=fidxf<idxf:
                cur_slices.append((f,slice(0,fidxf-fidx0)))
            ## End index within this file's index range, but not start index
            elif fidx0<idxf<=fidxf:
                cur_slices.append((f,slice(0,idxf-fidx0)))
        grid_slices.append(cur_slices)

    ## Extract subgrids from the timegrid files in chronological order,
    ## according to the sample format, concatenating across files if needed
    open_files = {}
    static_grid = None
    for sample in grid_slices:
        ## Close files that are no longer in use and remove from the dict
        del_keys = []
        for k in open_files.keys():
            if k not in [p for p,_ in sample]:
                open_files[k].close()
                del_keys.append(k)
        for k in del_keys:
            del open_files[k]
        ## Open new files and add them to the dict
        open_files.update({
            tmp_path:h5py.File(
                tmp_path, mode="r",
                rdcc_nbytes=buf_size_mb*1024**2,
                rdcc_nslots=buf_size_mb*15
                )
            for tmp_path,_ in sample
            if tmp_path not in open_files.keys()
            })

        ## Extract the full dynamic grid associated with this sample
        d = np.concatenate(
                [open_files[f]["/data/dynamic"][s,yslice,xslice]
                    for f,s in sample],
                axis=0)[:,m_valid]
        t = np.concatenate(
                [open_files[f]["/data/time"][s] for f,s in sample],
                axis=0)

        if static_grid is None:
            tmp_static = open_files[sample[0][0]]["/data/static"]
            tmp_static = tmp_static[yslice,xslice][m_valid]
            ## extract numeric static values
            s = tmp_static[...,sidxs]

        d = _calc_feat_array(
                src_array=d,
                static_array=tmp_static,
                stored_feat_idxs=fidx,
                derived_data=derived,
                )

        yield (d,s,t,ix)

if __name__=="__main__":
    timegrid_dir = Path("/rstor/mdodson/thesis/timegrids")
    timegrids_val = [
            p.as_posix() for p in timegrid_dir.iterdir()
            if "timegrid_2021" in p.name
            ]
    timegrids_train = [
            p.as_posix() for p in timegrid_dir.iterdir()
            if not "timegrid_2021" in p.name
            ]

    window_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "dswrf", "apcp",
            "soilm-10", "soilm-40", "soilm-100", "soilm-200", "weasd",
            ]
    horizon_feats = [
            "lai", "veg", "tmp", "spfh", "pres", "ugrd", "vgrd",
            "dlwrf", "dswrf", "apcp",
            ]
    pred_feats = [ 'soilm-10', 'soilm-40', 'soilm-100', 'soilm-200', "weasd" ]
    static_feats = [ "pct_sand", "pct_silt", "pct_clay", "elev", "elev_std" ]
    int_feats = [ "int_veg" ]

    """ Performance testing for sample generation from timegrid """

    from time import perf_counter
    max_count = 50_000
    #max_count = 10
    batch_size = 64
    prefetch = 2

    gen_init_settings = {
        "num_procs":3,
        "deterministic":False,
        "block_size":8,
        "buf_size_mb":4096,
        "samples_per_timegrid":1024,
        "max_offset":23,
        "sample_separation":53,
        "include_init_state_in_predictors":True,
        "load_full_grid":True,
        }

    g = timegrid_sequence_dataset(
            timegrid_paths=timegrids_train,
            window_size=24,
            horizon_size=24*14,
            window_feats=window_feats,
            horizon_feats=horizon_feats,
            pred_feats=pred_feats,
            static_feats=static_feats,
            static_int_feats=[("int_veg",14)],
            static_conditions=[
                #("int_veg", "lambda a: np.any(np.stack(" + \
                #    "[a==v for v in (7,8,9,10,11)], axis=-1), axis=-1)"),
                #("pct_silt", "lambda a: a>.2"),
                #("m_conus", "lambda a: a==1."),
                ],
            **gen_init_settings,
            seed=200007221750,
            )

    count = 0
    time_diffs = []
    prev_time = perf_counter()
    for (w,h,s,si,t),p in g.batch(batch_size).prefetch(prefetch):
        dt = perf_counter()-prev_time
        time_diffs.append(dt)
        count += 1
        if count == max_count:
            break
        prev_time = perf_counter()

    print(time_diffs)
    total_time = sum(time_diffs)

    print(f"Generator initialization settings:")
    ppt(gen_init_settings)
    print(f"Total time: {total_time}")
    print(f"Batch count: {len(time_diffs)}")
    print(f"Avg per batch: {total_time/len(time_diffs)}")
    print(f"batch: {batch_size} ; prefetch: {prefetch}")
