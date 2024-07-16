import pickle as pkl
import numpy as np
import os
import sys
import h5py
import json
import random as rand
from list_feats import dynamic_coeffs,static_coeffs
from pathlib import Path
from random import random
from pprint import pprint as ppt

import tensorflow as tf

def get_dynamic_coeffs(fields=None):
    """
    Load the averages and standard devia of dynamic features from the
    configured list, returning them as a 2-tuple (means, stdevs)
    """
    if fields:
        dc = dict(dynamic_coeffs)
        dc = [dc[k] for k in fields]
    else:
        _,dc = zip(*dynamic_coeffs)
    dc = np.vstack(dc).T
    return (dc[0],dc[1])

def get_static_coeffs(fields=None):
    """
    Load the averages and standard devia of static features from the
    configured list, returning them as a 2-tuple (means, stdevs)
    """
    if fields:
        sc = dict(static_coeffs)
        sc = [sc[k] for k in fields]
    else:
        _,sc = zip(*static_coeffs)
    sc = np.vstack(sc).T
    return (sc[0],sc[1])

def gen_timegrid_samples(
        timegrid_paths, window_size, horizon_size,
        window_feats, horizon_feats, pred_feats,
        static_feats, static_int_feats, static_conditions=[],
        num_procs=1, deterministic=False, block_size=16, buf_size_mb=128,
        samples_per_timegrid=256, max_offset=0, sample_separation=1,
        include_init_state_in_predictors=False, load_full_grid=False,
        seed=None):
    """
    Versatile generator for providing data samples consisting of a window,
    a horizon, a static vector, and a label (truth) vector using a sample
    hdf5 file with a superset of features.

    Feature types:
    window : feats used to initialize the model prior to the first prediction
    horizon : covariate feats used to inform predictions
    pred : feats to be predicted by the model
    static : time-invariant feats used to parameterize dynamics
    static_int : 2-TUPLE like (feature_name, embed_size) of time-invariant
        integer features to be one-hot encoded in embed_size vector.

    :@param *_size: Number of timesteps from the pivot in the window or horizon
    :@param *_feats: String feature labels in order of appearence
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

        window_idxs = tuple(ddict["flabels"].index(l) for l in window_feats)
        horizon_idxs = tuple(ddict["flabels"].index(l) for l in horizon_feats)
        pred_idxs = tuple(ddict["flabels"].index(l) for l in pred_feats)
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

            ## extract and separate dynamic features
            seq_length = window_size+horizon_size
            tmp_dynamic = D[tmp_sidx:tmp_sidx+seq_length,
                    tmp_gidx[0],tmp_gidx[1],:]
            tmp_window = tmp_dynamic[:window_size, window_idxs]
            tmp_horizon = tmp_dynamic[-horizon_size:, horizon_idxs]
            tmp_pred = tmp_dynamic[-pred_size:, pred_idxs]

            ## collect the time segment
            tmp_time = T[tmp_sidx:tmp_sidx+seq_length]

            ## collect static features
            tmp_static = S[*tmp_gidx, static_idxs]
            tmp_static_int = static_oh[*tmp_gidx]
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

def make_sequence_hdf5(
        seq_h5_path, timegrid_paths, window_size, horizon_size,
        window_feats, horizon_feats, pred_feats,
        static_feats, static_int_feats, static_conditions=[],
        num_procs=1, deterministic=False, block_size=16, buf_size_mb=128,
        samples_per_timegrid=256, max_offset=0, sample_separation=1,
        include_init_state_in_predictors=False, load_full_grid=False,
        seed=None, prefetch_count=1, batch_size=64, max_batches=None,
        samples_per_chunk=64, debug=False):
    """
    Create a new hdf5 file of training-ready samples extracted using a
    dataset generator from gen_timegrid_samples

    --(old parameters )--

    The majority of parameters are used to initialize a gen_timegrid_samples
    dataset generator, and are identical to those documented above

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
        P = f.create_dataset(
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
        gen = gen_timegrid_samples(**timegrid_dataset_params)

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
            P.resize((h5idx, *P.shape[1:]))
            S.resize((h5idx, *S.shape[1:]))
            SI.resize((h5idx, *SI.shape[1:]))
            T.resize((h5idx, *T.shape[1:]))

            ## Load the batch into the new file
            W[sample_slice,...] = w.numpy()
            H[sample_slice,...] = h.numpy()
            P[sample_slice,...] = p.numpy()
            S[sample_slice,...] = s.numpy()
            SI[sample_slice,...] = si.numpy()
            T[sample_slice,...] = t.numpy()
            f.flush()

            batch_counter += 1
            if batch_counter == max_batches:
                break
        f.close()

def gen_sequence_samples(sequence_hdf5s:list, window_feats, horizon_feats,
        pred_feats, static_feats, static_int_feats, seed=None, frequency=1,
        sample_on_frequency=True, num_procs=1, block_size=64,
        buf_size_mb=128., deterministic=False, yield_times:bool=False,
        dynamic_norm_coeffs:dict={}, static_norm_coeffs:dict={}, **kwargs):
        #use_residual_pred_coeffs:bool=False,  **kwargs):
    """
    get a tensorflow dataset that generates samples from sequence hdf5s,
    which must have been created by make_sequence_hdf5

    Note: Unlike make_sequence_hdf5 and gen_timegrid_samples, static feature
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

    --( shuffling and frequency-splitting )--

    :@param seed: Optional random seed determining the shuffle order of chunks
        returned from the sequence hdf5s
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

    :@param use_residual_pred_coeffs: If True, the norm coefficients used for
        output values are selected from the dynamic_norm_coeffs with the
        feature key prepended with "res_". For example, "soilm-10" will use
        the dynamic coefficients associated with "res_soilm-10".
    """
    ## Make a pass over all the files to make sure the data is valid
    assert len(sequence_hdf5s), "There must be at least one sequence hdf5"
    window_size = None
    for tmp_path in sequence_hdf5s:
        #with h5py.File(tmp_path, "r") as tmp_file:

        tmp_file = h5py.File(tmp_path, "r")
        #print(type(tmp_file["data"].attrs))
        #print(type(tmp_file["data"].attrs["gen_params"]))
        #print(tmp_file["data"].attrs["gen_params"])
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
            for feat_type,feat_labels in output_feats.items():
                if not all(l in tmp_params[feat_type] for l in feat_labels):
                    raise ValueError(
                            f"Requested {feat_type} args are not a subset " + \
                            f"of those in {tmp_path.as_posix()}:\n",
                            tmp_params[feat_type])

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
    pred_size = horizon_size + int(include_init_state_in_predictors)
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
        w_fidx = tuple(
                tmp_params["window_feats"].index(l)
                for l in window_feats)
        h_fidx = tuple(
                tmp_params["horizon_feats"].index(l)
                for l in horizon_feats)
        p_fidx = tuple(
                tmp_params["pred_feats"].index(l)
                for l in pred_feats)
        s_fidx = tuple(
                tmp_params["static_feats"].index(l)
                for l in static_feats)

        ## Get slices along the batch axis identifying each chunk and shuffle
        ## an index array to homogenize the data.
        batch_chunk_slices = sorted([
            s[0] for s in F["/data/window"].iter_chunks()
            ])
        chunk_idxs = np.arange(len(batch_chunk_slices))
        rng.shuffle(chunk_idxs)
        print(seq_h5.name, chunk_idxs)

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

        ## Extract valid chunks one at a time.
        for i in range(chunk_idxs.size):
            tmp_slice = batch_chunk_slices[chunk_idxs[i]]
            cidxs = np.arange(tmp_slice.stop-tmp_slice.start)
            rng.shuffle(cidxs)
            tmp_window = F["/data/window"][tmp_slice,...][...,w_fidx][cidxs]
            tmp_horizon = F["/data/horizon"][tmp_slice,...][...,h_fidx][cidxs]
            tmp_pred = F["/data/pred"][tmp_slice,...][...,p_fidx][cidxs]
            tmp_static = F["/data/static"][tmp_slice,...][...,s_fidx][cidxs]
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

    g = gen_timegrid_samples(
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
