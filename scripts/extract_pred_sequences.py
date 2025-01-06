"""
Script using eval_models.sequence_preds_to_hdf5 to run one or more models over
a series of samples from sequence files, and store the results in a new hdf5.

Then results can be analyzed using multiprocessed sequence evaluation methods
(which require a stored prediction hdf5 and do NOT use Evaluator objects to
dynamically evaluate generated data). The pkls from these regional bulk stats
may then be aggregated into full-domain bulk statistics.
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

from testbed import model_methods as mm
from testbed import eval_models
import tracktrain as tt

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
        eval_models.sequence_preds_to_hdf5(
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
    #model_names = ("lstm-17-235",)
    #model_names = ("lstm-16-505",)
    #model_names = ("lstm-19-191", "lstm-20-353")
    #model_names = ("lstm-21-522", "lstm-22-339")
    #model_names = ("lstm-23-217",)
    #model_names = ("lstm-24-401", "lstm-25-624")
    #model_names = ("snow-4-005",)
    #model_names = ("snow-7-069",)
    #model_names = ("lstm-rsm-6-083",)
    model_names = ("lstm-rsm-9-231",)
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
            and pt[-1] in model_names
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
        for i,subdict in enumerate(pool.imap(
                eval_models.mp_eval_joint_hists,kwargs)):
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
        for i,subdict in enumerate(pool.imap(
                eval_models.mp_eval_static_error,args)):
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
        for i,subdict in enumerate(pool.imap(
                eval_models.mp_eval_error_horizons,args)):
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
        for i,subdict in enumerate(pool.imap(
                eval_models.mp_eval_temporal_error,kwargs)):
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
