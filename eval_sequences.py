""" """
import numpy as np
import warnings
from pathlib import Path
from pprint import pprint

import tracktrain as tt

import model_methods as mm
from eval_models import sequence_preds_to_hdf5,gen_sequence_predictions
from evaluators import EvalHorizon,EvalTemporal,EvalStatic,EvalJointHist
from list_feats import dynamic_coeffs,static_coeffs
from list_feats import derived_feats,hist_bounds

def get_infiltration_ratio_func(precip_lower_bound=.01):
    def _infil_ratio(soilm,precip):
        return np.where(precip>precip_lower_bound,soilm/precip,0)
    return _infil_ratio

def get_evaluator_objects(eval_types:list, model_dir:tt.ModelDir,
        data_source:str, eval_feat:str, pred_feat:str, use_absolute_error:bool,
        hist_resolution=128, coarse_reduce_func="mean"):
    """
    Returns a list of pre-configured Evaluator subclass objects identified by
    unique strings in the eval_types list (see this method code for details ;D)
    This light wrapper function is just a convenience since the configuration
    of Evaluators can be a bit verbose.

    :@param eval_types: list of string identifiers for Evaluator configurations
        which much match one of the keys of the evals dict below
    :@param model_dir: initialized tracktrain.ModelDir object associated with
        a trained model to evaluate. The model is not loaded or executed by
        this method, but its specific configuration is needed to set feature
        indeces for histograms and static combination evaluators.
    :@param data_source: Descriptive string for the source of data used for
        this evaluation, for example "validation", "val-open-shrubland", etc.
        Make sure this is dash-separated to follow the naming convention.
    :@param eval_feat: Predicted feature with the coordinates appearing in the
        evaluated data. For example, if a model outputs soilm coords by default
        but eval_models.gen_sequence_predictions converts those outputs to rsm,
        eval_feat will be "rsm-10" for the 0-10cm layer.
    :@param pred_feat: String name of the desired feature in the coordinates
        produced by the model (ie before any output conversion).
    :@param use_absolute_error: If True, the Evaluator objects that distinguish
        between error and the absolute magnitude of error will use the latter,
        and vice versa.
    :@param hist_resolution: Number of bins in histogram evaluators
    :@param coarse_reduce_func: In histogram evaluators that are executed on
        coarsened-output models, and which use horizon-derived data inputs, a
        string must be used to identify a function for reducing the hourly data
        chunks to the coarsened resolution. See EvalJointHist documentation.
    """
    ## eval_feat is the converted output ; pred_feat is the model output
    md = model_dir

    pred_feat_idx = md.config["feats"]["pred_feats"].index(pred_feat)
    apcp_idx = md.config["feats"]["horizon_feats"].index("apcp")

    ## initialize some evaluator objects to run batch-wise on the generator
    evals = {
            f"horizon":EvalHorizon(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ),
            f"temporal":EvalTemporal(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                use_absolute_error=use_absolute_error,
                ),
            f"static-combos":EvalStatic(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                soil_idxs=[md.config["feats"]["static_feats"].index(l)
                    for l in ("pct_sand", "pct_silt", "pct_clay")],
                use_absolute_error=use_absolute_error,
                ),
            ## validation histogram
            f"hist-true-pred":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    ("true_res", pred_feat_idx),
                    (*hist_bounds[f"res-{eval_feat}"], hist_resolution),
                    ),
                ax2_args=(
                    ("pred_res", pred_feat_idx),
                    (*hist_bounds[f"res-{eval_feat}"], hist_resolution),
                    ),
                ),
            ## residual error wrt saturation level
            f"hist-saturation-error":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    ("err_res", pred_feat_idx),
                    (*hist_bounds[f"err-res-{eval_feat}"], hist_resolution),
                    ),
                ax2_args=(
                    ("true_state", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                use_absolute_error=use_absolute_error,
                ),
            ## infiltration rate in %/mm (if RSM) or ratio (if soilm)
            f"hist-infiltration":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    (("true_res", pred_feat_idx), ("horizon", apcp_idx)),
                    #get_infiltration_ratio_func(),
                    "lambda s,p:np.where(p>.01,s/p,np.nan)",
                    (-.2,8, hist_resolution),
                    ),
                ax2_args=(
                    (("pred_res", pred_feat_idx), ("horizon", apcp_idx)),
                    #get_infiltration_ratio_func(),
                    "lambda s,p:np.where(p>.01,s/p,np.nan)",
                    (-.2,8, hist_resolution),
                    ),
                use_absolute_error=use_absolute_error,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                coarse_reduce_func="max",
                ),
            ## error rates wrt true state / true residual configuration
            "hist-state-increment":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    ("true_state", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ax2_args=(
                    ("true_res", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ## Calculate the mean residual error per bin
                covariate_feature=("err_res", pred_feat_idx),
                use_absolute_error=use_absolute_error,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                ),
            ## error rates wrt humidity/temperature residual configuration
            "hist-humidity-temp":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    ("horizon",
                        md.config["feats"]["horizon_feats"].index("spfh")),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ax2_args=(
                    ("horizon",
                        md.config["feats"]["horizon_feats"].index("tmp")),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ## Calculate the mean residual error per bin
                coarse_reduce_func="mean",
                covariate_feature=("err_res", pred_feat_idx),
                use_absolute_error=use_absolute_error,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                ),
            }
    selected_evals = []
    for et in eval_types:
        assert et in evals.keys(), f"{et} must be one of\n{list(evals.keys())}"
        tmp_name = f"eval_{data_source}_{md.name}_{eval_feat}_{et}"
        tmp_eval = evals[et]
        selected_evals.append((tmp_name,tmp_eval))
    return selected_evals

if __name__=="__main__":
    sequence_h5_dir = Path("data/sequences/")
    model_parent_dir = Path("data/models/new")
    pred_h5_dir = Path("data/predictions")

    pkl_dir = Path("data/performance/partial")
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
    #weights_file = "accfnn-rsm-8_249_0.008.weights.h5"
    #weights_file = "accrnn-rsm-2_536_0.011.weights.h5"
    #weights_file = "acclstm-rsm-2_235_0.003.weights.h5"
    #weights_file = "lstm-rsm-17_248_0.000.weights.h5" ## full-column
    weights_file = "lstm-25_624_3.189.weights.h5"
    #weights_file = None

    #h5_chunk_size = 64
    gen_batch_size = 2048
    max_batches = 128
    #pred_feat = "rsm-10"
    pred_feat = "soilm-10"
    eval_feat = "rsm-10"

    #'''
    #max_batches = 4
    ## Arguments sufficient to initialize a generators.sequence_dataset,
    ## except feature arguments, which are determined from the ModelDir config
    seq_gen_args = {
            #"seed":200007221750,
            "frequency":1,
            "sample_on_frequency":True,
            "num_procs":5,
            "block_size":8,
            "buf_size_mb":128.,
            "deterministic":True,
            "shuffle":True,
            "yield_times":True,
            "dynamic_norm_coeffs":{k:v[2:] for k,v in dynamic_coeffs},
            "static_norm_coeffs":dict(static_coeffs),
            "derived_feats":derived_feats,
            "max_samples_per_file":int(max_batches*gen_batch_size/12) \
                    if not max_batches is None else None,
            "debug":False,
            }

    ## Parse information about the model from the weights file naming scheme
    mname,epoch = Path(Path(weights_file).stem).stem.split("_")[:2]
    model_label = "-".join((mname,epoch))
    model_dir_path = model_parent_dir.joinpath(mname)

    ## List out all available test data sequence hdf5s
    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("se", "sc", "sw", "ne", "nc", "nw"),
            season_strs=("warm", "cold"),
            time_strs=("2018-2021", "2021-2024"),
            )

    ## initialize the ModelDir instance associated with the requested weights
    md = tt.ModelDir(
            model_dir_path,
            custom_model_builders={
                "lstm-s2s":lambda args:mm.get_lstm_s2s(**args),
                "acclstm":lambda args:mm.get_acclstm(**args),
                "accrnn":lambda args:mm.get_accrnn(**args),
                "accfnn":lambda args:mm.get_accfnn(**args),
                },
            )

    ## initialize a sequence prediction generator instance with the ModelDir
    gen = gen_sequence_predictions(
            model_dir=md,
            sequence_generator_args={
                **seq_gen_args,
                **md.config["feats"],
                "sequence_hdf5s":[p for p in seq_h5s],
                },
            weights_file_name=weights_file,
            gen_batch_size=gen_batch_size,
            max_batches=max_batches,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            gen_numpy=True,
            output_conversion="soilm_to_rsm"
            )

    ## initialize some evaluator objects to run batch-wise on the generator
    evals = get_evaluator_objects(
            eval_types=[
                "horizon", "temporal", "static-combos", "hist-true-pred",
                "hist-saturation-error", "hist-infiltration",
                "hist-state-increment", "hist-humidity-temp",
                ],
            model_dir=md,
            data_source="test",
            eval_feat=eval_feat,
            pred_feat=pred_feat,
            use_absolute_error=False,
            hist_resolution=256,
            coarse_reduce_func="max",
            )
    for inputs,true_states,predicted_residuals in gen:
        print(f"New batch; {true_states.shape = }")
        for _,ev in evals:
            ev.add_batch(inputs,true_states,predicted_residuals)
    for name,ev in evals:
        ev.to_pkl(pkl_dir.joinpath(f"{name}.pkl"))
    #'''
