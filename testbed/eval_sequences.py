""" """
import gc
import numpy as np
import warnings
from pathlib import Path
from pprint import pprint

import tracktrain as tt

from testbed import model_methods as mm
from testbed.eval_models import gen_sequence_predictions
from testbed.evaluators import EvalHorizon,EvalTemporal,EvalEfficiency
from testbed.evaluators import EvalStatic,EvalJointHist
from testbed.list_feats import dynamic_coeffs,static_coeffs
from testbed.list_feats import derived_feats,hist_bounds

def get_infiltration_ratio_func(precip_lower_bound=.01):
    def _infil_ratio(soilm,precip):
        return np.where(precip>precip_lower_bound,soilm/precip,0)
    return _infil_ratio

def get_sequence_evaluator_objects(eval_types:list, model_dir:tt.ModelDir,
        data_source:str, eval_feat:str, pred_feat:str, use_absolute_error:bool,
        sequence_generator_args={}, hist_resolution=128,
        coarse_reduce_func="mean"):
    """
    Returns a list of pre-configured sequence Evaluator subclass objects
    identified by unique strings in the eval_types list (see this method code
    for details ;D) This light wrapper function is just a convenience since
    the configuration of Evaluators can be a bit verbose.

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
    if "apcp" in md.config["feats"]["horizon_feats"]:
        apcp_idx = md.config["feats"]["horizon_feats"].index("apcp")
    else:
        if "hist-infiltration" in eval_types:
            raise ValueError(f"Precipitation is not a horizon feature!")
        apcp_idx = None
    if "spfh" in md.config["feats"]["horizon_feats"]:
        spfh_idx = md.config["feats"]["horizon_feats"].index("spfh")
    else:
        if "hist-humidity-temp" in eval_types:
            raise ValueError(f"humidity is not a horizon feature!")
        spfh_idx = None
    if "tmp" in md.config["feats"]["horizon_feats"]:
        temp_idx = md.config["feats"]["horizon_feats"].index("tmp")
    else:
        if "hist-humidity-temp" in eval_types:
            raise ValueError(f"Temperature is not a horizon feature!")
        temp_idx = None

    ## list the evaluator labels for which it matters whether error bias vs
    ## absolute error value is distinguished in the output file name
    absolute_error_relevant = [
            "temporal", "static-combos", "hist-humidity-temp",
            "hist-state-increment",
            ]
    ## Evaluator instances that consider all feats simultaneously, so the
    ## eval_feat field in the file name should be general
    contains_all_feats = ["horizon", "temporal", "static-combos"]
    ## initialize some evaluator objects to run batch-wise on the generator
    evals = {
            f"horizon":EvalHorizon(
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        "xlabel":"Forecast distance (hours)",
                        }
                    },

                ),
            f"temporal":EvalTemporal(
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        }
                    },
                use_absolute_error=use_absolute_error,
                ),
            f"static-combos":EvalStatic(
                attrs={"model_config":md.config,
                    "gen_args":sequence_generator_args},
                soil_idxs=[
                    md.config["feats"]["static_feats"].index(l)
                    if "static-combos" in eval_types else None
                    for l in ("pct_sand", "pct_silt", "pct_clay")
                    ],
                use_absolute_error=use_absolute_error,
                ),
            f"efficiency":EvalEfficiency(
                pred_feat_idx=pred_feat_idx,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                attrs={"model_config":md.config,
                    "gen_args":sequence_generator_args},
                ),
            ## validation histogram
            f"hist-true-pred":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        "title":"Validation Histogram " + \
                                f"{eval_feat} ({md.name})",
                        "ylabel":"True Increment Change",
                        "xlabel":"Predicted Increment Change",
                        }
                    },
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
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        "title":"Joint distribution of increment error in" + \
                                f" {eval_feat} wrt state",
                        "xlabel":"Hourly increment error in ({eval_feat})",
                        "ylabel":"True state magnitude for ({eval_feat})",
                        }
                    },
                ax1_args=(
                    ("true_state", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ax2_args=(
                    ("err_res", pred_feat_idx),
                    (*hist_bounds[f"err-res-{eval_feat}"], hist_resolution),
                    ),
                use_absolute_error=use_absolute_error,
                ),
            ## infiltration rate in %/mm (if RSM) or ratio (if soilm)
            f"hist-infiltration":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        "title":"Validation curve of 10cm infiltration " + \
                                "ratio with ",
                        "ylabel":"True ratio of rainfall to 10cm infiltration",
                        "xlabel":"Predicted ratio of rainfall to 10cm " + \
                                "infiltration",
                        }
                    },
                ax1_args=(
                    (("true_res", pred_feat_idx), ("horizon", apcp_idx)),
                    #get_infiltration_ratio_func(),
                    "lambda s,p:np.where(p>.01,s/p,np.nan)",
                    (-.4,1.5, hist_resolution),
                    ),
                ax2_args=(
                    (("pred_res", pred_feat_idx), ("horizon", apcp_idx)),
                    #get_infiltration_ratio_func(),
                    "lambda s,p:np.where(p>.01,s/p,np.nan)",
                    (-.4,1.5, hist_resolution),
                    ),
                ignore_nan=True,
                covariate_feature=("true_state", pred_feat_idx),
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                coarse_reduce_func="max",
                ),
            ## error rates wrt true state / true residual configuration
            "hist-state-increment":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        "title":"Joint distribution of true state and true" + \
                                "increment with MAE contours",
                        "ylabel":"True state ({eval_feat})",
                        "xlabel":"True increment change ({eval_feat}) ",
                        },
                    },
                ax1_args=(
                    ("true_state", pred_feat_idx),
                    (*hist_bounds[eval_feat], hist_resolution),
                    ),
                ax2_args=(
                    ("true_res", pred_feat_idx),
                    (*hist_bounds["res-"+eval_feat], hist_resolution),
                    ),
                ## Calculate the mean residual error per bin
                covariate_feature=("err_res", pred_feat_idx),
                use_absolute_error=use_absolute_error,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                ),
            ## error rates wrt humidity/temperature residual configuration
            "hist-humidity-temp":EvalJointHist(
                attrs={
                    "model_config":md.config,
                    "gen_args":sequence_generator_args,
                    "plot_spec":{
                        "title":"Joint distribution of humidity and temp" + \
                                "with MAE contours",
                        "ylabel":"Specific humidity (kg/kg)",
                        "xlabel":"Temperature (K)",
                        }
                    },
                ax1_args=(
                    ("horizon", spfh_idx),
                    (*hist_bounds["spfh"], hist_resolution),
                    ),
                ax2_args=(
                    ("horizon", temp_idx),
                    (*hist_bounds["tmp"], hist_resolution),
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
        tmp_name = list(map(str, ("eval",data_source,md.name,eval_feat,et)))
        if et in absolute_error_relevant:
            tmp_name.append(["bias", "abs-err"][use_absolute_error])
        else:
            tmp_name.append("na")
        if et in contains_all_feats:
            tmp_name[3] = eval_feat.split("-")[0]
        tmp_eval = evals[et]
        selected_evals.append(("_".join(tmp_name),tmp_eval))
    return selected_evals

def eval_model_on_sequences(pkl_dir:Path, model_dir_path:Path,
        weights_file:str, eval_getter_args:list, sequence_gen_args:dict,
        sequence_hdf5s, gen_batch_size=256, max_batches=None,
        output_conversion="soilm_to_rsm", reset_model_each_batch=False,
        dynamic_norm_coeffs={}, static_norm_coeffs={},):
    """
    High-level method that executes a model over a sequence dataset using
    eval_models.gen_sequence_predictions, and runs a series of Evaluator
    subclass objects on the results batch-wise.

    :@param pkl_dir: Directory where Evaluator pkl files are generated
    :@param model_dir_path: Path to the ModelDir-created directory of the model
        to be evaluated
    :@param weights_file: File name (only) of the ".weights.hdf5 " model file
        to execute, which is anticipated to be stored in the above model dir.
    :@param eval_getter_args: a list of dictionary keyword arguments to
        get_sequence_evaluator_objects excluding only the model_dir argument.
        Each entry may list multiple Evaluator objects to evaluate for a
        particular feature, absolute error/bias, reduction function, or
        histogram resolution
    :@param sequence_gen_args: dict of arguments to gen_sequence_predictions
        specifying how to declare the data generator. Exclude the "*_feats"
        and "sequence_hdf5s" arguments which are provided based on the ModelDir
        configuration and argument to this method, respectively.
    :@param sequence_hdf5s: list of sequence hdf5s to interleave in producing
        the generated data
    :@param gen_batch_size: Number of samples drawn per batch/evaluation.
    :@param max_batches: Optional maximum number of batches to be evaluated
    :@param output_conversion: Specify which conversion function to run within
        the generator if the provided model produces the opposite unit type.
        Must be either "soilm_to_rsm" or "rsm_to_soilm".
    :@param reset_model_each_batch: Some large custom models seem to overflow
        session memory for some reason when evaluated on many large batches.
        This option will reset the tensorflow session state and reload the
        model weights for each batch if set to True.
    """
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
                **sequence_gen_args,
                **md.config["feats"],
                "sequence_hdf5s":list(sequence_hdf5s),
                },
            weights_file_name=weights_file,
            gen_batch_size=gen_batch_size,
            max_batches=max_batches,
            dynamic_norm_coeffs=dynamic_norm_coeffs,
            static_norm_coeffs=static_norm_coeffs,
            gen_numpy=True,
            output_conversion=output_conversion,
            reset_model_each_batch=reset_model_each_batch,
            )

    ## initialize some evaluator objects to run batch-wise on the generator
    evals = []
    for eargs in eval_getter_args:
        evals += get_sequence_evaluator_objects(
                model_dir=md,
                sequence_generator_args=sequence_gen_args,
                **eargs
                )

    ## run each of the evaluators on every batch from the generator
    for inputs,true_states,predicted_residuals in gen:
        print(f"{md.name} new batch; {true_states.shape = }")
        for _,ev in evals:
            ev.add_batch(inputs,true_states,predicted_residuals)
    out_paths = []
    for name,ev in evals:
        out_paths.append(pkl_dir.joinpath(f"{name}.pkl"))
        ev.to_pkl(out_paths[-1])
    return out_paths

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/testbed/")
    sequence_h5_dir = proj_root_dir.joinpath("data/sequences/")
    model_parent_dir = proj_root_dir.joinpath("data/models/new")
    pred_h5_dir = proj_root_dir.joinpath("data/predictions")
    #pkl_dir = proj_root_dir.joinpath("data/eval_sequence_pkls")
    pkl_dir = proj_root_dir.joinpath("data/eval_rr-rmb_pkls")
    #pkl_dir = proj_root_dir.joinpath("data/eval_seq-test_pkls")

    ## only models that predict rsm at 3 depth levels (tf 2.14)
    rsm_models = [
        ## Fully-connected models (feedforward only)
        "accfnn-rsm-0_final.weights.h5", "accfnn-rsm-1_final.weights.h5",
        "accfnn-rsm-2_final.weights.h5", "accfnn-rsm-3_final.weights.h5",
        "accfnn-rsm-4_final.weights.h5", "accfnn-rsm-5_final.weights.h5",
        "accfnn-rsm-6_final.weights.h5", "accfnn-rsm-7_final.weights.h5",
        "accfnn-rsm-8_final.weights.h5", "accfnn-rsm-9_final.weights.h5",

        ## State-accumulating LSTMs
        "acclstm-rsm-0_final.weights.h5",
        "acclstm-rsm-1_056_0.003.weights.h5",
        "acclstm-rsm-2_final.weights.h5", "acclstm-rsm-3_final.weights.h5",
        "acclstm-rsm-4_final.weights.h5", "acclstm-rsm-5_final.weights.h5",
        "acclstm-rsm-6_final.weights.h5", "acclstm-rsm-7_final.weights.h5",
        "acclstm-rsm-8_final.weights.h5", "acclstm-rsm-9_final.weights.h5",
        "acclstm-rsm-10_final.weights.h5", "acclstm-rsm-11_final.weights.h5",
        "acclstm-rsm-12_final.weights.h5",

        ## Accumulating RNNs
        "accrnn-rsm-0_final.weights.h5", "accrnn-rsm-1_final.weights.h5",
        "accrnn-rsm-2_final.weights.h5", "accrnn-rsm-3_final.weights.h5",
        "accrnn-rsm-4_final.weights.h5", "accrnn-rsm-5_final.weights.h5",
        "accrnn-rsm-6_final.weights.h5",
        ## RNN variations (w/o intermediate weight propagation)
        "accrnn-rsm-9_final.weights.h5", "accrnn-rsm-11_final.weights.h5",

        ## Basic LSTMs
        "lstm-rsm-0_final.weights.h5", "lstm-rsm-2_final.weights.h5",
        "lstm-rsm-3_final.weights.h5", "lstm-rsm-5_final.weights.h5",
        "lstm-rsm-6_final.weights.h5", "lstm-rsm-7_021_0.015.weights.h5",
        "lstm-rsm-8_final.weights.h5", "lstm-rsm-9_final.weights.h5",
        "lstm-rsm-10_final.weights.h5", "lstm-rsm-11_final.weights.h5",
        "lstm-rsm-12_final.weights.h5", "lstm-rsm-19_final.weights.h5",
        "lstm-rsm-20_final.weights.h5",

        ## acclstm-rsm-9 shape variations
        "acclstm-rsm-14_final.weights.h5", "acclstm-rsm-15_final.weights.h5",
        "acclstm-rsm-16_final.weights.h5",

        ## acclstm-rsm-9 learning rate variations
        "acclstm-rsm-17_final.weights.h5", "acclstm-rsm-18_final.weights.h5",
        "acclstm-rsm-19_final.weights.h5",
        ]

    ## Basic LSTMs predicting 4-layer soilm + snow (tf 2.15)
    soilm_models = [
        "lstm-1_final.weights.h5", "lstm-2_final.weights.h5",
        "lstm-3_final.weights.h5", "lstm-4_final.weights.h5",
        "lstm-5_final.weights.h5", "lstm-6_final.weights.h5",
        "lstm-7_final.weights.h5", "lstm-8_final.weights.h5",
        "lstm-9_final.weights.h5", "lstm-10_final.weights.h5",
        "lstm-11_final.weights.h5", "lstm-12_final.weights.h5",
        "lstm-13_final.weights.h5", "lstm-14_final.weights.h5",
        "lstm-15_final.weights.h5", "lstm-16_final.weights.h5",
        "lstm-17_final.weights.h5", "lstm-18_final.weights.h5",
        "lstm-19_final.weights.h5", "lstm-20_final.weights.h5",
        "lstm-21_final.weights.h5", "lstm-22_final.weights.h5",
        "lstm-23_final.weights.h5", "lstm-24_final.weights.h5",
        "lstm-25_final.weights.h5", "lstm-26_final.weights.h5",
        "lstm-27_final.weights.h5",
        ]

    ## size of each batch drawn.
    #gen_batch_size = 256
    #gen_batch_size = 128 ## for rr,rmb eval
    gen_batch_size = 2048 ## for feature variation eval
    ## Maximum number of batches to draw for evaluation
    #max_batches = 32
    #max_batches = 1024 ## for rr,rmb eval
    max_batches = 64 ## for feature variation eval
    ## Model predicted unit. Used to identify feature indeces in truth/pred
    pred_feat_unit = "rsm"
    ## Output unit. Determines which set of evaluators are executed
    eval_feat_unit = "rsm"
    ## Subset of model weights to evaluate
    #weights_to_eval = soilm_models

    #weights_to_eval = [m for m in rsm_models if m[:12]=="accfnn-rsm-8"]
    #weights_to_eval = [m for m in soilm_models if m[:7]=="lstm-20"]
    #weights_to_eval = [m for m in rsm_models if m[:10]=="lstm-rsm-9"]

    #weights_to_eval = [m for m in rsm_models if m[:12]=="accrnn-rsm-2"]
    #weights_to_eval = [m for m in rsm_models if m[:12]=="accfnn-rsm-5"]
    #weights_to_eval = [m for m in rsm_models if m[:13] in
    #        [f"acclstm-rsm-{j}"] for j in range(13,20)]
    #weights_to_eval = [m for m in rsm_models if m[:9]=="lstm-rsm-"]
    #weights_to_eval = [m for m in rsm_models if m[:9]=="lstm-rsm-"]
    #weights_to_eval = [m for m in rsm_models if "accfnn" in m]
    #weights_to_eval = ["acclstm-rsm-4_final.weights.h5"]

    ## initial soilm model w area density; no loss function norming
    #weights_to_eval = soilm_models[0:7]
    #weights_to_eval = soilm_models[7:14]
    #weights_to_eval = soilm_models[14:21]
    #weights_to_eval = soilm_models[21:28]

    ## acclstm-rsm-9 variations (with norming in loss function)
    '''
    weights_to_eval = [
        #"acclstm-rsm-9_final.weights.h5", # done
        #"acclstm-rsm-14_final.weights.h5", # done
        #"acclstm-rsm-15_final.weights.h5", # done
        #"acclstm-rsm-16_final.weights.h5", # done
        #"acclstm-rsm-17_final.weights.h5", # done
        #"acclstm-rsm-18_final.weights.h5", # done
        #"acclstm-rsm-19_final.weights.h5", # done
        #"acclstm-rsm-20_final.weights.h5", # done
        ]
    '''

    ## initial acclstm-rsm runs
    '''
    weights_to_eval = [
        "acclstm-rsm-0_final.weights.h5",
        "acclstm-rsm-1_056_0.003.weights.h5",
        "acclstm-rsm-2_final.weights.h5",
        "acclstm-rsm-3_final.weights.h5",
        "acclstm-rsm-4_final.weights.h5",
        "acclstm-rsm-5_final.weights.h5",
        "acclstm-rsm-6_final.weights.h5",
        "acclstm-rsm-7_final.weights.h5",
        "acclstm-rsm-8_final.weights.h5",
        "acclstm-rsm-9_final.weights.h5",
        "acclstm-rsm-10_final.weights.h5",
        "acclstm-rsm-11_final.weights.h5",
        "acclstm-rsm-12_final.weights.h5",
        ]
    '''

    ## initial lstm-rsm runs (without norming in loss function)
    '''
    weights_to_eval = [
            "lstm-rsm-0_final.weights.h5", "lstm-rsm-2_final.weights.h5",
            "lstm-rsm-3_final.weights.h5", "lstm-rsm-5_final.weights.h5",
            "lstm-rsm-6_final.weights.h5", "lstm-rsm-7_final.weights.h5",
            "lstm-rsm-8_final.weights.h5",

            "lstm-rsm-9_final.weights.h5",
            "lstm-rsm-10_final.weights.h5", "lstm-rsm-11_final.weights.h5",
            "lstm-rsm-12_final.weights.h5", "lstm-rsm-19_final.weights.h5",
            "lstm-rsm-20_final.weights.h5",
            ]
    '''
    ## lstm-rsm-9 variations (with norming in loss function)
    '''
    weights_to_eval = [
            #"lstm-rsm-21_final.weights.h5", "lstm-rsm-22_final.weights.h5",
            #"lstm-rsm-23_final.weights.h5", "lstm-rsm-24_final.weights.h5",
            #"lstm-rsm-26_final.weights.h5",

            #"lstm-rsm-27_final.weights.h5",
            "lstm-rsm-28_final.weights.h5", "lstm-rsm-29_final.weights.h5",
            #"lstm-rsm-30_final.weights.h5", "lstm-rsm-31_final.weights.h5",

            #"lstm-rsm-48_final.weights.h5", "lstm-rsm-49_final.weights.h5",
            ]
    '''

    ## lstm-rsm-4 variations (no norming in loss function)
    '''
    weights_to_eval = [
        #"acclstm-rsm-21_final.weights.h5",
        #"acclstm-rsm-22_final.weights.h5",
        #"acclstm-rsm-23_final.weights.h5",
        #"acclstm-rsm-25_final.weights.h5",
        #"acclstm-rsm-26_final.weights.h5",
        #"acclstm-rsm-27_final.weights.h5",
        #"acclstm-rsm-28_final.weights.h5",
        #"acclstm-rsm-29_final.weights.h5",
        #"acclstm-rsm-30_final.weights.h5",
        #"acclstm-rsm-31_final.weights.h5",
        #"acclstm-rsm-32_final.weights.h5",
        #"acclstm-rsm-33_final.weights.h5",
        ]
    '''

    ## Initial best models
    '''
    weights_to_eval = [
            #"lstm-rsm-9_final.weights.h5", "accfnn-rsm-8_final.weights.h5",
            #"accfnn-rsm-5_final.weights.h5",
            #"acclstm-rsm-4_final.weights.h5",
            #"lstm-20_final.weights.h5",
            ]
    '''

    ## feature variations on acclstm-rsm-9
    '''
    weights_to_eval = [
        "lstm-rsm-34_final.weights.h5", "lstm-rsm-35_final.weights.h5", # v
        "lstm-rsm-36_final.weights.h5", "lstm-rsm-37_final.weights.h5", # v
        "lstm-rsm-38_final.weights.h5", "lstm-rsm-39_final.weights.h5", # v
        "lstm-rsm-40_final.weights.h5", "lstm-rsm-41_final.weights.h5", # v
        "lstm-rsm-42_final.weights.h5", "lstm-rsm-43_final.weights.h5", # v
        "lstm-rsm-44_final.weights.h5", # v
        "lstm-rsm-45_final.weights.h5", # v
        ]
    '''

    ## loss function variations on lstm-rsm-9
    #'''
    weights_to_eval = [
        #"lstm-rsm-9_final.weights.h5",

        ## 9:RMB=10, 50:RMB=50, 51:RMB=0, 48:RMB=100, 49:RMB=500
        #"lstm-rsm-48_final.weights.h5", "lstm-rsm-49_final.weights.h5",
        #"lstm-rsm-50_final.weights.h5", "lstm-rsm-51_final.weights.h5",

        ## 9:RR=1, 53:RR=.9995, 54:RR=.95, 55:RR=.5
        #"lstm-rsm-53_final.weights.h5", "lstm-rsm-54_final.weights.h5",
        #"lstm-rsm-55_final.weights.h5",

        ## MSE rather than MAE loss
        #"lstm-rsm-56_final.weights.h5",

        ## sand,silt,clay domainant trained models
        #"lstm-rsm-46_final.weights.h5",
        #"lstm-rsm-52_final.weights.h5",
        #"lstm-rsm-47_final.weights.h5",
        #"lstm-rsm-9_final.weights.h5",

        ## lstm-rsm-9 trained with loss norming
        #"lstm-rsm-57_final.weights.h5",

        ## lstm-rsm-9 trained without lai, pres, elevation
        #"lstm-rsm-58_final.weights.h5",

        ## retraining model with wind negated
        #"lstm-rsm-39_final.weights.h5",

        ## fractional cover rather than LAI
        #"lstm-rsm-59_final.weights.h5",

        ## wind magnitude rather than components
        "lstm-rsm-60_final.weights.h5",
        ]
    #'''


    print(f"{weights_to_eval = }")

    #'''
    ## Arguments sufficient to initialize a generators.sequence_dataset,
    ## except feature arguments, which are determined from the ModelDir config
    f_freeze = "np.any((a[0]>270)&(a[0]<276)&(a[1]>.003)&(a[1]<.006),axis=1)"
    f_hot = "np.any((a[0]>300)&(a[0]<310)&(a[1]>.02)&(a[1]<.03),axis=1)"
    f_wetrain = "np.any((a[0]>.85)&(a[0]<.95)&" + \
            "(np.diff(a[0],axis=1)>.075)&(np.diff(a[0],axis=1)<.25),axis=1)"
    seq_gen_args = {
            "seed":200007221750, ## standard seed
            #"seed":102934659156243850, ## for rr,rmb evaluation
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
            "debug":True,

            "horizon_conditions":[
                #(("tmp","spfh"), f"lambda a:{f_freeze}"),
                #(("tmp","spfh"), f"lambda a:{f_hot}"),
                ],
            "pred_conditions":[
                #(("rsm-10",), f"lambda a:{f_wetrain}"),
                ],
            "static_conditions":[
                #(("pct_sand",), "lambda s:s[0]>.55"),
                #(("pct_silt",), "lambda s:s[0]>.35"),
                #(("pct_clay",), "lambda s:s[0]>.3"),
                ],
            }

    ## list of dicts encoding arguments to get_sequence_evaluator_objects,
    ## which will be applied to all the provided models. Exclude the
    ## model_dir:ModelDir parameter, which is specified as an argument to
    ## eval_model_on_sequences
    rsm_evaluator_getter_args = [
            ## First-layer evaluators, error bias
            {
            "eval_types":[
                "horizon",
                "temporal",
                "static-combos",
                "hist-true-pred",
                "hist-saturation-error",
                "hist-state-increment",
                "hist-humidity-temp",
                "efficiency",
                ],
            "data_source":"test",
            "eval_feat":"rsm-10",
            "pred_feat":f"{pred_feat_unit}-10",
            "use_absolute_error":False,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ## Second-layer evaluators, error bias
            {
            "eval_types":[
                "hist-true-pred",
                "hist-saturation-error",
                "hist-state-increment",
                "efficiency",
                ],
            "data_source":"test",
            "eval_feat":"rsm-40",
            "pred_feat":f"{pred_feat_unit}-40",
            "use_absolute_error":False,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ## Third-layer evaluators, error bias
            {
            "eval_types":[
                "hist-true-pred",
                "hist-saturation-error",
                "hist-state-increment",
                "efficiency",
                ],
            "data_source":"test",
            "eval_feat":"rsm-100",
            "pred_feat":f"{pred_feat_unit}-100",
            "use_absolute_error":False,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ## First-layer evaluators, error magnitude
            {
            "eval_types":[
                "temporal",
                "static-combos",
                "hist-state-increment",
                "hist-humidity-temp",
                ],
            "data_source":"test",
            "eval_feat":"rsm-10",
            "pred_feat":f"{pred_feat_unit}-10",
            "use_absolute_error":True,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ## Second-layer evaluators, error magnitude
            {
            "eval_types":[
                "hist-state-increment",
                ],
            "data_source":"test",
            "eval_feat":"rsm-40",
            "pred_feat":f"{pred_feat_unit}-40",
            "use_absolute_error":True,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ## Third-layer evaluators, error magnitude
            {
            "eval_types":[
                "hist-state-increment",
                ],
            "data_source":"test",
            "eval_feat":"rsm-100",
            "pred_feat":f"{pred_feat_unit}-100",
            "use_absolute_error":True,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            },
            ]
    ## The only evaluator that is reasonable to keep in soil moisture area
    ## density coordinates is the infiltration ratio, since rain values and
    ## soil moisture increment change are both in kg/m^2/hr
    soilm_evaluator_getter_args = [
            {
            "eval_types":["hist-infiltration"],
            "data_source":"test",
            "eval_feat":"soilm-10",
            "pred_feat":f"{pred_feat_unit}-10",
            "use_absolute_error":True,
            "hist_resolution":512,
            "coarse_reduce_func":"max",
            }
            ]

    ## List out all available test data sequence hdf5s
    seq_h5s = mm.get_seq_paths(
            sequence_h5_dir=sequence_h5_dir,
            region_strs=("se", "sc", "sw", "ne", "nc", "nw"),
            season_strs=("warm", "cold"),
            time_strs=("2018-2021", "2021-2024"),
            )

    for weights_file in weights_to_eval:
        print(f"Evaluating {weights_file}")
        ## Parse information about the model from the weights file name scheme
        mname,epoch = Path(Path(weights_file).stem).stem.split("_")[:2]
        #model_label = "-".join((mname,epoch))
        model_dir_path = model_parent_dir.joinpath(mname)

        out_pkls = eval_model_on_sequences(
                model_dir_path=model_dir_path,
                pkl_dir=pkl_dir,
                weights_file=weights_file,
                eval_getter_args={
                    "soilm":soilm_evaluator_getter_args,
                    "rsm":rsm_evaluator_getter_args,
                    }[eval_feat_unit],
                sequence_gen_args=seq_gen_args,
                sequence_hdf5s=seq_h5s,
                gen_batch_size=gen_batch_size,
                max_batches=max_batches,
                output_conversion={
                    "soilm":"rsm_to_soilm",
                    "rsm":"soilm_to_rsm",
                    }[eval_feat_unit],
                dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
                static_norm_coeffs=dict(static_coeffs),
                reset_model_each_batch=True,
                )
        #'''
        print(f"Generated evaluator pkls:")
        pprint(out_pkls)
        gc.collect()
