""" """
import numpy as np
from pathlib import Path
from pprint import pprint

import tracktrain as tt

import model_methods as mm
from eval_models import sequence_preds_to_hdf5,gen_sequence_predictions
from evaluators import EvalHorizon,EvalTemporal,EvalStatic,EvalJointHist

def get_infiltration_ratio_func(precip_lower_bound=.01):
    def _infil_ratio(soilm,precip):
        return np.where(precip>precip_lower_bound,soilm/precip,np.nan)
    return _infil_ratio

if __name__=="__main__":
    from list_feats import dynamic_coeffs,static_coeffs
    from list_feats import derived_feats,hist_bounds
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
    weights_file = "lstm-25_624_3.189.weights.h5"
    #weights_file = "lstm-27_577_4.379.weights.h5"
    #weights_file = "snow-4_005_0.532.weights.h5"
    #weights_file = "snow-6_230_0.064.weights.h5"
    #weights_file = "snow-7_069_0.676.weights.h5"
    #weights_file = "lstm-rsm-1_458_0.001.weights.h5"
    #weights_file = "lstm-rsm-6_083_0.013.weights.h5"
    #weights_file = "lstm-rsm-9_231_0.003.weights.h5"
    #weights_file = "accfnn-rsm-8_249_0.008.weights.h5"
    #weights_file = "accrnn-rsm-2_536_0.011.weights.h5"
    #weights_file = None

    #h5_chunk_size = 64
    gen_batch_size = 1028
    max_batches = 512
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
            "max_samples_per_file":int(max_batches*gen_batch_size/12),
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

    ## eval_feat is the converted output ; pred_feat is the model output
    eval_feat = "rsm-10"
    pred_feat = "soilm-10"

    ## initialize some evaluator objects to run batch-wise on the generator
    evals = {
            f"{md.name}_horizon":EvalHorizon(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ),
            f"{md.name}_temporal":EvalTemporal(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                use_absolute_error=False,
                ),
            f"{md.name}_static":EvalStatic(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                soil_idxs=[md.config["feats"]["static_feats"].index(l)
                    for l in ("pct_sand", "pct_silt", "pct_clay")],
                use_absolute_error=False,
                ),
            ## rsm-10 validation histogram
            f"{md.name}_hist-val_{eval_feat}":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    ("true_res",
                        md.config["feats"]["pred_feats"].index(pred_feat)),
                    (*hist_bounds[f"res-{eval_feat}"], 96),
                    ),
                ax2_args=(
                    ("pred_res",
                        md.config["feats"]["pred_feats"].index(pred_feat)),
                    (*hist_bounds[f"res-{eval_feat}"], 96),
                    ),
                ),
            ## rsm-10 residual error wrt saturation level
            f"{md.name}_hist-saturation_{eval_feat}":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    ("true_state",
                        md.config["feats"]["pred_feats"].index(pred_feat)),
                    (*hist_bounds[eval_feat], 96),
                    ),
                ax2_args=(
                    ("err_res",
                        md.config["feats"]["pred_feats"].index(pred_feat)),
                    (-.2,.2, 96), ## TODO: error histogram bounds
                    ),
                use_absolute_error=False,
                ),
            ## infiltration rate in %/mm (if RSM) or ratio (if soilm)
            f"{md.name}_hist-infiltration_{eval_feat}":EvalJointHist(
                attrs={"model_config":md.config, "gen_args":seq_gen_args},
                ax1_args=(
                    (
                        ("true_res",
                            md.config["feats"]["pred_feats"].index(pred_feat)),
                        ("horizon",
                            md.config["feats"]["horizon_feats"].index("apcp")),
                        ),
                    #get_infiltration_ratio_func(precip_lower_bound=.01),
                    "lambda s,p:np.where(p>.01,s/p,np.nan)",
                    (-.2,8, 96),
                    ),
                ax2_args=(
                    (
                        ("pred_res",
                            md.config["feats"]["pred_feats"].index(pred_feat)),
                        ("horizon",
                            md.config["feats"]["horizon_feats"].index("apcp")),
                        ),
                    "lambda s,p:np.where(p>.01,s/p,np.nan)",
                    (-.2,8, 96),
                    ),
                use_absolute_error=False,
                ignore_nan=True,
                pred_coarseness=md.config["feats"]["pred_coarseness"],
                coarse_reduce_func="max",
                ),
            }
    for inputs,true_states,predicted_residuals in gen:
        print(f"New batch; {true_states.shape = }")
        for ev in evals.values():
            ev.add_batch(inputs,true_states,predicted_residuals)

    for name,ev in evals.items():
        ev.to_pkl(pkl_dir.joinpath(f"{name}.pkl"))

    '''
    sequence_preds_to_hdf5(
            model_dir=md,
            sequence_generator_args={
                **seq_gen_args,
                **md.config["feats"],
                "sequence_hdf5s":[p for p in seq_h5s],
                },
            pred_h5_path=Path("tmp.h5"),
            chunk_size=h5_chunk_size,
            gen_batch_size=gen_batch_size,
            weights_file_name=weights_file,
            dynamic_norm_coeffs={k:v[2:] for k,v in dynamic_coeffs},
            static_norm_coeffs=dict(static_coeffs),
            max_batches=max_batches,
            )
    '''
