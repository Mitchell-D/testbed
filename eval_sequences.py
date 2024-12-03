""" """
import numpy as np
from pathlib import Path

import tracktrain as tt

import model_methods as mm
from eval_models import sequence_preds_to_hdf5

if __name__=="__main__":
    from list_feats import dynamic_coeffs,static_coeffs,derived_feats
    sequence_h5_dir = Path("data/sequences/")
    model_parent_dir = Path("data/models/new")
    pred_h5_dir = Path("data/predictions")

    error_horizons_pkl = Path(f"data/performance/error_horizons.pkl")
    temporal_pkl = Path(f"data/performance/temporal_absolute.pkl")
    hists_pkl = Path(f"data/performance/validation_hists_7d.pkl")
    static_error_pkl = Path(f"data/performance/static_error.pkl")

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
    weights_file = "lstm-rsm-9_231_0.003.weights.h5"
    #weights_file = None

    h5_chunk_size = 128
    gen_batch_size = 128
    max_batches = 64
    ## Arguments sufficient to initialize a generators.sequence_dataset,
    ## except feature arguments, which are determined from the ModelDir config
    seq_gen_args = {
            "seed":200007221750,
            "frequency":1,
            "sample_on_frequency":True,
            "num_procs":5,
            "block_size":16,
            "buf_size_mb":128.,
            "deterministic":True,
            "shuffle":True,
            "yield_times":True,
            "dynamic_norm_coeffs":{k:v[2:] for k,v in dynamic_coeffs},
            "static_norm_coeffs":dict(static_coeffs),
            "derived_feats":derived_feats,
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
