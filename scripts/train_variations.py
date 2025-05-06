"""
Script for dispatching single tracktrain-based training runs at a time,
each based on the configuration dict below. Each run will create a new
ModelDir directory and populate it with model info, the configuration,
and intermittent models saved duing training.
"""
from pathlib import Path
from train_single import train_single
from copy import deepcopy

import tracktrain as tt

def rec_merge(primary_dict:dict, secondary_dict:dict, _depth=0):
    """
    Deep-merge dictionaries, defaulting to secondary_dict for collisions.

    If one dict key maps to a dict and the corresponding key for the other
    maps to a different object type, the secondary_dict value is assumed.

    :@param secondary_dict: Dict containing the value to default to in case of
        a collision, in other words a subset dictionary of updated values.
    :@param primary_dict: Dict containing values to be replaced or carried.
    :@param _depth: Recursion counter for knowing when to copy.
    """
    if _depth == 0:
        primary_dict = deepcopy(primary_dict)
        secondary_dict = deepcopy(secondary_dict)
    for sk,sv in secondary_dict.items():
        ## If both dicts contain a dict at the key, merge them
        if sk in primary_dict.keys() \
                and isinstance(primary_dict[sk], dict) \
                and isinstance(sv, dict):
            primary_dict[sk] = rec_merge(primary_dict[sk], sv, _depth+1)
        else:
            primary_dict[sk] = sv
    return primary_dict

if __name__=="__main__":
    root_proj = Path("/rhome/mdodson/testbed/")
    model_parent_dir = root_proj.joinpath("data/models/new")

    #base_model = "acclstm-rsm-4_final.weights.h5"
    base_model = "lstm-rsm-9_final.weights.h5"

    use_residual_norm = False

    #'''
    variations = [
            ## model-shape
            #{"model_name":"acclstm-rsm-25",
            #    "model":{
            #        "lstm_layer_units":[64,64,64,64,64],
            #        "input_linear_embed_size":64,
            #        },
            #    "notes":"acclstm-rsm-4 variation; 1 layer deeper",
            #    },
            #{"model_name":"acclstm-rsm-26",
            #    "model":{
            #        "lstm_layer_units":[64,64,64],
            #        "input_linear_embed_size":64,
            #        },
            #    "notes":"acclstm-rsm-4 variation; 1 layer shallower",
            #    },
            #{"model_name":"acclstm-rsm-27",
            #    "model":{
            #        "lstm_layer_units":[128,128,128,128],
            #        "input_linear_embed_size":128,
            #        },
            #    "notes":"acclstm-rsm-4 variation; twice as wide",
            #    },
            #{"model_name":"acclstm-rsm-28",
            #    "model":{
            #        "lstm_layer_units":[32,32,32,32],
            #        "input_linear_embed_size":32,
            #        },
            #    "notes":"acclstm-rsm-4 variation; half as wide",
            #    },

            ## No variation (except perhaps increment norm in loss)
            #{"model_name":"accrnn-rsm-8",
            #    "model":{"propagate_intermediate_hidden_state":False},
            #    "notes":"accrnn-rsm-4 variation; prop output state",
            #    },
            ## learning-rate
            #{"model_name":"accrnn-rsm-9",
            #    "model":{"propagate_intermediate_hidden_state":False},
            #    "train":{"lr_scheduler_args":{"lr_min":1e-4, "lr_max":1e-2}},
            #    "notes":"accrnn-rsm-4 variation; prop output state; LR oom higher",
            #    },
            #{"model_name":"accrnn-rsm-10",
            #    "model":{"propagate_intermediate_hidden_state":False},
            #    "train":{"lr_scheduler_args":{"lr_min":1e-6, "lr_max":1e-4}},
            #    "compile":{"learning_rate":.005},
            #    "notes":"accrnn-rsm-4 variation; prop output state; LR oom lower",
            #    },
            #{"model_name":"accrnn-rsm-11",
            #    "model":{"propagate_intermediate_hidden_state":False},
            #    "train":{"lr_scheduler_args":{"decay":5e-4}},
            #    "notes":"accrnn-rsm-4 variation; prop output state; decay oom slower",
            #    },

            ### residual magnitude bias
            #{"model_name":"acclstm-rsm-29",
            #    "data":{"loss_fn_args":{"residual_magnitude_bias":50}},
            #    "notes":"lstm-rsm-9 variation; high rmb (50)",
            #    },
            #{"model_name":"acclstm-rsm-30",
            #    "data":{"loss_fn_args":{"residual_magnitude_bias":0}},
            #    "notes":"lstm-rsm-9 variation; no rmb",
            #    },
            ## residual ratio
            #{"model_name":"acclstm-rsm-31",
            #    "data":{"loss_fn_args":{"residual_ratio":.995}},
            #    "notes":"lstm-rsm-9 variation; much higher state loss",
            #    },
            #{"model_name":"acclstm-rsm-32",
            #    "data":{"loss_fn_args":{"residual_ratio":1.}},
            #    "notes":"lstm-rsm-9 variation; no state loss",
            #    },
            ## Loss function
            #{"model_name":"acclstm-rsm-33",
            #    "data":{"loss_fn_args":{"use_mse":True}},
            #    "notes":"lstm-rsm-4 variation; using mse loss not mae",
            #    },

            ## Soil texture
            #{"model_name":"lstm-rsm-46",
            #    "data":{"static_conditions":[
            #        (("pct_sand",), "lambda s:s[0]>.55"),
            #        ]},
            #    "notes":"lstm-rsm-9 variation; sand-dominant soils only",
            #    },
            #{"model_name":"lstm-rsm-47",
            #    "data":{"static_conditions":[
            #        (("pct_clay",), "lambda s:s[0]>.3"),
            #        ]},
            #    "notes":"lstm-rsm-9 variation; clay-dominant soils only",
            #    },
            #{"model_name":"lstm-rsm-52",
            #    "data":{"static_conditions":[
            #        (("pct_silt",), "lambda s:s[0]>.35"),
            #        ]},
            #    "notes":"lstm-rsm-9 variation; silt-dominant soils only",
            #    },

            ## residual magnitude bias
            #{"model_name":"lstm-rsm-48",
            #    "data":{"loss_fn_args":{"residual_magnitude_bias":100}},
            #    "notes":"lstm-rsm-9 variation; very high rmb (100)",
            #    },
            #{"model_name":"lstm-rsm-49",
            #    "data":{"loss_fn_args":{"residual_magnitude_bias":500}},
            #    "notes":"lstm-rsm-9 variation; very very high rmb (500)",
            #    },
            #{"model_name":"lstm-rsm-50",
            #    "data":{"loss_fn_args":{"residual_magnitude_bias":50}},
            #    "notes":"lstm-rsm-9 variation; no residual norm ; " + \
            #        "slightly more rmb (50)",
            #    },
            #{"model_name":"lstm-rsm-51",
            #    "data":{"loss_fn_args":{"residual_magnitude_bias":0}},
            #    "notes":"lstm-rsm-9 variation; no residual norm ; zero rmb",
            #    },

            #{"model_name":"lstm-rsm-53",
            #    "data":{"loss_fn_args":{"residual_ratio":.9995}},
            #    "notes":"lstm-rsm-9 variation; some state loss",
            #    },
            #{"model_name":"lstm-rsm-54",
            #    "data":{"loss_fn_args":{"residual_ratio":.95}},
            #    "notes":"lstm-rsm-9 variation; much more state loss",
            #    },
            {"model_name":"lstm-rsm-55",
                "data":{"loss_fn_args":{"residual_ratio":.5}},
                "notes":"lstm-rsm-9 variation; half state loss",
                },
            ]

    mname,epoch = Path(Path(base_model).stem).stem.split("_")[:2]
    model_dir_path = model_parent_dir.joinpath(mname)
    base_config = tt.ModelDir(model_dir_path).config

    ## Apply each variation to the base config dict and re-train the model.
    for variant in variations:
        new_config = rec_merge(base_config, variant)
        print(f"\nTraining with variant:\n{variant}")
        try:
            best_model = train_single(
                    config=new_config,
                    sequences_dir=Path("/rstor/mdodson/thesis/sequences"),
                    model_parent_dir=model_parent_dir,
                    use_residual_norm=use_residual_norm,
                    )
        except Exception as e:
            print(e)
            continue
    exit(0)
    #'''

    ## Specify a list of lists of feats to exclude from training
    feat_negations = [
            #({"window_feats":["lai"], "horizon_feats":["lai"]},
            #    34, "lstm-rsm-9 without lai"), # v
            #({"window_feats":["veg"], "horizon_feats":["veg"]},
            #    35, "lstm-rsm-9 without veg"),
            #({"window_feats":["tmp"], "horizon_feats":["tmp"]},
            #    36, "lstm-rsm-9 without tmp"), # v
            #({"window_feats":["spfh"], "horizon_feats":["spfh"]},
            #    37, "lstm-rsm-9 without spfh"),
            #({"window_feats":["pres"], "horizon_feats":["pres"]},
            #    38, "lstm-rsm-9 without pres"), # v
            #({"window_feats":["windmag"], "horizon_feats":["windmag"]},
            #    39, "lstm-rsm-9 without windmag"),
            #({"window_feats":["dlwrf"], "horizon_feats":["dlwrf"]},
            #    40, "lstm-rsm-9 without dlwrf"), # v
            #({"window_feats":["dswrf"], "horizon_feats":["dswrf"]},
            #    41, "lstm-rsm-9 without dswrf"),
            ({"window_feats":["apcp"], "horizon_feats":["apcp"]},
                42, "lstm-rsm-9 without apcp"),
            ({"window_feats":["weasd"], "horizon_feats":["weasd"]},
                43, "lstm-rsm-9 without weasd"),
            #({"static_feats":["pct_sand", "pct_silt", "pct_clay"]},
            #    44, "lstm-rsm-9 without static soil texture"), # v
            #({"static_feats":["elev", "elev_std"]},
            #    45, "lstm-rsm-9 without static elevation"), # v
            #({"static_int_feats":["int_veg"],"total_static_int_input_size":0},
            #    46, "lstm-rsm-9 without static vegetation"),
            ]

    base_name = "lstm-rsm-{model_number}"
    mname,epoch = Path(Path(base_model).stem).stem.split("_")[:2]
    model_dir_path = model_parent_dir.joinpath(mname)
    base_config = tt.ModelDir(model_dir_path).config
    for neg,mn,note in feat_negations:
        config_update = {
                "model_name":base_name.format(model_number=mn),
                "feats":{
                    fkey:[
                        f for f in base_config["feats"][fkey]
                        if f not in flist
                        ] for fkey,flist in neg.items()
                    },
                "notes":note,
                }
        new_config = rec_merge(base_config, config_update)
        try:
            best_model = train_single(
                    config=new_config,
                    sequences_dir=Path("/rstor/mdodson/thesis/sequences"),
                    model_parent_dir=model_parent_dir,
                    use_residual_norm=use_residual_norm,
                    )
        except Exception as e:
            print(e)
            continue
