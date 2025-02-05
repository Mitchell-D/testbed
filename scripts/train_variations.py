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

    base_model = "acclstm-rsm-9_final.weights.h5"
    variation_label = "shape-variations"

    variations = [
            ## model-shape
            #{"model_name":"acclstm-rsm-13",
            #    "model":{"lstm_layer_units":[64,64,64,64,64]},
            #    "notes":"acclstm-rsm-9 variation; 1 layer deeper",
            #    },
            #{"model_name":"acclstm-rsm-14",
            #    "model":{"lstm_layer_units":[64,64,64]},
            #    "notes":"acclstm-rsm-9 variation; 1 layer shallower",
            #    },
            #{"model_name":"acclstm-rsm-15",
            #    "model":{
            #        "lstm_layer_units":[128,128,128,128],
            #        "ann_in_units":128,
            #        },
            #    "notes":"acclstm-rsm-9 variation; twice as wide",
            #    },
            #{"model_name":"acclstm-rsm-16",
            #    "model":{
            #        "lstm_layer_units":[32,32,32,32],
            #        "ann_in_units":32,
            #        },
            #    "notes":"acclstm-rsm-9 variation; half as wide",
            #    },

            ## learning-rate
            {"model_name":"acclstm-rsm-17",
                "train":{"lr_scheduler_args":{"lr_min":1e-4, "lr_max":5e-3}},
                "notes":"acclstm-rsm-9 variation; LR oom higher",
                },
            {"model_name":"acclstm-rsm-18",
                "train":{"lr_scheduler_args":{"lr_min":1e-6, "lr_max":5e-5}},
                "compile":{"learning_rate":.005},
                "notes":"acclstm-rsm-9 variation; LR oom lower",
                },
            {"model_name":"acclstm-rsm-19",
                "train":{"lr_scheduler_args":{"decay":1e-4}},
                "notes":"acclstm-rsm-9 variation; decay oom slower",
                },

            ## Feature exclusion
            #"acclstm-rsm-20":{
            #    "feats":{
            #        "window_feats":["lai", "veg", "tmp", "spfh", "pres",
            #            "windmag", "dlwrf", "dswrf", "apcp",
            #            "rsm-10", "rsm-40", "rsm-100", ],
            #        "horizon_feats":["lai", "veg", "tmp", "spfh", "pres",
            #            "windmag", "dlwrf", "dswrf", "apcp", "weasd" ],
            #        "static_feats":["pct_sand", "pct_silt", "pct_clay",
            #            "elev", "elev_std"],
            #        },
            #    "acclstm-rsm-9 variation; decay order of magnitude slower",
            #    },
            ]

    mname,epoch = Path(Path(base_model).stem).stem.split("_")[:2]
    model_dir_path = model_parent_dir.joinpath(mname)
    base_config = tt.ModelDir(model_dir_path).config

    for variant in variations:
        new_config = rec_merge(base_config, variant)
        print(f"\nTraining with variant:\n{variant}")
        try:
            best_model = train_single(
                    config=new_config,
                    sequences_dir=Path("/rstor/mdodson/thesis/sequences"),
                    model_parent_dir=model_parent_dir,
                    )
        except Exception as e:
            print(e)
            continue
