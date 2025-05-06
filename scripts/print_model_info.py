
import json
from pathlib import Path

from tracktrain import ModelDir,ModelSet
from testbed.evaluators import EvalEfficiency

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/testbed")
    model_root_dir = proj_root_dir.joinpath("data/models/new")
    eval_root_dir = proj_root_dir.joinpath("data/eval_sequence_pkls")
    json_dir = proj_root_dir.joinpath("data")

    ev_effs = [
            (p,p.stem.split("_")[1:]) for p in eval_root_dir.iterdir()
            if (p.stem.split("_")[1] == "test")
            and (p.stem.split("_")[4] == "efficiency")
            ]

    entropy = json.load(json_dir.joinpath("model-entropy.json").open("r"))

    ## Extract efficiency metrics and information into a json
    #'''
    minfo = {}
    for p,pt in ev_effs:
        print(f"Loading metrics: {p.stem}")
        _,model,feat,*_ = pt
        ev = EvalEfficiency().from_pkl(p)
        if model not in minfo.keys():
            minfo[model] = {"metrics":{}, "config":{}}
        if model in entropy.keys():
            tmp_edict = entropy[model][feat]
        else:
            tmp_edict = {}
        minfo[model]["metrics"][feat] = {
                "state":{
                    "cc":(ev.get_mean("s", "cc"),ev.get_var("s", "cc")),
                    "kge":(ev.get_mean("s", "kge"),ev.get_var("s", "kge")),
                    "nse":(ev.get_mean("s", "nse"),ev.get_var("s", "nse")),
                    "nnse":(ev.get_mean("s", "nnse"),ev.get_var("s", "nnse")),
                    "mae":(ev.get_mean("s", "mae"),ev.get_var("s", "mae")),
                    "mse":(ev.get_mean("s", "mse"),ev.get_var("s", "mse")),
                    },
                "res":{
                    "cc":(ev.get_mean("r", "cc"),ev.get_var("r", "cc")),
                    "kge":(ev.get_mean("r", "kge"),ev.get_var("r", "kge")),
                    "nse":(ev.get_mean("r", "nse"),ev.get_var("r", "nse")),
                    "nnse":(ev.get_mean("r", "nnse"),ev.get_var("r", "nnse")),
                    "mae":(ev.get_mean("r", "mae"),ev.get_var("r", "mae")),
                    "mse":(ev.get_mean("r", "mse"),ev.get_var("r", "mse")),
                    "ent_total":tmp_edict.get("ent_total", None),
                    "ent_y":tmp_edict.get("ent_y", None),
                    "ent_p":tmp_edict.get("ent_p", None),
                    "ent_loss":tmp_edict.get("info_loss", None),
                    "ent_fi":tmp_edict.get("fi", None),
                    },
                }

    for model in minfo.keys():
        print(f"Extracting model info: {model}")
        mpath = model_root_dir.joinpath(model)
        md = ModelDir(mpath)
        ## extract the number of parameters from the summary file
        for l in mpath.joinpath(f"{model}_summary.txt").open("r").readlines():
            if "Trainable params: " not in l:
                continue
            tprm = l.replace("Trainable params: ","")
            tprm = int(tprm[:tprm.index("(")])

        minfo[model]["config"] = {
                "pred_coarseness":md.config["feats"]["pred_coarseness"],
                "loss_fn_args":md.config["data"].get("loss_fn_args", {}),
                "model_type":md.config["model_type"],
                "trainable_params":tprm,
                "notes":md.config["notes"],
                }

    json_path = json_dir.joinpath(f"model-info.json")
    json.dump(minfo, json_path.open("w"), indent=2)
    #'''
