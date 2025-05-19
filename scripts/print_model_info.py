
import json
from pathlib import Path

from tracktrain import ModelDir,ModelSet
from testbed.evaluators import EvalEfficiency
from testbed.list_feats import units_names_mapping
from testbed import plotting

if __name__=="__main__":
    proj_root_dir = Path("/rhome/mdodson/testbed")
    model_root_dir = proj_root_dir.joinpath("data/models/new")
    json_dir = proj_root_dir.joinpath("data")
    fig_dir = proj_root_dir.joinpath("figures/performance-partial")

    eval_root_dir = proj_root_dir.joinpath("data/eval_sequence_pkls")
    #eval_root_dir = proj_root_dir.joinpath("data/eval_rr-rmb_pkls")
    entropy = json.load(json_dir.joinpath("model-entropy.json").open("r"))
    #entropy = json.load(json_dir.joinpath(
    #    "model-entropy_rr-rmb.json").open("r"))
    json_path = json_dir.joinpath(f"model-info.json")
    #json_path = json_dir.joinpath(f"model-info_rr-rmb.json")


    ev_effs = [
            (p,p.stem.split("_")[1:]) for p in eval_root_dir.iterdir()
            if (p.stem.split("_")[1] == "test")
            and (p.stem.split("_")[4] == "efficiency")
            ]

    ## Extract efficiency metrics and information into a json
    #'''
    minfo = {}
    for p,pt in ev_effs:
        #print(f"Loading metrics: {p.stem}")
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
                    "ent-total":tmp_edict.get("ent-total", None),
                    "ent-y":tmp_edict.get("ent-y", None),
                    "ent-p":tmp_edict.get("ent-p", None),
                    "ent-mi":tmp_edict.get("mi", None),
                    "ent-loss":tmp_edict.get("info-loss", None),
                    "ent-fi":tmp_edict.get("fi", None),
                    },
                }

    for model in minfo.keys():
        #print(f"Extracting model info: {model}")
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

    json.dump(minfo, json_path.open("w"), indent=2)
    #'''

    ## Print tables of groups of models
    model_groups = [
        ## initial best
        {
            "group_label":"initial-best",
            "group_title":"Best Models Per Category",
            "models":[ "accfnn-rsm-8", "lstm-20", "lstm-rsm-9"],
            },
        ## initial accfnn-rsm runs
        {
            "group_label":"initial-accfnn-rsm",
            "group_title":"Initial Runs of accfnn-rsm",
            "models":[
                "accfnn-rsm-0", "accfnn-rsm-1", "accfnn-rsm-2", "accfnn-rsm-3",
                "accfnn-rsm-4", "accfnn-rsm-5", "accfnn-rsm-6", "accfnn-rsm-7",
                "accfnn-rsm-8", "accfnn-rsm-9",
                ],
            },
        ## initial lstm-soilm runs
        {
            "group_label":"initial-lstm-soilm",
            "group_title":"Initial Runs of lstm (predicting soilm)",
            "models":[
                "lstm-1", "lstm-2", "lstm-3", "lstm-4", "lstm-8", "lstm-9",
                "lstm-10", "lstm-11", "lstm-12", "lstm-13", "lstm-14",
                "lstm-15", "lstm-16", "lstm-20", "lstm-21", "lstm-22",
                "lstm-23", "lstm-24", "lstm-25", "lstm-26", "lstm-27",
                ],
            },
        ## initial lstm-rsm runs
        {
            "group_label":"initial-lstm-rsm",
            "group_title":"Initial Runs of lstm-rsm",
            "models":[
                "lstm-rsm-0","lstm-rsm-2","lstm-rsm-3","lstm-rsm-5",
                "lstm-rsm-6","lstm-rsm-9","lstm-rsm-10","lstm-rsm-11",
                "lstm-rsm-12", "lstm-rsm-19","lstm-rsm-20"
                ],
            },
        ## initial acclstm-rsm models w/o loss func increment norming
        {
            "group_label":"initial-acclstm-rsm",
            "group_title":"Initial Runs of acclstm-rsm",
            "models":[
                "acclstm-rsm-0", "acclstm-rsm-1", "acclstm-rsm-2",
                "acclstm-rsm-3", "acclstm-rsm-4", "acclstm-rsm-5",
                "acclstm-rsm-6", "acclstm-rsm-7", "acclstm-rsm-8",
                "acclstm-rsm-9", "acclstm-rsm-10", "acclstm-rsm-11",
                "acclstm-rsm-12",
                ],
            },
        ## acclstm-rsm-9 variations w/ inc norming in loss function
        {
            "group_label":"variations-acclstm-rsm-9",
            "group_title":"Model Variations on acclstm-rsm-9",
            "models":[
                "acclstm-rsm-9", "acclstm-rsm-4", "acclstm-rsm-14",
                "acclstm-rsm-15", "acclstm-rsm-16", "acclstm-rsm-17",
                "acclstm-rsm-18", "acclstm-rsm-19", "acclstm-rsm-20",
                ],
            },
        ## lstm-rsm-9 variations including increment norm in loss function
        {
            "group_label":"variations-lstm-rsm-9",
            "group_title":"Model Variations on lstm-rsm-9",
            "models":[
                "lstm-rsm-9", "lstm-rsm-21", "lstm-rsm-22", "lstm-rsm-23",
                "lstm-rsm-24", "lstm-rsm-26", "lstm-rsm-27", "lstm-rsm-28",
                "lstm-rsm-29", "lstm-rsm-30", "lstm-rsm-31",
                ],
            },
        ## acclstm-rsm-4 variations w/o norming in loss function
        {
            "group_label":"variations-acclstm-rsm-4",
            "group_title":"Model Variations on acclstm-rsm-4",
            "models":[
                "acclstm-rsm-4", "acclstm-rsm-9", "acclstm-rsm-21",
                "acclstm-rsm-22", "acclstm-rsm-23", "acclstm-rsm-25",
                "acclstm-rsm-26", "acclstm-rsm-27", "acclstm-rsm-28",
                "acclstm-rsm-29", "acclstm-rsm-30", "acclstm-rsm-31",
                "acclstm-rsm-32", "acclstm-rsm-33",
                ],
            },

        ## Feature variations on lstm-rsm-9
        {
            "group_label":"variations-feat-lstm-rsm-9",
            "group_title":"Feature Variations on lstm-rsm-9",
            "models":[
                "lstm-rsm-9",
                "lstm-rsm-34", "lstm-rsm-35", "lstm-rsm-36", "lstm-rsm-37",
                "lstm-rsm-38", "lstm-rsm-39", "lstm-rsm-40", "lstm-rsm-41",
                "lstm-rsm-42", "lstm-rsm-43", "lstm-rsm-44", "lstm-rsm-45",
                ],
            },

        ## residual magnitude bias variations on lstm-rsm-9
        {
            "group_label":"variations-rmb-lstm-rsm-9",
            "group_title":"Loss Magnitude Bias Variations on lstm-rsm-9",
            "models":[
                "lstm-rsm-9",
                "lstm-rsm-51", "lstm-rsm-50", "lstm-rsm-48", "lstm-rsm-49",
                ],
            },
        ## residual ratio variations on lstm-rsm-9
        {
            "group_label":"variations-rr-lstm-rsm-9",
            "group_title":"Loss Increment Ratio Variations on lstm-rsm-9",
            "models":[
                "lstm-rsm-9", "lstm-rsm-53", "lstm-rsm-54", "lstm-rsm-55",
                ],
            },
        ## MSE variation on lstm-rsm-9
        {
            "group_label":"variations-mse-lstm-rsm-9",
            "group_title":"MSE Loss Function Variation on lstm-rsm-9",
            "models":[
                "lstm-rsm-9", "lstm-rsm-56",
                ],
            },
        ## Loss function norming variation on lstm-rsm-9
        {
            "group_label":"variations-lossnorm-lstm-rsm-9",
            "group_title":"Loss Function Norming Variation on lstm-rsm-9",
            "models":[
                "lstm-rsm-9", "lstm-rsm-57",
                ],
            },
        ## Multiple feature negations variation on lstm-rsm-9
        {
            "group_label":"variations-multineg-lstm-rsm-9",
            "group_title":"lstm-rsm-9 Without Pres., LAI, Elevation",
            "models":[
                "lstm-rsm-9", "lstm-rsm-58",
                ],
            },
        ## Fractional cover rather than LAI variation on lstm-rsm-9
        ## and windmag rather than wind components
        {
            "group_label":"variations-fcover-lstm-rsm-9",
            "group_title":"Fractional Cover + Wind Variants on lstm-rsm-9",
            "models":[
                "lstm-rsm-9", "lstm-rsm-59", "lstm-rsm-60",
                ],
            },
        ]

    ## Since efficiency bar plots group multiple models, a label must be
    ## specified that summarizes the grouping
    dataset = "test" ## for consistency with other eval figures
    plot_groups = [
            #"initial-accfnn-rsm",
            #"initial-lstm-soilm",
            #"initial-lstm-rsm",
            #"initial-best",
            #"initial-acclstm-rsm",
            #"variations-acclstm-rsm-9",
            #"variations-lstm-rsm-9",
            #"variations-acclstm-rsm-4",
            "variations-feat-lstm-rsm-9",
            #"variations-rmb-lstm-rsm-9",
            #"variations-rr-lstm-rsm-9",
            #"variations-mse-lstm-rsm-9",
            #"variations-lossnorm-lstm-rsm-9",
            #"variations-multineg-lstm-rsm-9",
            #"variations-fcover-lstm-rsm-9",
            ]
    print_feats = ["rsm-10", "rsm-40", "rsm-100"]
    print_standard_metrics = [("state", "mae"), ("state", "cc")]
    print_ent_metrics = ["info-loss","fi"]

    ## plot entropy efficiency bar graphs
    '''
    ent_metrics = {
            "mi":"Mutual Information (nats)",
            "info-loss":"Uncertainty Contribution (nats)",
            "fi":"Fractional Information (nats/nats)"
            }
    for mg in model_groups:
        if mg["group_label"] not in plot_groups:
            continue
        models = mg["models"]
        group_label = mg["group_label"]
        group_title = mg["group_title"]

        ent_bar_dict = {
                e:{
                    m:{f:entropy[m][f][e] for f in entropy[m].keys()}
                    for m in models if m in entropy.keys()
                    }
                for e in ent_metrics.keys()
                }

        for e in ent_metrics.keys():
            fig_path = fig_dir.joinpath(
                    f"eval_{dataset}_efficiency_{group_label}_{e}_res.png")
            plotting.plot_nested_bars(
                    data_dict=ent_bar_dict[e],
                    labels={k:v[1] for k,v in units_names_mapping.items()},
                    plot_error_bars=False,
                    bar_order=["rsm-10", "rsm-40", "rsm-100"],
                    group_order=[m for m in models
                        if m in ent_bar_dict[e].keys()],
                    plot_spec={
                        "title":f"{group_title} Inc. {ent_metrics[e]}",
                        "ylabel":ent_metrics[e],
                        #"xlabel":"Model Instance",
                        "xlabel":"",
                        "ylim":{
                            "mi":[0,1.5],
                            "info-loss":[0,2.],
                            "fi":[0,.5],
                            }[e],
                        "bar_spacing":.5,
                        "figsize":(24,12),
                        "xtick_rotation":30,
                        "title_fontsize":32,
                        "xtick_fontsize":24,
                        "legend_fontsize":20,
                        "label_fontsize":24,
                        },
                    bar_colors=["xkcd:forest green",
                        "xkcd:bright blue", "xkcd:light brown"],
                    fig_path=fig_path,
                    )
            print(f"Generated {fig_path.name}")
    '''

    ## print table
    '''
    for mg in model_groups:
        if mg["group_label"] not in plot_groups:
            continue
        rows = []
        for model in mg["models"]:
            col_labels = []
            multirows = []
            feat_rows = [[] for f in print_feats]
            if model not in minfo.keys():
                continue
            cfg = minfo[model]["config"]
            col_labels.append("Name")
            multirows.append("\multirow{{{nrows}}}{{6em}}{{{mname}}}".format(
                mname=model,
                nrows=len(print_feats),
                ))
            col_labels.append("Desc")
            multirows.append("\multirow{{{nrows}}}{{16em}}{{{desc}}}".format(
                desc=cfg["notes"],
                nrows=len(print_feats),
                ))
            col_labels.append("Weights")
            multirows.append("\multirow{{{nrows}}}{{4em}}{{{nparams}}}".format(
                nparams=cfg["trainable_params"],
                nrows=len(print_feats),
                ))
            #col_labels.append("IR")
            #multirows.append("\multirow{{{nrows}}}{{4em}}{{{rr}}}".format(
            #    rr=cfg["loss_fn_args"].get("residual_ratio","1"),
            #    nrows=len(print_feats),
            #    ))
            #col_labels.append("MB")
            #multirows.append("\multirow{{{nrows}}}{{4em}}{{{rmb}}}".format(
            #    rmb=cfg["loss_fn_args"].get("residual_magnitude_bias","0"),
            #    nrows=len(print_feats),
            #    ))
            #col_labels.append("IN")
            #multirows.append("\multirow{{{nrows}}}{{4em}}{{{rn}}}".format(
            #    rn=(not cfg["loss_fn_args"].get("residual_norm") is None),
            #    nrows=len(print_feats),
            #    ))

            for sr,m in print_standard_metrics:
                col_labels.append(" ".join((sr,m)))
                for i,f in enumerate(print_feats):
                    feat_rows[i].append(
                            f"{minfo[model]['metrics'][f][sr][m][0]:.3f}")

            for m in print_ent_metrics:
                col_labels.append(m)
                for i,f in enumerate(print_feats):
                    if model not in entropy.keys():
                        feat_rows[i].append("")
                    else:
                        feat_rows[i].append(f"{entropy[model][f][m]:.3f}")
            str_mr = " & ".join(multirows)
            str_fr = [f" {' & '.join(rt)} \\\\" for rt in feat_rows]
            rows.append(str_mr + " &" + (len(multirows)*" &").join(str_fr))
        print()
        print(mg["group_title"])
        print(" & ".join(col_labels) + " \\\\")
        print("\n".join(rows))
    '''

    ## print models sorted by performance
    #'''
    feat_fi_rank = {}
    feat_mae_rank = {}
    for mg in model_groups:
        if mg["group_label"] not in plot_groups:
            continue
        rows = []
        for model in mg["models"]:
            for f in entropy[model].keys():
                if f not in feat_fi_rank.keys():
                    feat_fi_rank[f] = []
                    feat_mae_rank[f] = []
                feat_fi_rank[f].append((model,entropy[model][f]["fi"]))
                mae = (model,minfo[model]["metrics"][f]["state"]["mae"][0])
                feat_mae_rank[f].append(mae)

    print("Fractional Information")
    for f in feat_fi_rank.keys():
        print(f)
        for i,m in enumerate(list(sorted(feat_fi_rank[f], key=lambda t:t[1]))):
            print(i+1,m)
    print("Mean Absolute Error")
    for f in feat_mae_rank.keys():
        print(f)
        for i,m in enumerate(list(sorted(
            feat_mae_rank[f],key=lambda t:t[1]))[::-1]):
            print(i+1,m)
    #'''
