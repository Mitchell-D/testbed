import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import h5py

import generators
from eval_grids import parse_grid_params

def plot_quad_sequence(
        pred_array, fig_path=None, true_array=None, pred_coarseness=1,
        plot_spec={}, show=False):
    """
    Plot a series of true and predicted sequences in a 4-panel plot,
    each panel containing the data from a single feature

    :@param true_array:(N,S,4) shaped array with N sequence samples, each of
        length S and having 4 features corresponding to feat_labels
    :@param pred_array:(N,S,4) shaped array with N sequence samples, each of
        length S and having 4 features corresponding to feat_labels
    """
    if not true_array is None:
        assert true_array.shape==pred_array.shape, \
                (true_array.shape, pred_array.shape)
    ps = {
            "true_linewidth":1, "pred_linewidth":1,
            "true_linestyle":"-", "pred_linestyle":"-", "main_title":"",
            "quad_titles":("", "", "", ""), "figsize":(12,12), "dpi":100,
            "yscale":"linear", "lines_rgb":None, "grid":False,
            }
    ps.update(plot_spec)
    seq_range = np.arange(pred_array.shape[1]) * pred_coarseness
    plt.clf()
    fig,ax = plt.subplots(2, 2)
    num_px = pred_array.shape[0]
    cm = matplotlib.cm.get_cmap("hsv", num_px)

    has_legend = False
    for n in range(4):
        i = n // 2
        j = n % 2
        for px in range(pred_array.shape[0]):
            if not ps.get("lines_rgb") is None:
                color_true = ps["lines_rgb"][px]
                color_pred = ps["lines_rgb"][px]
            else:
                color_true = ps.get("true_color", cm(px))
                color_pred = ps.get("pred_color", cm(px))


            if not true_array is None:
                tmp_ax_true, = ax[i,j].plot(
                        seq_range,
                        true_array[px,:,n],
                        color=color_true,
                        linewidth=ps.get("true_linewidth"),
                        alpha=ps.get("line_opacity"),
                        linestyle=ps.get("true_linestyle", "-")
                        )
            tmp_ax_pred, = ax[i,j].plot(
                    seq_range,
                    pred_array[px,:,n],
                    color=color_pred,
                    linewidth=ps.get("pred_linewidth"),
                    alpha=ps.get("line_opacity"),
                    linestyle=ps.get("pred_linestyle", "-")
                    )

            ## Add a legend if it is requested but hasn't been added yet
            if not has_legend and not ps.get("pred_legend_label") is None:
                if true_array is None:
                    fig_legend = fig.legend(
                            (tmp_ax_pred,),
                            (ps.get("pred_legend_label"),),
                            loc=ps.get("legend_location", "upper left")
                            )
                else:
                    fig_legend = fig.legend(
                            (tmp_ax_pred, tmp_ax_true),
                            (ps.get("pred_legend_label"),
                                ps.get("true_legend_label")),
                            loc=ps.get("legend_location", "upper left"),
                            prop={"size": ps.get("legend_size",12)},
                            )

            ax[i,j].set_title(ps["quad_titles"][n],
                    fontsize=ps.get("quad_title_size",12))
            if plot_spec.get("yrange"):
                ax[i,j].set_ylim(plot_spec.get("yrange"))
            if plot_spec.get("xrange"):
                ax[i,j].set_xlim(plot_spec.get("xrange"))
            ax[i,j].set_xscale(plot_spec.get("xscale", "linear"))
            ax[i,j].set_yscale(plot_spec.get("yscale", "linear"))

    fig.supxlabel(plot_spec.get("xlabel"), fontsize=ps.get("xlabel_size", 16))
    fig.supylabel(plot_spec.get("ylabel"), fontsize=ps.get("ylabel_size", 16))

    if ps.get("grid"):
        plt.grid()
    if ps["main_title"] != "":
        fig.suptitle(ps["main_title"], fontsize=ps.get("main_title_size", 16))
    plt.tight_layout()
    if not fig_path is None:
        print(f"Saving {fig_path.as_posix()}")
        if ps.get("figsize"):
            fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",
                dpi=ps.get("dpi"))
    if show:
        plt.show()
    plt.close()
    return

if __name__=="__main__":
    fig_dir = Path("figures/subgrid_samples")
    subgrid_h5_dir = Path("data/subgrid_samples/")

    plot_feats = ("soilm-10", "soilm-40", "soilm-100", "soilm-200")
    for tmp_path in subgrid_h5_dir.iterdir():
        _,model_config,gen_args = parse_grid_params(tmp_path)
        tmp_h5 = h5py.File(tmp_path, mode="r")
        file_feats = gen_args.get("pred_feats")
        feat_idxs = tuple(file_feats.index(s) for s in plot_feats)
        p_mean,p_stdev = generators.get_dynamic_coeffs(file_feats)
        P = tmp_h5["/data/preds"]
        Y = tmp_h5["/data/truth"]
        S = tmp_h5["/data/static"][...]

        ## Get the soil percent indeces to assign RGB colors
        sfeats = gen_args["static_feats"]
        soil_feats = ("pct_sand", "pct_silt", "pct_clay")
        soil_idxs = tuple(sfeats.index(s) for s in soil_feats)
        soil_mean,soil_stdev = generators.get_static_coeffs(soil_feats)
        soil_rgb = S[...,soil_idxs]

        for sample_idx in range(P.shape[0]):
            pr = P[sample_idx,...,] # / p_stdev
            ys = Y[sample_idx,...]
            ps = ys[:,0,:][:,np.newaxis,:] + np.cumsum(pr, axis=1)
            yr = ys[:,1:]-ys[:,:-1]
            es = ps - ys[:,1:,:]
            er = pr - yr

            plot_quad_sequence(
                    true_array=yr[...,feat_idxs],
                    pred_array=pr[...,feat_idxs],
                    fig_path=fig_dir.joinpath(
                        tmp_path.stem+"_res-value.png"),
                    pred_coarseness=model_config["feats"]["pred_coarseness"],
                    plot_spec={
                        "main_title":"Hourly change in soil moisture",
                        "quad_titles":plot_feats,
                        #"yscale":"symlog",
                        "yscale":"linear",
                        "true_color":"blue",
                        "pred_color":"orange",
                        "xlabel":"Forecast hour" + \
                                " \n "+" ".join(tmp_path.stem.split("_")[1:]),
                        "ylabel":"Soil moisture layer residual kg/(hr m^2)",
                        "line_opacity":.4,
                        "true_linestyle":"-",
                        "pred_linestyle":"-",
                        "true_linewidth":2,
                        "pred_linewidth":2,
                        "main_title_size":18,
                        "legend_location":"lower left",
                        "pred_legend_label":"Predicted State",
                        "true_legend_label":"True State",
                        "figsize":(11,7),
                        },
                    show=False
                    )

            plot_quad_sequence(
                    pred_array=er,
                    fig_path=fig_dir.joinpath(
                        tmp_path.stem+"_res-error.png"),
                    pred_coarseness=model_config["feats"]["pred_coarseness"],
                    plot_spec={
                        "main_title":"Error in soil moisture residual",
                        "quad_titles":plot_feats,
                        #"yscale":"symlog",
                        "yscale":"linear",
                        "xlabel":"Forecast hour" + \
                                " \n "+" ".join(tmp_path.stem.split("_")[1:]),
                        "ylabel":"Error in layer residual (kg/(hr m^2))",
                        "line_opacity":.3,
                        "lines_rgb":soil_rgb,
                        "true_linewidth":2,
                        "pred_linewidth":2,
                        "main_title_size":18,
                        "figsize":(11,7),
                        "legend_location":"lower left",
                        "pred_legend_label":"Color from soil texture RGB",
                        },
                    show=False
                    )

            plot_quad_sequence(
                    true_array=ys[:,1:,:],
                    pred_array=ps,
                    fig_path=fig_dir.joinpath(
                        tmp_path.stem+"_state-value.png"),
                    pred_coarseness=model_config["feats"]["pred_coarseness"],
                    plot_spec={
                        "main_title":"Hourly volumetric soil mosisture",
                        "quad_titles":plot_feats,
                        "true_color":"blue",
                        "pred_color":"orange",
                        "xlabel":"Forecast hour" + \
                                " \n "+" ".join(tmp_path.stem.split("_")[1:]),
                        "ylabel":"Layer moisture content (kg/m^2)",
                        "line_opacity":.4,
                        "true_linestyle":"-",
                        "pred_linestyle":"-",
                        "true_linewidth":2,
                        "pred_linewidth":2,
                        "figsize":(11,7),
                        "main_title_size":18,
                        "legend_location":"lower left",
                        "pred_legend_label":"Predicted State",
                        "true_legend_label":"True State",
                        },
                    show=False
                    )

            plot_quad_sequence(
                    pred_array=es,
                    fig_path=fig_dir.joinpath(
                        tmp_path.stem+"_state-error.png"),
                    pred_coarseness=model_config["feats"]["pred_coarseness"],
                    plot_spec={
                        "main_title":"Error in volumetric soil mosisture",
                        "quad_titles":plot_feats,
                        "xlabel":"Forecast hour" + \
                                " \n "+" ".join(tmp_path.stem.split("_")[1:]),
                        "ylabel":"Error in layer moisture content (kg/m^2)",
                        "line_opacity":.3,
                        "lines_rgb":soil_rgb,
                        "true_linewidth":2,
                        "pred_linewidth":2,
                        "figsize":(11,7),
                        "main_title_size":18,
                        "legend_location":"lower left",
                        "pred_legend_label":"Color from soil texture RGB",
                        },
                    show=False
                    )
        #break
