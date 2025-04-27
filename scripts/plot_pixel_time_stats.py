"""
Script for running plotting methods on pkls associated with gridded Evaluator
objects, which were probably produced by testbed.eval_grids.eval_model_on_grids
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
from pprint import pprint

from testbed import evaluators
from testbed.eval_grids import GridDomain,GridTile
from testbed import plotting

proj_root = Path("/rhome/mdodson/testbed")
fig_dir = proj_root.joinpath("figures/eval_grid_figs")
#fig_dir = proj_root.joinpath("figures/eval_grid_slope-tiles")
eval_pkl_dir = proj_root.joinpath("data/eval_grid_pkls")

## Specify a subset of grid Evaluator pkls to plot based on name fields:
## eval-grid_{domain}_{md.name}_{eval_feat}_{et}_{na|bias|abs-err}.pkl

## Spatiotemporal domains to plot (2nd field of file name)
plot_domains = [
        #"kentucky-flood",
        #"high-sierra",
        #"sandhills",
        #"hurricane-laura",
        #"gtlb-drought-fire",
        #"dakotas-flash-drought",
        #"hurricane-florence",
        #"eerie-mix",
        #"full",
        "2000-2011",
        ]
## substrings of model names to plot (3rd field of file name)
plot_models_contain = [
        #"accfnn",
        #"accrnn",
        #"lstm-rsm",
        #"acclstm-rsm-1",
        "lstm-rsm-9",
        #"accfnn-rsm-8",
        #"accrnn-rsm-2",
        #"accfnn-rsm-5",
        #"lstm-20",
        #"lstm-18",
        #"acclstm-rsm-4",
        ]
## evlauated features to plot (4th field of file name)
plot_eval_feats = [
        "rsm",
        "rsm-10",
        "rsm-40",
        "rsm-100",
        "soilm-10"
        ]
## Evaluator instance types to include (5th field of file name)
plot_eval_type = [
        #"horizon",
        #"temporal",
        #"static-combos",
        #"hist-true-pred",
        #"hist-saturation-error",
        #"hist-state-increment",
        #"hist-humidity-temp",
        #"hist-infiltration",
        #"spatial-stats",
        "pixelwise-time-stats",
        ]
## error types of evaluators to plot (6th field of file name)
plot_error_type = [
        "na",
        "bias",
        "abs-err",
        ]

## Select which 4-panel configurations to plot (from plot_spatial_stats)
plot_spatial_stats = [
        "res-mean",
        "state-mean",
        "res-err-bias-mean",
        "res-err-bias-stdev",
        "res-err-abs-mean",
        "res-err-abs-stdev",
        "state-err-bias-mean",
        "state-err-bias-stdev",
        "state-err-abs-mean",
        "state-err-abs-stdev",
        "temp-spfh-apcp-mean",
        "temp-spfh-apcp-stdev",
        ]

## --------( END BASIC CONFIGURATION )--------

## Specify 4-panel figure configurations of spatial statistics data
common_spatial_plot_spec = {
        "text_size":24,
        "show_ticks":False,
        "cmap":"gnuplot2",
        "figsize":(32,16),
        #"figsize":(18,12),
        "title_fontsize":36,
        "use_pcolormesh":True,
        "cbar_orient":"horizontal",
        "cbar_shrink":1.,
        #"cbar_shrink":.6,
        "cbar_pad":.02,
        #"geo_bounds":[-95,-80,32,42],
        }

season_spatial_plot_info = [
    {
        "feat":("true_state", "rsm-100"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM (100-200cm; 2018-2023)\n{minfo}",
            },
        },
    {
        "feat":("true_state", "rsm-40"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM (10-40cm; 2018-2023)\n{minfo}",
            },
        },
    {
        "feat":("true_state", "rsm-10"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM (0-10cm; 2018-2023)\n{minfo}",
            },
        },
    {
        "feat":("err_state", "rsm-100"),
        "error_type":"bias",
        "plot_spec":{
            "title":"Quarterly Mean State Bias (100-200cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[-6e-2, -6e-2, -6e-2, -6e-2],
            "vmax":[6e-2, 6e-2, 6e-2, 6e-2],
            "cmap":"seismic_r",
            }
        },
    {
        "feat":("err_state", "rsm-40"),
        "error_type":"bias",
        "plot_spec":{
            "title":"Quarterly Mean State Bias (40-100cm; 2018-2023)\n{minfo}",
            "vmin":[-6e-2, -6e-2, -6e-2, -6e-2],
            "vmax":[6e-2, 6e-2, 6e-2, 6e-2],
            "cmap":"seismic_r",
            }
        },
    {
        "feat":("err_state", "rsm-10"),
        "error_type":"bias",
        "plot_spec":{
            "title":"Quarterly Mean State Bias (0-10cm; 2018-2023)\n{minfo}",
            "vmin":[-1.2e-1, -1.2e-1, -1.2e-1, -1.2e-1],
            "vmax":[1.2e-1, 1.2e-1, 1.2e-1, 1.2e-1],
            "cmap":"seismic_r",
            }
        },
    {
            "feat":("err_state", "rsm-100"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean State Error (100-200cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.05, 0.05, 0.05, 0.05],
            "cmap":"gnuplot2"
            }
        },
    {
            "feat":("err_state", "rsm-40"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean State Error (40-100cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.05, 0.05, 0.05, 0.05],
            "cmap":"gnuplot2"
            }
        },
    {
            "feat":("err_state", "rsm-10"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean State Error (0-10cm; 2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.07, 0.07, 0.07, 0.07],
            "cmap":"gnuplot2"
            },
        },
    {
            "feat":("horizon", "spfh"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean Hourly Humidity (2018-2023)\n{minfo}",
            }
        },
    {
            "feat":("horizon", "apcp"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean Hourly Precipitation (2018-2023)\n{minfo}",
            },
        },
    {
            "feat":("horizon", "tmp"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean Temperature (2018-2023)\n{minfo}",
            },
        },
    ]

## subset available pkls according to selection string configuration
eval_pkls = [
        (p,pt) for p,pt in map(
            lambda f:(f,f.stem.split("_")),
            sorted(eval_pkl_dir.iterdir()))
        if pt[0] == "eval-grid"
        and pt[1] in plot_domains
        and any(s in pt[2] for s in plot_models_contain)
        and pt[3] in plot_eval_feats
        and pt[4] in plot_eval_type
        and (len(pt)==5 or pt[5] in plot_error_type)
        and "PARTIAL" not in pt
        ]
## Ignore spatial stats with error types not needed
eval_pkls = list(filter(
        lambda p:p[1][4] != "spatial-stats" or any([
            spatial_plot_info[k]["error_type"] == p[1][5]
            for k in plot_spatial_stats
            ]),
        eval_pkls
        ))

print(f"Found {len(eval_pkls)} matching eval pkls:")
print("\n".join([p[0].name for p in eval_pkls]))

month_group_labels = [
    "December-February", "March-May",
    "June-August", "September-November",
    ]
month_group_ints = [ (12,1,2),(3,4,5),(6,7,8),(9,10,11) ]

sspi = season_spatial_plot_info
for p,pt in filter(lambda p:p[1][4]=="pixelwise-time-stats", eval_pkls):
    print(f"Plotting from {p.name}")
    ev = evaluators.EvalGridAxes().from_pkl(p)
    _,data_source,model,eval_feat,eval_type,error_type = pt

    ## Gotta do this since indeces are concatenated along along axis 1
    ## with EvalGridAxis concatenation. Probably need to just keep a list.
    idx_zero_splits = list(np.where(
        ev.indeces[1:,0]-ev.indeces[:-1,0] < 0
        )[0] + 1)
    idx_zero_splits = [0] + idx_zero_splits + [ev.indeces.shape[0]]
    tile_slices = [slice(start_tile,end_tile) for start_tile,end_tile
            in zip(idx_zero_splits[:-1], idx_zero_splits[1:])]
    tiles_info = list(zip(
        ev.attrs["latlon"], ev.attrs["tiles"], tile_slices))

    mean_times = [
            datetime.fromtimestamp(int(t))
            for t in np.average(ev.time, axis=1)
            ]
    tmp_months = np.array([t.month for t in mean_times])
    group_means = [
            np.average(ev.average[mask], axis=0)
            for mask in [
                np.any(np.stack([
                    tmp_months==m for m in mg
                    ], axis=-1), axis=-1)
                for mg in month_group_ints
                ]
            ]

    for sspix,spkd in enumerate(sspi):
        if spkd["error_type"] != error_type:
            continue
        ix_sp = ev.attrs["flabels"].index(spkd["feat"])
        feats = np.stack([gm[...,ix_sp] for gm in group_means], axis=-1)

        gridded_feats = []
        for ll,tl,slc in tiles_info:
            tmp_tile_shape = (*ll.shape[:2], feats.shape[-1])
            tmp_tile_feats = np.full(tmp_tile_shape, np.nan)
            ix = ev.indeces[slc]
            ## Batch and sequence axes should be size 1 (marginalized)
            tmp_tile_feats[ix[:,0], ix[:,1],:] = feats[slc,0,:]
            gridded_feats.append(tmp_tile_feats)

        ## plot each of the requested spatial plots
        xt = ev.attrs["domain"].mosaic_shape[-1]
        tile_arrays = [ev.attrs["latlon"], gridded_feats]
        for j,ta in enumerate(tile_arrays):
            rows = [ta[i:i + xt] for i in range(0,len(ta),xt)]
            tile_arrays[j] = np.concatenate(
                    [np.concatenate(x, axis=1) for x in rows], axis=0)
        latlon,feats = tile_arrays

        if "title" in spkd["plot_spec"].keys():
            spkd["plot_spec"]["title"] = spkd["plot_spec"]["title"].format(
                    minfo=" ".join([model, data_source]))
        substr = "qtrly-" + "-".join([
            s.replace("_","-") for s in spkd["feat"]])
        fname = "_".join([
            "eval-grid", data_source, model, eval_type, error_type, substr
            ])
        fpath = fig_dir.joinpath(fname+".png")
        print(f"Generating {fpath}")
        plotting.geo_quad_plot(
                data=[feats[...,i] for i in range(feats.shape[-1])],
                flabels=month_group_labels,
                latitude=latlon[...,0],
                longitude=latlon[...,1],
                plot_spec={
                    **common_spatial_plot_spec,
                    **spkd.get("plot_spec", {}),
                    },
                fig_path=fpath,
                )
