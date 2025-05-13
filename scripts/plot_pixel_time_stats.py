"""
Script for running plotting methods on pkls of pixelwise-time-stats Evaluator
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
from testbed.list_feats import statsgo_texture_default,soil_texture_colors


## Collect soil texture arrays and their corresponding text labels
soil_mapping = [
        (np.array(texture), label, color)
        for label,abbrv_label,texture,color in [
            (*statsgo_texture_default[ix],soil_texture_colors[ix])
            for ix in list(range(1,13))+[16]
            ]
        ]

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
        "full",
        #"2000-2011",
        #"lt-north-michigan",
        #"lt-high-plains",
        #"lt-cascades",
        #"lt-fourcorners",
        #"lt-miss-alluvial",
        #"lt-atlanta",
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

        #"lstm-rsm-46", ## sand-dominant model
        ]
## evlauated features to plot (4th field of file name)
plot_eval_feats = [
        "rsm",
        "rsm-10",
        "rsm-40",
        "rsm-100",
        #"soilm",
        #"soilm-10",
        #"soilm-40",
        #"soilm-100",
        #"soilm-200",
        ]
## Evaluator instance types to include (5th field of file name)
plot_eval_type = [
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
        "figsize":(18,10),
        #"figsize":(18,12),
        "title_fontsize":36,
        "use_pcolormesh":True,
        "cbar_orient":"horizontal",
        "cbar_shrink":.9,
        #"cbar_shrink":.6,
        "cbar_pad":.02,
        #"geo_bounds":[-95,-80,32,42],
        }

season_spatial_plot_info = [
    {
        "feat":("true_state", "soilm-100"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean SOILM (kg/m^2; 40-100cm; " + \
                    "2018-2023)\n{minfo}",
            },
        },
    {
        "feat":("true_state", "soilm-40"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean SOILM (kg/m^2; 10-40cm; " + \
                    "2018-2023)\n{minfo}",
            },
        },
    {
        "feat":("true_state", "soilm-10"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean SOILM (kg/m^2; 0-10cm; " + \
                    "2018-2023)\n{minfo}",
            },
        },
    {
        "feat":("true_state", "rsm-100"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM (40-100cm; 2018-2023)\n{minfo}",
            "vmin":[-.2,-.2,-.2,-.2],
            "vmax":[1.,1.,1.,1.],
            },
        },
    {
        "feat":("true_state", "rsm-40"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM (10-40cm; 2018-2023)\n{minfo}",
            "vmin":[-.2,-.2,-.2,-.2],
            "vmax":[1.,1.,1.,1.],
            },
        },
    {
        "feat":("true_state", "rsm-10"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM (0-10cm; 2018-2023)\n{minfo}",
            "vmin":[-.2,-.2,-.2,-.2],
            "vmax":[1.,1.,1.,1.],
            },
        },
    {
        "feat":("err_state", "rsm-100"),
        "error_type":"bias",
        "plot_spec":{
            "title":"Quarterly Mean RSM State Bias (40-100cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[-1.2e-1, -1.2e-1, -1.2e-1, -1.2e-1],
            "vmax":[1.2e-1, 1.2e-1, 1.2e-1, 1.2e-1],
            "cmap":"seismic_r",
            }
        },
    {
        "feat":("err_state", "rsm-40"),
        "error_type":"bias",
        "plot_spec":{
            "title":"Quarterly Mean RSM State Bias " + \
                    "(10-40cm; 2018-2023)\n{minfo}",
            "vmin":[-1.2e-1, -1.2e-1, -1.2e-1, -1.2e-1],
            "vmax":[1.2e-1, 1.2e-1, 1.2e-1, 1.2e-1],
            "cmap":"seismic_r",
            }
        },
    {
        "feat":("err_state", "rsm-10"),
        "error_type":"bias",
        "plot_spec":{
            "title":"Quarterly Mean RSM State Bias" + \
                    " (0-10cm; 2018-2023)\n{minfo}",
            "vmin":[-1.2e-1, -1.2e-1, -1.2e-1, -1.2e-1],
            "vmax":[1.2e-1, 1.2e-1, 1.2e-1, 1.2e-1],
            "cmap":"seismic_r",
            }
        },
    {
        "feat":("err_state", "rsm-100"),
        "error_type":"abs-err",
        "use_mape":True,
        "plot_spec":{
            "title":"Quarterly Mean Percent State Error (40-100cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.2, 0.2, 0.2, 0.2],
            "cmap":"gnuplot2"
            }
        },
    {
        "feat":("err_state", "rsm-40"),
        "error_type":"abs-err",
        "use_mape":True,
        "plot_spec":{
            "title":"Quarterly Mean Percent State Error (10-40cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.2, 0.2, 0.2, 0.2],
            "cmap":"gnuplot2"
            }
        },
    {
        "feat":("err_state", "rsm-10"),
        "error_type":"abs-err",
        "use_mape":True,
        "plot_spec":{
            "title":"Quarterly Mean Percent State Error " + \
                    "(0-10cm; 2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.2, 0.2, 0.2, 0.2],
            "cmap":"gnuplot2"
            },
        },
    {
        "feat":("err_state", "rsm-100"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM State Error (40-100cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.07, 0.07, 0.07, 0.07],
            "cmap":"gnuplot2"
            }
        },
    {
        "feat":("err_state", "rsm-40"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM State Error (10-40cm; " + \
                    "2018-2023)\n{minfo}",
            "vmin":[0, 0, 0, 0],
            "vmax":[0.07, 0.07, 0.07, 0.07],
            "cmap":"gnuplot2"
            }
        },
    {
        "feat":("err_state", "rsm-10"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean RSM State Error " + \
                    "(0-10cm; 2018-2023)\n{minfo}",
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
            "vmin":[0,0,0,0],
            "vmax":[0.35,0.35,0.35,0.35],
            },
        },
    {
        "feat":("horizon", "tmp"),
        "error_type":"abs-err",
        "plot_spec":{
            "title":"Quarterly Mean Temperature (2018-2023)\n{minfo}",
            "vmin":[255, 255, 255, 255],
            "vmax":[305, 305, 305, 305],
            },
        },
    ]

time_series_ps_default = {
        "fig_size":(24,8),
        "dpi":120,
        "legend_ncols":2,
        "legend_font_size":10,
        "label_size":14,
        "title_size":20,
        "xtick_rotation":30,
        "xtick_align":"right",
        "date_format":"%Y-%m-%d",
        "time_locator":"month",
        "time_locator_interval":6,
        "spine_increment":.07,
        "zero_axis":True,
        "grid":True,
        "xrange":(datetime(2018,1,1), datetime(2024,1,1)),
        }
time_series_plot_info = [
    ## monthly absolute error in state binned by elevation
    {
        "name":"monthly-elev-abs-err-state-rsm-10",
        "feats":[ ("err_state", "rsm-10"), ],
        "error_type":"abs-err",
        "agg_type":"elev",
        "bin_monthly":True,
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Error in 0-10cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 0-10cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-elev-abs-err-state-rsm-40",
        "bin_monthly":True,
        "feats":[ ("err_state", "rsm-40"), ],
        "error_type":"abs-err",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Error in 10-40cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 10-40cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-elev-abs-err-state-rsm-100",
        "bin_monthly":True,
        "feats":[ ("err_state", "rsm-100"), ],
        "error_type":"abs-err",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Error in 40-100cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 40-100cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    ## monthly bias in state binned by elevation
    {
        "name":"monthly-elev-bias-state-rsm-10",
        "feats":[ ("err_state", "rsm-10"), ],
        "error_type":"bias",
        "agg_type":"elev",
        "bin_monthly":True,
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Bias in 0-10cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 0-10cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-elev-bias-state-rsm-40",
        "bin_monthly":True,
        "feats":[ ("err_state", "rsm-40"), ],
        "error_type":"bias",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Bias in 10-40cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 10-40cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-elev-bias-state-rsm-100",
        "bin_monthly":True,
        "feats":[ ("err_state", "rsm-100"), ],
        "error_type":"bias",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Bias in 40-100cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 40-100cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    ## bias in state binned by soil texture
    {
        "name":"monthly-txtr-abs-err-state-rsm-10",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-10"),
            ],
        "error_type":"abs-err",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Error in 0-10cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Absolute Error in 0-10cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-txtr-abs-err-state-rsm-40",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-40"),
            ],
        "error_type":"abs-err",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Error in 10-40cm State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Absolute Error in 10-40cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-txtr-abs-err-state-rsm-100",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-100"),
            ],
        "error_type":"abs-err",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Error in 40-100cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Absolute Error in 40-100cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-txtr-bias-state-rsm-10",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-10"),
            ],
        "error_type":"bias",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Bias in 0-10cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 0-10cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-txtr-bias-state-rsm-40",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-40"),
            ],
        "error_type":"bias",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Bias in 10-40cm State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 10-40cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-txtr-bias-state-rsm-100",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-100"),
            ],
        "error_type":"bias",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Bias in 40-100cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in 40-100cm RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    ## absolute error, bias, and true value of state per level
    {
        "name":"monthly-all-bias-state",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-10"),
            ("err_state", "rsm-40"),
            ("err_state", "rsm-100"),
            ],
        "error_type":"bias",
        "agg_type":"all",
        "plot_spec":{
            "title":"Overall Mean Bias in RSM State (2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Bias in RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-all-abs-err-state",
        "bin_monthly":True,
        "feats":[
            ("err_state", "rsm-10"),
            ("err_state", "rsm-40"),
            ("err_state", "rsm-100"),
            ],
        "error_type":"abs-err",
        "agg_type":"all",
        "plot_spec":{
            "title":"Overall Mean Absolute Error in RSM State " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Absolute Error in RSM",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    {
        "name":"monthly-all-true-state",
        "bin_monthly":True,
        "feats":[
            ("true_state", "rsm-10"),
            ("true_state", "rsm-40"),
            ("true_state", "rsm-100"),
            ],
        "error_type":"abs-err",
        "agg_type":"all",
        "plot_spec":{
            "title":"True RSM State (2018-2024)\n{minfo}",
            "xlabel":"Month",
            "ylabel":"Relative Soil Moisture",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            "line_width":3,
            },
        },
    ## multi y-axis forcings
    {
        "name":"monthly-all-forcings",
        "bin_monthly":True,
        "feats":[
            ("horizon", "tmp"),
            ("horizon", "spfh"),
            ("horizon", "apcp"),
            ("horizon", "weasd"),
            ],
        "error_type":"abs-err",
        "agg_type":"all-multiy",
        "plot_spec":{
            "title":"Monthly average forcings",
            "xlabel":"Month",
            "time_locator_interval":1,
            "xrange":(datetime(2000,1,1), datetime(2000,12,1)),
            "xtick_rotation":0,
            "xtick_align":"center",
            },
        },
    ## bias in state binned by elevation
    {
        "name":"elev-bias-state-rsm-10",
        "bin_monthly":False,
        "feats":[ ("err_state", "rsm-10"), ],
        "error_type":"bias",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Bias in 0-10cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in 0-10cm RSM",
            },
        },
    {
        "name":"elev-bias-state-rsm-40",
        "bin_monthly":False,
        "feats":[ ("err_state", "rsm-40"), ],
        "error_type":"bias",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Bias in 10-40cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in 10-40cm RSM",
            },
        },
    {
        "name":"elev-bias-state-rsm-100",
        "bin_monthly":False,
        "feats":[ ("err_state", "rsm-100"), ],
        "error_type":"bias",
        "agg_type":"elev",
        "elev_bins":[(0,1000),(1000,2000),(2000,2500),
            (2500,3000),(3000,3500),(3500,4000)],
        "plot_spec":{
            "title":"Mean Bias in 40-100cm RSM State wrt Elevation " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in 40-100cm RSM",
            },
        },
    ## bias in state binned by soil texture
    {
        "name":"txtr-abs-err-state-rsm-10",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-10"),
            ],
        "error_type":"abs-err",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Error in 0-10cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Absolute Error in 0-10cm RSM",
            },
        },
    {
        "name":"txtr-abs-err-state-rsm-40",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-40"),
            ],
        "error_type":"abs-err",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Error in 10-40cm State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Absolute Error in 10-40cm RSM",
            },
        },
    {
        "name":"txtr-abs-err-state-rsm-100",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-100"),
            ],
        "error_type":"abs-err",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Error in 40-100cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Absolute Error in 40-100cm RSM",
            },
        },
    {
        "name":"txtr-bias-state-rsm-10",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-10"),
            ],
        "error_type":"bias",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Bias in 0-10cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in 0-10cm RSM",
            },
        },
    {
        "name":"txtr-bias-state-rsm-40",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-40"),
            ],
        "error_type":"bias",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Bias in 10-40cm State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in 10-40cm RSM",
            },
        },
    {
        "name":"txtr-bias-state-rsm-100",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-100"),
            ],
        "error_type":"bias",
        "agg_type":"texture",
        "plot_spec":{
            "title":"Mean Bias in 40-100cm RSM State wrt Soil Texture " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in 40-100cm RSM",
            },
        },
    ## absolute error, bias, and true value of state per level
    {
        "name":"all-bias-state",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-10"),
            ("err_state", "rsm-40"),
            ("err_state", "rsm-100"),
            ],
        "error_type":"bias",
        "agg_type":"all",
        "plot_spec":{
            "title":"Overall Mean Bias in RSM State (2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Bias in RSM",
            },
        },
    {
        "name":"all-abs-err-state",
        "bin_monthly":False,
        "feats":[
            ("err_state", "rsm-10"),
            ("err_state", "rsm-40"),
            ("err_state", "rsm-100"),
            ],
        "error_type":"abs-err",
        "agg_type":"all",
        "plot_spec":{
            "title":"Overall Mean Absolute Error in RSM State " + \
                    "(2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Absolute Error in RSM",
            },
        },
    {
        "name":"all-true-state",
        "bin_monthly":False,
        "feats":[
            ("true_state", "rsm-10"),
            ("true_state", "rsm-40"),
            ("true_state", "rsm-100"),
            ],
        "error_type":"abs-err",
        "agg_type":"all",
        "plot_spec":{
            "title":"True RSM State (2018-2024)\n{minfo}",
            "xlabel":"Initialization Time",
            "ylabel":"Relative Soil Moisture",
            },
        },
    ## multi y-axis forcings
    {
        "name":"all-forcings",
        "bin_monthly":False,
        "feats":[
            ("horizon", "tmp"),
            ("horizon", "spfh"),
            ("horizon", "apcp"),
            ("horizon", "weasd"),
            ],
        "error_type":"abs-err",
        "agg_type":"all-multiy",
        }
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

""" Spatial plots """

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
    ## 1d slices of each tile wrt the pixel axis
    tile_slices = [slice(start_tile,end_tile) for start_tile,end_tile
            in zip(idx_zero_splits[:-1], idx_zero_splits[1:])]
    ## store 1d slices alongside 2d latlon and GridTile objects.
    tiles_info = list(zip(
        ev.attrs["latlon"], ev.attrs["tiles"], tile_slices))

    #'''
    ## Bin and average the feature data within month groupings
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
    ## iterate over seasonal spatial plot types
    for sspix,spkd in enumerate(sspi):
        if spkd["error_type"] != error_type:
            continue
        if spkd["feat"] not in ev.attrs["flabels"]:
            continue
        ix_sp = ev.attrs["flabels"].index(spkd["feat"])
        feats = np.stack([gm[...,ix_sp] for gm in group_means], axis=-1)

        ## If MAPE is requested, go through the whole gridding thing w the mean
        if spkd["feat"][0] == "err_state" and spkd.get("use_mape"):
            ix_mmean = ev.attrs["flabels"].index(
                    ("true_state", spkd["feat"][1]))
            mmean = np.stack([gm[...,ix_mmean] for gm in group_means], axis=-1)
            print(feats.shape, mmean.shape)
            feats /= mmean
            #feats /= np.concatenate([
            #    mmean for i in range(feats.shape[1])
            #    ], axis=1)

        ## Make 2d arrays for each tile and populate them with the feat data
        gridded_feats = []
        mape_feats = []
        for ll,tl,slc in tiles_info:
            tmp_tile_shape = (*ll.shape[:2], feats.shape[-1])
            tmp_tile_feats = np.full(tmp_tile_shape, np.nan)
            ix = ev.indeces[slc]
            ## Batch and sequence axes should be size 1 (marginalized)
            tmp_tile_feats[ix[:,0], ix[:,1],:] = feats[slc,0,:]
            gridded_feats.append(tmp_tile_feats)

        ## Concatenate the 2d arrays of latlon and feats into a full 2d domain
        xt = ev.attrs["domain"].mosaic_shape[-1]
        tile_arrays = [ev.attrs["latlon"], gridded_feats]
        for j,ta in enumerate(tile_arrays):
            rows = [ta[i:i + xt] for i in range(0,len(ta),xt)]
            tile_arrays[j] = np.concatenate(
                    [np.concatenate(x, axis=1) for x in rows], axis=0)
        latlon,feats = tile_arrays

        ## Establish the title and output image path, and plot the groups
        plot_spec = spkd.get("plot_spec", {}).copy()
        if "title" in spkd["plot_spec"].keys():
            plot_spec["title"] = plot_spec["title"].format(
                    minfo=" ".join([model, data_source]))
        substr = "qtrly-" + "-".join([
            s.replace("_","-") for s in spkd["feat"]])
        fname = "_".join([
            "eval-grid", data_source, model, eval_type,
            error_type, substr + ["","-mape"][spkd.get("use_mape",False)]
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
                    **plot_spec,
                    },
                fig_path=fpath,
                )
    #'''

    """ 1D time series """

    sfeats = ev.attrs["model_config"]["feats"]["static_feats"]
    for tspi in time_series_plot_info:
        if tspi["error_type"] != error_type:
            continue
        fidxs = [ev.attrs["flabels"].index(f) for f in tspi["feats"]
                if f in ev.attrs["flabels"]]
        dtimes = [datetime.fromtimestamp(int(t)) for t in ev.time[:,0]]
        fname = "_".join([
            "eval-grid", data_source, model, eval_type, tspi["name"]
            ]) + ".png"

        month_masks = []
        if tspi.get("bin_monthly", False):
            mtimes = np.array([
                datetime.fromtimestamp(int(t)).month
                for t in np.average(ev.time, axis=1)
                ])
            month_masks = [mtimes==m for m in range(1,13)]

        plot_spec = tspi.get("plot_spec", {}).copy()
        if "title" in plot_spec.keys():
            plot_spec["title"] = plot_spec["title"].format(
                    minfo=" ".join([model, data_source]))
        if tspi["agg_type"]=="all-multiy":
            ## binning over all pixels, separated by feature
            feats = [np.average(ev.average[...,f], axis=(1,2)) for f in fidxs]
            if tspi.get("bin_monthly", False):
                for i in range(len(feats)):
                    feats[i] = np.stack([
                        np.average(feats[i][m],axis=0) for m in month_masks
                        ], axis=0)
                    dtimes = [datetime(2000,m,1) for m in list(range(1,13)) ]
                    plot_spec["date_format"] = "%b"
            plotting.plot_time_lines_multiy(
                    time_series=feats,
                    times=dtimes,
                    plot_spec={
                        **time_series_ps_default,
                        "y_labels":[" ".join(f) for f in tspi["feats"]],
                        **plot_spec,
                        },
                    fig_path=fig_dir.joinpath(fname),
                    )

        elif tspi["agg_type"]=="all":
            feats = [np.average(ev.average[...,f], axis=(1,2)) for f in fidxs]
            if tspi.get("bin_monthly", False):
                for i in range(len(feats)):
                    feats[i] = np.stack([
                        np.average(feats[i][m],axis=0) for m in month_masks
                        ], axis=0)
                    dtimes = [datetime(2000,m,1) for m in list(range(1,13)) ]
                    plot_spec["date_format"] = "%b"
            plotting.plot_lines(
                    domain=dtimes,
                    ylines=feats,
                    labels=[" ".join(f) for f in tspi["feats"]],
                    plot_spec={
                        **time_series_ps_default,
                        **plot_spec,
                        },
                    fig_path=fig_dir.joinpath(fname),
                    )

        ## binning over soil textures
        elif tspi["agg_type"]=="texture":
            soil_feats = ("pct_sand", "pct_silt", "pct_clay")
            soil_idxs = tuple(sfeats.index(s) for s in soil_feats)
            soil_rgb = np.clip(ev.static[...,soil_idxs], 0, 1)
            unq_txtr = np.unique(soil_rgb, axis=0)

            legend_labels = []
            txtr_colors = []
            ## If averaging soil textures, assume the legend labels them
            for i in range(unq_txtr.shape[0]):
                tmp_txtr = unq_txtr[i]
                for map_txtr,label,map_color in soil_mapping:
                    if np.all(np.isclose(tmp_txtr, map_txtr, 1e-4, 1e-4)):
                        legend_labels.append(label)
                        #txtr_colors.append(map_color)
                        txtr_colors.append(map_txtr)

            m_txtr = [np.all(soil_rgb==ut, axis=1) for ut in unq_txtr]
            feats = [np.average(ev.average[:,m], axis=(1,2)) for m in m_txtr]
            feats = [f[...,fidxs] for f in feats]

            ## drop silt because there's literally just 1 pixel globally
            if "silt" in legend_labels:
                silt_idx = legend_labels.index("silt")
                feats.pop(silt_idx)
                legend_labels.pop(silt_idx)
                txtr_colors.pop(silt_idx)

            if tspi.get("bin_monthly", False):
                for i in range(len(feats)):
                    feats[i] = np.stack([
                        np.average(feats[i][m],axis=0) for m in month_masks
                        ], axis=0)
                    dtimes = [datetime(2000,m,1) for m in list(range(1,13)) ]
                    plot_spec["date_format"] = "%b"

            plotting.plot_lines(
                    domain=dtimes,
                    ylines=feats,
                    labels=legend_labels,
                    plot_spec={
                        **time_series_ps_default,
                        "colors":txtr_colors,
                        **plot_spec,
                        },
                    fig_path=fig_dir.joinpath(fname),
                    )

        ## binning over elevation levels
        elif tspi["agg_type"]=="elev":
            elev = ev.static[...,sfeats.index("elev")]
            m_elev = [(elev>=emin)&(elev<emax)
                    for emin,emax in tspi["elev_bins"]]
            ebins,feats = zip(*[
                    (b,np.average(ev.average[:,m], axis=(1,2))[...,fidxs])
                    for b,m in zip(tspi["elev_bins"], m_elev)
                    if np.any(m)
                    ])
            feats = list(feats)
            if tspi.get("bin_monthly", False):
                for i in range(len(feats)):
                    feats[i] = np.stack([
                        np.average(feats[i][m],axis=0) for m in month_masks
                        ], axis=0)
                    dtimes = [datetime(2000,m,1) for m in list(range(1,13)) ]
                    plot_spec["date_format"] = "%b"
            plotting.plot_lines(
                    domain=dtimes,
                    ylines=feats,
                    labels=[f"{e0}-{ef} meters" for e0,ef in ebins],
                    plot_spec={
                        **time_series_ps_default,
                        **plot_spec,
                        },
                    fig_path=fig_dir.joinpath(fname),
                    )

        print(f"Generated {fname}")
        print(tspi["agg_type"], [f.shape for f in feats])
