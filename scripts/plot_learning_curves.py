import numpy as np
from pathlib import Path

import tracktrain as tt

if __name__=="__main__":
    #model_type = "lstm-s2s"
    model_type = "accrnn"
    name_substring = "accrnn"
    metric = "state_only"

    candidate_mds = [tt.ModelDir(p) for p in Path("data/models/new").iterdir()]
    ms = tt.ModelSet([
        md for md in candidate_mds
        if md.config.get("model_type") == model_type
        and name_substring in md.name
        and metric in md.metric_labels
        ])
    model_dirs = []

    ms.plot_metrics(
            metrics=(metric,),
            show=False,
            fig_path=Path(f"figures/learning-curves_accrnn_state.png"),
            use_notes=False,
                plot_spec={
                    "ylim":(0,.75),
                    "xlim":(0,310),
                    #"facecolor":"xkcd:dark grey",
                    "legend_cols":3,
                    "cmap":"nipy_spectral",
                    "xlabel":"Epoch",
                    "ylabel":"State Mean Absolute Error",
                    "title":f"Learning Curves ({name_substring} {metric})",
                    "fontsize_title":24,
                    "fontsize_labels":18,
                    "fontsize_legend":10,
                    },
                )
