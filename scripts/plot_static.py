"""
Generate masks of each unique class combination of vegetation and soil pixels,
and plot a grid showing the number of pixels in each combination category.
"""
import numpy as np
import pickle as pkl
import h5py
from netCDF4 import Dataset
import json
from pathlib import Path
import matplotlib.pyplot as plt

from testbed.list_feats import umd_veg_classes,statsgo_textures
from testbed.list_feats import umd_veg_lai_bounds,umd_veg_rsmin
from testbed.list_feats import slopetype_drainage,textures_vegstress
from testbed.list_feats import soil_texture_colors,umd_veg_colors
from testbed.plotting import plot_geo_ints,plot_geo_scalar,plot_lines

def get_soil_veg_combo_masks(veg_ints:np.ndarray, soil_ints:np.ndarray,
        print_combos:bool=False):
    """
    Given integer-labeled class masks for soil and vegetation, calculates and
    returns a grid of boolean masks associated with each distinct combination,
    which is useful for stratified and uniform sampling techniques.

    The classes are expected to be assigned with the integer order defined by
    the corresponding indeces in the label lists list_feats.umd_veg_classes
    and list_feats.statsgo_textures

    :@param veg_ints: Integer array of vegetation classes
    :@param soil_ints: Same-shape integer array of soil classes
    :@param print_combos: If True, prints the labels and comma-separated
        counts for each combination

    :@return: 2-tuple of arrays (combos, combo_masks) where combos is a (N,2)
        shaped array of N valid combinations of (vegetation, soil) classes
        identified by their integers, and combo_masks is a (P,Q,N) shaped array
        of boolean masks over the (P,Q) grid identifying the positions of
        samples of each combination.
    """
    ## Create a (N,2) array of each of the N combinations of (vegetation, soil)
    combos = np.reshape(np.stack(
            np.meshgrid(
                np.unique(veg_ints.astype(int)),
                np.unique(soil_ints.astype(int))),
            axis=-1,
            ), (-1, 2))

    ## Get a (P,Q,N) shaped boolean grid setting samples matching each of the
    ## N possible combinations to True.
    combo_masks = np.stack([
            np.logical_and((veg_ints==combos[i,0]),(soil_ints==combos[i,1]))
            for i in range(combos.shape[0])
            ], axis=-1)

    if print_combos:
        for i in range(combos.shape[0]):
            tmp_veg_label = umd_veg_classes[combos[i,0]]
            tmp_soil_label = statsgo_textures[combos[i,1]]
            tmp_num_samples = np.count_nonzero(combo_masks[...,i])
            print(", ".join(
                (tmp_veg_label, tmp_soil_label, str(tmp_num_samples))
                ))
    return combos,combo_masks

def plot_soil_veg_matrix(combos, combo_masks, fig_path:Path,
        title="", vmax=None, cmap="magma", norm="linear"):
    unq_veg = tuple(np.unique(combos[:,0]))
    unq_soil = tuple(np.unique(combos[:,1]))

    matrix = np.zeros((len(unq_veg), len(unq_soil)))
    for i in range(combos.shape[0]):
        tmp_veg_idx = unq_veg.index(combos[i,0])
        tmp_soil_idx = unq_soil.index(combos[i,1])
        matrix[tmp_veg_idx,tmp_soil_idx] = np.count_nonzero(combo_masks[...,i])
    matrix[matrix==0] = np.nan

    print(sorted(list(matrix[np.isfinite(matrix)]))[-12:])
    fig,ax = plt.subplots()
    cb = ax.imshow(matrix, cmap=cmap, vmax=vmax, norm=norm)
    fig.colorbar(cb)

    # Adding labels to the matrix
    ax.set_yticks(
            range(len(unq_veg)),
            [umd_veg_classes[u] for u in unq_veg],
            )
    ax.set_xticks(
            range(len(unq_soil)),
            [statsgo_textures[u] for u in unq_soil],
            rotation=45,
            ha='right',
            )
    plt.title(title, fontsize=14)
    plt.xlabel("STATSGO Soil Textures", fontsize=12)
    plt.ylabel("UMD Vegetation Classes", fontsize=12)

    fig.savefig(fig_path, bbox_inches="tight")
    return matrix

if __name__=="__main__":

    #'''
    proj_root_dir = Path("/rhome/mdodson/testbed")
    gridstat_dir = Path("data/grid_stats")
    static_pkl_path = proj_root_dir.joinpath(
            "data/static/nldas_static_cropped.pkl")

    #grid_bounds,locale = (slice(None,None), slice(None,None)),"full"
    #grid_bounds,locale = (slice(80,108), slice(35,55)),"lt-high-sierra"
    #grid_bounds,locale = (slice(25,50), slice(308,333)),"lt-north-michigan"
    #grid_bounds,locale = (slice(40,65), slice(184,209)),"lt-high-plains"
    grid_bounds,locale = (slice(123,168), slice(259,274)),"lt-miss-alluvial"

    soil_ints_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_statsgo-soil-classes_{locale}.png")
    veg_ints_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_umd-veg-classes_{locale}.png")
    elev_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_elev_{locale}.png")
    elev_stdev_fig_path = proj_root_dir.joinpath(
            f"figures/static/static_elev-stdev_{locale}.png")


    """ Generate pixel masks for each veg/soil class combination """
    ## Load the full-CONUS static pixel grid
    slabels,sdata = pkl.load(static_pkl_path.open("rb"))
    ## Get the integer-identified soil texture and vegetation class grids
    lat = sdata[slabels.index("lat")]
    lon = sdata[slabels.index("lon")]
    int_veg = sdata[slabels.index("int_veg")]
    int_soil = sdata[slabels.index("int_soil")]
    slope = sdata[slabels.index("slope")]
    elev = sdata[slabels.index("elev")]
    elev_std = sdata[slabels.index("elev_std")]
    m_valid = sdata[slabels.index("m_valid")].astype(bool)

    ## Print a table of soil textures with their corresponding properties
    '''
    sprop_labels = [ "porosity", "fieldcap", "wiltingp",
            "bparam", "matricp", "hydcond" ]
    column_labels = ["texture"] + sprop_labels
    rows = []
    rows.append(" & ".join(column_labels) + " \\\\")
    rows.append("\\hline")
    for sint in np.unique(int_soil[m_valid]):
        m_sint = (int_soil == sint) & m_valid
        txtr_label = statsgo_textures[sint]
        columns = [txtr_label]
        for spl in sprop_labels:
            sprop = sdata[slabels.index(spl)][m_sint]
            if spl == "hydcond":
                sprop *= 1000 ## convert to mm/s
            columns.append(f"{np.average(sprop):.3f}")
        rows.append(" & ".join(columns) + " \\\\")
    table = "\n".join(rows)
    print(table)
    '''

    ## print RSM of field capacity and vegetation stress
    #'''
    porosity = sdata[slabels.index("porosity")]
    fieldcap = sdata[slabels.index("fieldcap")]
    wiltingp = sdata[slabels.index("wiltingp")]
    rsm_fieldcap = (fieldcap - wiltingp) / (porosity - wiltingp)
    for sint in np.unique(int_soil[m_valid]):
        label = statsgo_textures[sint]
        m_tmp = (int_soil == sint) & m_valid
        tmp_rsm_fc = np.average(rsm_fieldcap[m_tmp])
        tmp_wp = np.average(wiltingp[m_tmp])
        tmp_por = np.average(porosity[m_tmp])
        tmp_rsm_vs = (textures_vegstress[label]-tmp_wp)/(tmp_por-tmp_wp)
        print(f"{label:<20} {tmp_rsm_fc:<8.3f} {tmp_rsm_vs:<8.3f}")
    exit(0)
    #'''


    ## Plot combination matrix of soil textures and vegetation
    '''
    ## Get masks identifying all unique combinations of veg/soil classes
    combos,combo_masks = get_soil_veg_combo_masks(
            veg_ints=int_veg,
            soil_ints=int_soil,
            print_combos=False,
            )
    ## Restrict counting to valid pixels
    combo_masks = np.logical_and(m_valid[...,np.newaxis], combo_masks)
    ## Make a grid plot of the number of samples within each combination.
    plot_soil_veg_matrix(
            combos=combos,
            combo_masks=combo_masks,
            fig_path=proj_root_dir.joinpath(
                "figures/static/static_soil-veg-combos.png"),
            title="Number of Pixels per Soil and Vegetation\nClass " + \
                    "Combination (Full Domain)",
            #cmap="magma_r",
            cmap="turbo",
            norm="linear",
            #vmax=50000,
            )
    exit(0)
    '''

    ## Plot integer vegetation map
    #'''
    plot_geo_ints(
            int_data=np.where(m_valid, int_veg, np.nan)[*grid_bounds],
            lat=lat[*grid_bounds],
            lon=lon[*grid_bounds],
            geo_bounds=None,
            #int_ticks=np.array(list(range(14)))*(13/14)+.5,
            int_labels=[umd_veg_classes[ix] for ix in range(14)],
            fig_path=veg_ints_fig_path,
            show=False,
            plot_spec={
                "cmap":"tab20b",
                "cbar_pad":0.02,
                #"cbar_orient":"horizontal",
                "cbar_orient":"vertical",
                "cbar_shrink":.8,
                "cbar_tick_rotation":-45,
                "cbar_fontsize":14,
                "title":f"UMD Vegetation Classes ({locale})",
                "title_fontsize":18,
                "interpolation":"none",
                },
            colors=[umd_veg_colors[l] for l in umd_veg_classes],
            )
    print(f"Generated {veg_ints_fig_path.as_posix()}")
    plt.clf()
    #'''

    ## Plot integer soil texture map
    #'''
    int_soils_masked = np.where(m_valid, int_soil, np.nan)
    plot_geo_ints(
            int_data=int_soil[*grid_bounds],
            lat=lat[*grid_bounds],
            lon=lon[*grid_bounds],
            geo_bounds=None,
            #int_ticks=(np.array(list(range(15))))*(14/15)+.5,
            int_labels=[statsgo_textures[ix] for ix in range(15)],
            fig_path=soil_ints_fig_path,
            show=False,
            plot_spec={
                "cmap":"gist_ncar",
                "cbar_pad":0.02,
                #"cbar_orient":"horizontal",
                "cbar_orient":"vertical",
                "cbar_shrink":.8,
                "cbar_tick_rotation":-45,
                "cbar_fontsize":14,
                "title":f"STATSGO Soil Texture Classes ({locale})",
                "title_fontsize":18,
                "interpolation":"none",
                },
            colors=[soil_texture_colors[ix] for ix in range(15)],
            )
    print(f"Generated {soil_ints_fig_path.as_posix()}")
    #exit(0)
    #'''

    ## Plot scalar elevation
    #'''
    plot_geo_scalar(
            data=np.where(m_valid, elev, np.nan)[*grid_bounds],
            latitude=lat[*grid_bounds],
            longitude=lon[*grid_bounds],
            bounds=None,
            plot_spec={
                "title":f"GTOPO30 Elevation in meters ({locale})",
                "cmap":"gnuplot",
                "cbar_label":"Elevation (meters)",
                #"cbar_orient":"horizontal",
                "cbar_orient":"vertical",
                "cbar_shrink":1.,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                },
            fig_path=elev_fig_path,
            )
    print(f"Generated {elev_fig_path.as_posix()}")
    plot_geo_scalar(
            data=np.where(m_valid, elev_std, np.nan)[*grid_bounds],
            latitude=lat[*grid_bounds],
            longitude=lon[*grid_bounds],
            bounds=None,
            plot_spec={
                "title":f"Standard Deviation of Elevation ({locale})",
                "cmap":"gnuplot",
                "cbar_label":"Elevation Std. Deviation (meters)",
                #"cbar_orient":"horizontal",
                "cbar_orient":"vertical",
                "cbar_shrink":1.,
                "cbar_pad":.02,
                "fontsize_title":18,
                "fontsize_labels":14,
                },
            fig_path=elev_stdev_fig_path,
            )
    print(f"Generated {elev_stdev_fig_path.as_posix()}")
    #'''

    ## plot all real-valued static datasets
    '''
    slabels_to_plot = [
            "pct_sand", "pct_silt", "pct_clay", "porosity", "fieldcap",
            "wiltingp", "bparam", "matricp", "hydcond", "elev", "elev_std",
            "slope", "aspect", "vidx", "hidx"
            ]
    for l in slabels_to_plot:
        plot_geo_scalar(
                data=np.where(m_valid, sdata[slabels.index(l)], np.nan),
                latitude=lat,
                longitude=lon,
                bounds=None,
                plot_spec={
                    "title":f"{l.capitalize()} (Full Domain)",
                    "cmap":"gnuplot",
                    "cbar_label":"",
                    "cbar_orient":"horizontal",
                    "cbar_pad":.02,
                    "fontsize_title":18,
                    "fontsize_labels":14,
                    "norm":"linear",
                    },
                fig_path=proj_root_dir.joinpath(
                    f"figures/static/static_{l}.png"),
                )
    '''

    ## Collect data and make plots for each region independently
    '''
    from eval_timegrid import parse_timegrid_path
    from krttdkit.visualize import geoplot as gp
    timegrid_dir = Path("data/timegrids/")
    times,yranges,xranges = zip(*map(
        parse_timegrid_path,
        sorted(timegrid_dir.iterdir())
        ))
    times,yranges,xranges,paths = zip(*sorted(
        (*parse_timegrid_path(p), p)
        for p in timegrid_dir.iterdir()
        ))
    xranges,yranges = set(xranges),set(yranges)

    unique_regions = []
    for x0,xf in sorted(list(set(xranges))):
        for y0,yf in sorted(list(set(yranges))):
            substr = f"y{y0:03}-{yf:03}_x{x0:03}-{xf:03}"
            unique_regions.append((
                substr,
                next(p for p in timegrid_dir.iterdir() if substr in p.name)
                ))

    ## Collect data for each region independently
    for substr,tmp_path in unique_regions:
        with h5py.File(tmp_path, mode="r") as tmp_file:
            tmp_file = h5py.File(tmp_path, mode="r")
            tmp_static = tmp_file["/data/static"][...]
            tmp_slabels = json.loads(
                    tmp_file["data"].attrs["static"]
                    )["flabels"]
            print(f"\n{tmp_path}")
            tmp_combos,tmp_masks = get_soil_veg_combo_masks(
                    veg_ints=tmp_static[...,tmp_slabels.index("int_veg")],
                    soil_ints=tmp_static[...,tmp_slabels.index("int_soil")],
                    print_combos=True,
                    )

            m_valid = tmp_static[...,tmp_slabels.index("m_valid")].astype(bool)
            tmp_masks = np.logical_and(m_valid[...,np.newaxis], tmp_masks)

            ## Plot a vegetation/soil combination matrix
            plot_soil_veg_matrix(
                    combos=tmp_combos,
                    combo_masks=tmp_masks,
                    fig_path=Path(f"figures/static/combos_{substr}.png"),
                    cmap="plasma",
                    norm="linear",
                    vmax="1000",
                    )
            ## Plot a RGB of soil texture percentages
            rgb_soil = np.stack([
                tmp_static[...,tmp_slabels.index("pct_sand")],
                tmp_static[...,tmp_slabels.index("pct_silt")],
                tmp_static[...,tmp_slabels.index("pct_clay")],
                ], axis=-1)
            rgb_soil = (255*rgb_soil).astype(np.uint8)
            rgb_soil[np.logical_not(m_valid)] = 0
            gp.generate_raw_image(
                    rgb_soil,
                    Path(f"figures/static/texture_{substr}.png")
                    )

    '''

    ## plot GVF and LAI based parameters
    '''
    gvf_nc_path = proj_root_dir.joinpath("data/static/NLDAS_gfrac.nc4")
    gvf_nc = Dataset(gvf_nc_path.as_posix(), "r")
    lat1d = gvf_nc["lat"][...][::-1]
    lon1d = gvf_nc["lon"][...]
    lat = np.stack([lat1d for i in range(lon1d.size)], axis=1)
    lon = np.stack([lon1d for i in range(lat1d.size)], axis=0)
    crop_slice = (slice(29,None), slice(2,None))
    gvf = gvf_nc["NLDAS_gfrac"][...][:,::-1]
    #print(lat.shape, lon.shape, gvf.shape)
    months = ["January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December"]
    for i,m in enumerate(months):
        plot_geo_scalar(
                data=np.where(m_valid, gvf[i][*crop_slice], np.nan),
                latitude=lat[*crop_slice],
                longitude=lon[*crop_slice],
                bounds=None,
                plot_spec={
                    "title":m,
                    "cmap":"gnuplot",
                    "cbar_label":"",
                    "cbar_orient":"horizontal",
                    "cbar_pad":.02,
                    "fontsize_title":24,
                    "fontsize_labels":18,
                    "norm":"linear",
                    "vmin":0,
                    "vmax":1,
                    },
                fig_path=proj_root_dir.joinpath(
                    f"figures/static/gvf/gvf_{i+1:02}_{m.lower()}.png"),
                )

    gvf_stats = {}
    lai_stats = {}
    for i,v in enumerate(np.unique(int_veg)):
        veg_key = umd_veg_classes[v]
        gvf_stats[veg_key] = {"means":[], "stdevs":[]}
        lai_stats[veg_key] = {"means":[], "stdevs":[]}
        m_veg = int_veg == v
        for j,m in enumerate(months):
            g = gvf[j][*crop_slice][m_veg]
            gvf_stats[veg_key]["means"].append(np.average(g))
            gvf_stats[veg_key]["stdevs"].append(np.std(g))
            lai_min,lai_max = umd_veg_lai_bounds[veg_key]
            lai = lai_min + g*(lai_max-lai_min)
            lai_stats[veg_key]["means"].append(np.average(lai))
            lai_stats[veg_key]["stdevs"].append(np.std(lai))

    #plot_stats_1d(
    #        data_dict=gvf_stats,
    #        x_labels=months,
    #        fig_path=proj_root_dir.joinpath(
    #            "figures/static/gvf/gvf_monthly_stats.png"),
    #        fill_sigma=1,
    #        fill_between=False,
    #        plot_spec={
    #            "title":"Monthly Average GVF with Standard Deviation",
    #            "xlabel":"Month",
    #            "ylabel":"Green Vegetation Fraction",
    #            "fig_size":(18,9),
    #            "title_size":24,
    #            "label_size":16,
    #            "legend_font_size":16,
    #            "yrange":[0,1],
    #            "legend_ncols":2,
    #            "colors":umd_veg_colors,
    #            "grid":True,
    #            }
    #        )
    veg_classes = list(gvf_stats.keys())
    plot_lines(
            domain=list(range(1,13)),
            ylines=[gvf_stats[v]["means"] for v in veg_classes],
            fig_path=proj_root_dir.joinpath(
                "figures/static/gvf/gvf_monthly_stats.png"),
            labels=veg_classes,
            plot_spec={
                "title":"Monthly Average GVF per Vegetation Class",
                "xlabel":"Month",
                "ylabel":"Green Vegetation Fraction",
                "fig_size":(18,9),
                "title_size":24,
                "label_size":16,
                "legend_font_size":16,
                "yrange":[0,1],
                "xrange":[1,12],
                "line_width":2.5,
                "legend_ncols":1,
                "colors":[umd_veg_colors[v] for v in veg_classes],
                "grid":True,
                }
            )
    plot_lines(
            domain=list(range(1,13)),
            ylines=[lai_stats[v]["means"] for v in veg_classes],
            fig_path=proj_root_dir.joinpath(
                "figures/static/gvf/lai_monthly_stats.png"),
            labels=veg_classes,
            plot_spec={
                "title":"Monthly Average LAI per Vegetation Class",
                "xlabel":"Month",
                "ylabel":"Leaf Area Index",
                "fig_size":(18,9),
                "title_size":24,
                "label_size":16,
                "legend_font_size":16,
                "yrange":[0,7.5],
                "xrange":[1,12],
                "legend_ncols":2,
                "line_width":2.5,
                "colors":[umd_veg_colors[v] for v in veg_classes],
                "grid":True,
                }
            )

    plot_lines(
            domain=list(range(1,13)),
            ylines=[
                np.array(lai_stats[v]["means"]) / umd_veg_rsmin[v]
                for v in veg_classes
                ],
            fig_path=proj_root_dir.joinpath(
                "figures/static/gvf/lai-scaled_monthly_stats.png"),
            labels=veg_classes,
            plot_spec={
                "title":"Monthly Average LAI per Vegetation Class\n" + \
                        "Scaled By Minimum Stomatal Resistance",
                "xlabel":"Month",
                "ylabel":"Scaled Leaf Area Index",
                "fig_size":(18,9),
                "title_size":24,
                "label_size":16,
                "legend_font_size":16,
                #"yrange":[0,7.5],
                "xrange":[1,12],
                "legend_ncols":2,
                "line_width":2.5,
                "colors":[umd_veg_colors[v] for v in veg_classes],
                "grid":True,
                }
            )
    '''

    ## plot the SLOPETYPE integer parameter and associated drainage rates
    '''
    gdas_file = proj_root_dir.joinpath("data/static").joinpath(
            "lis71_input_GDAStbot_viirsgvf_GDASforc.d01_conus3km.nc")

    ncd = Dataset(proj_root_dir.joinpath(gdas_file))
    lon = ncd["lon"][::-1]
    lat = ncd["lat"][::-1]
    st_ints = ncd["SLOPETYPE"][::-1]
    plot_geo_ints(
            int_data=st_ints,
            lat=lat,
            lon=lon,
            geo_bounds=None,
            fig_path=proj_root_dir.joinpath(
                f"figures/static/static_slopetype_3km.png"),
            plot_spec={
                #"cmap":"gist_ncar",
                "cbar_pad":0.02,
                "cbar_orient":"horizontal",
                "cbar_shrink":.9,
                #"cbar_tick_rotation":-45,
                "cbar_fontsize":14,
                "title_size":30,
                "label_size":18,
                "title":f"SLOPETYPE Categorical Parameter",
                "interpolation":"none",
                },
            colors=["white", "orange", "green", "purple", "red", "cyan",
                "blue", "pink", "white", "grey"],
            )

    drainage = np.full(st_ints.shape, np.nan)
    for v in np.unique(st_ints):
        drainage[st_ints==v] = slopetype_drainage.get(v, np.nan)

    drainage[st_ints==0] = np.nan
    plot_geo_scalar(
            data=drainage,
            latitude=lat,
            longitude=lon,
            bounds=None,
            plot_spec={
                "title":"SLOPETYPE Bottom Drainage Efficiency (unitless)",
                "cmap":"gnuplot",
                #"cbar_label":"Bottom Layer Drainage Efficiency",
                "cbar_orient":"horizontal",
                "cbar_pad":.02,
                "cbar_shrink":1.,
                "fontsize_title":24,
                "fontsize_labels":18,
                "cbar_fontsize":14,
                "norm":"linear",
                "vmin":0,
                "vmax":1,
                },
            fig_path=proj_root_dir.joinpath(
                f"figures/static/static_drainage_3km.png"),
            )
    '''
