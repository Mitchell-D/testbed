"""
Ordered listings of features (stored and derived) and transforms for converting
among and between them.

Don't change norm parameters lightly, because unfortunately many models do not
explicitly store normalization coefficients they used and rely on them being
stored here. Need to fix this in the future.
"""
## UMD Land cover vegetation classes ordered by index according to:
## https://ldas.gsfc.nasa.gov/nldas/vegetation-class
umd_veg_classes = [
        "water", "evergreen-needleleaf", "evergreen_broadleaf",
        "deciduous-needleleaf", "deciduous-broadleaf", "mixed-cover",
        "woodland", "wooded-grassland", "closed-shrubland", "open-shrubland",
        "grassland", "cropland", "bare", "urban"
        ]

## STATSGO soil textured ordered by index according to:
## http://www.soilinfo.psu.edu/index.cgi?soil_data&conus&data_cov&fract&methods
statsgo_textures = [
        "other", "sand", "loamy-sand", "sandy-loam", "silty-loam", "silt",
        "loam", "sandy-clay-loam", "silty-clay-loam", "clay-loam",
        "sandy-clay", "silty-clay", "clay", "organic-materials", "water",
        ]

## hardcoded version of the statsgo composition lookup table (sand, silt, clay)
## http://www.soilinfo.psu.edu/index.cgi?soil_data&conus&data_cov&fract&methods
statsgo_texture_default = {
        1: ('sand', 'S',                  [0.92, 0.05, 0.03]),
        2: ('loamy_sand', 'LS',           [0.82, 0.12, 0.06]),
        3: ('sandy_loam', 'SL',           [0.58, 0.32, 0.1 ]),
        4: ('silty_loam', 'SiL',          [0.17, 0.7 , 0.13]),
        5: ('silt', 'Si',                 [0.1 , 0.85, 0.05]),
        6: ('loam', 'L',                  [0.43, 0.39, 0.18]),
        7: ('sandy_clay_loam', 'SCL',     [0.58, 0.15, 0.27]),
        8: ('silty_clay_loam', 'SiCL',    [0.1 , 0.56, 0.34]),
        9: ('clay_loam', 'CL',            [0.32, 0.34, 0.34]),
        10: ('sandy_clay', 'SC',          [0.52, 0.06, 0.42]),
        11: ('silty_clay', 'SiC',         [0.06, 0.47, 0.47]),
        12: ('clay', 'C',                 [0.22, 0.2 , 0.58]),
        13: ('organic_materials', 'OM',   [0., 0., 0.]),
        14: ('water', 'W',                [0., 0., 0.]),
        15: ('bedrock', 'BR',             [0., 0., 0.]),
        16: ('other', 'O',                [0., 0., 0.]),
        0: ('other', 'O',                 [0., 0., 0.]),
        }

nldas_record_mapping = (
        (1,"tmp"),          ## 2m temperature (K)
        (2,"spfh"),         ## 2m specific humidity (kg/kg)
        (3,"pres"),         ## surface pressure (Pa)
        (4,"ugrd"),         ## 10m zonal wind speed (m/s)
        (5,"vgrd"),         ## 10m meridional wind speed (m/s)
        (6,"dlwrf"),        ## downward longwave radiative flux (W/m^2)
        #(7,"ncrain"),       ## percentage of precip that is convective
        #(8,"cape"),         ## convective available potential energy (J/kg)
        (9,"pevap"),        ## hourly potential evaporation (kg/m^2)
        (10,"apcp"),        ## hourly precip total (kg/m^2) or (mm)
        (11,"dswrf"),       ## downward shortwave radiative flux (W/m^2)
        )

noahlsm_record_mapping = (
        ## Water (kg/m^3)
        (9,"asnow"),        ## ice water precipitation
        (10,"arain"),       ## liquid precipitation
        (11,"evp"),         ## evapotranspiration
        (12,"ssrun"),       ## surface runoff (not infiltrated)
        (13,"bgrun"),       ## sub-surface runoff (base flow)
        (14,"snom"),        ## snow melt
        (17,"weasd"),       ## water equivalence of snow depth
        (18,"cnwat"),       ## canopy surface water
        (26,"soilm-10"),    ## depth-wise soil moisture content
        (27,"soilm-40"),
        (28,"soilm-100"),
        (29,"soilm-200"),
        #(30,"lsoil-10"),    ## depth-wise liquid soil moisture
        #(31,"lsoil-40"),
        #(32,"lsoil-100"),
        #(33,"lsoil-200"),

        ## Temperature (K)
        (19,"tsoil-10"),    ## depth-wise soil temperature
        (20,"tsoil-40"),
        (21,"tsoil-100"),
        (22,"tsoil-200"),

        ## Energy (W/m^2)
        #(1,"nswrs"),        ## net shortwave at surface
        #(2,"nlwrs"),        ## net longwave at surface
        #(3,"lhtfl"),        ## latent heat flux
        #(4,"shtfl"),        ## sensible heat flux
        #(5,"gflux"),        ## ground heat flux
        (36,"evcw"),        ## canopy water evaporation
        (37,"trans"),       ## transpiration
        (38,"evbs"),        ## bare soil evaporation

        ## Unitless
        (50,"lai"),         ## leaf area index
        (51,"veg"),         ## vegetation fraction
        #(34,"mstav-200"),   ## moisture availability 0-200cm
        #(35,"mstav-100"),   ## moisture availability 0-100cm
        )

dynamic_coeffs = [
    ('apcp', (0.0, 20.22250, 0.08855, 0.47219)),
    ('tmp', (256.75613, 311.28411, 285.62249, 5.32480)),
    ('pres', (64062.57812, 103205.84375, 92850.13281, 514.97894)),
    ('spfh', (0.00056, 0.02003, 0.00676, 0.00184)),
    ('ugrd', (-11.12481, 17.01286, 0.79677, 2.60494)),
    ('vgrd', (-14.37435, 14.93453, 0.61698, 2.95166)),
    ('dswrf', (0.0, 1137.35485, 193.26661, 240.59626)),
    ('dlwrf', (137.37471, 467.67932, 303.53692, 38.11955)),
    ('lai', (0.05999, 6.0, 2.94113, 0.13248)),
    ('veg', (0.0, 0.87129, 0.36952, 0.02366)),

    ('soilm-10', (3.83194, 43.56611, 24.22323, 2.33468)),
    ('soilm-40', (7.61064, 126.07396, 73.03880, 4.30944)),
    ('soilm-100', (12.0, 239.98191, 136.24519, 5.32982)),
    ('soilm-200', (20.00722, 463.33789, 238.34559, 5.01244)),
    ('weasd', (0.0, 805.42059, 4.60062, 1.53773)),

    ('rsm-10', (-.4, 1.1, .455, .242)),
    ('rsm-40', (-.4, 1.1, .458, .186)),
    ('rsm-100', (-.4, 1.1, .409, .141)),
    ('rsm-200', (-.4, 1.1, .421, .130)),
    ('rsm-fc', (-.4, 1.1, .427, .122)),

    ('evbs', (0., 602.2, 11., 15.)),
    ('trans', (0., 602.2, 11., 15.)),
    ('cnwat', (0., 602.2, 11., 15.)),

    ## Residual normalization coeffs in data coordinates;
    ## NOT in already-normalized coordinates
    #('res_soilm-10', (-1., 1.5, 0, 0.32)),
    #('res_soilm-40', (-1., 1.5, 0, 0.25)),
    #('res_soilm-100', (-.5, 1., 0, 0.15)),
    #('res_soilm-200', (-.5, 1., 0, 0.081)),
    #('res_weasd', (-.5, 1., 0, 0.1)),
    #('res_rsm-10', (None, None, 0, .0078)),
    #('res_rsm-40', (None, None, 0, .0020)),
    #('res_rsm-100', (None, None, 0, .00054)),
    #('res_rsm-200', (None, None, 0, .00018)),
    #('res_rsm-fc', (None, None, 0, .00054)),

    ## Residual norm coeffs in already-normalized coordinates,
    ## which should be used in the loss function to scale feat residuals
    ('res_rsm-10', (-.05, .05, 0, .0323)),
    ('res_rsm-40', (-.05, .05, 0, .0106)),
    ('res_rsm-100', (-.05, .05, 0, .00384)),
    ('res_rsm-200', (-.05, .05, 0, .00138)),
    ('res_rsm-fc', (-.05, .05, 0, .00447)),

    ## Soil temperature normalization coefficients
    ('tsoil-10', (None,  None, 285, 4.8)),
    ('tsoil-40', (None,  None, 285, 2)),
    ('tsoil-100', (None,  None, 285, 1.5)),
    ('tsoil-200', (None,  None, 285, 1.2)),

    ## Residual norm coeffs in already-normalized coordinates,
    ## which should be used in the loss function to scale feat residuals
    ('res_tsoil-10', (None, None, 0, .2)),
    ('res_tsoil-40', (None, None, 0, .035)),
    ('res_tsoil-100', (None, None, 0, .0062)),
    ('res_tsoil-200', (None, None, 0, .0047)),
    ]

'''
## OLD!
dynamic_coeffs = [
        ('lai', (2.98181285, 1.64273885)),
        ('veg', (0.37530722, 0.26120498)),
        ('tmp', (285.7206664 ,  11.93766774)),
        ('spfh', (0.0068803 , 0.00468152)),
        ('pres', (92803.06962 ,  7699.616714)),
        ('ugrd', (0.76728547, 2.9982088 )),
        ('vgrd', (0.64238808, 3.25051813)),
        ('dlwrf', (304.1775354,  68.4974331)),
        ('ncrain', (0.0470666 , 0.19641987)),
        ('cape', (202.2872838, 527.7525554)),
        ('pevap', (0.20229804, 0.2735195 )),
        ('apcp', (0.09068938, 0.63642765)),
        ('dswrf', (194.1786308, 265.0293252)),
        ('soilm-10', (24.30383426,  8.65985905)),
        ('soilm-40', (73.16257206, 21.83291832)),
        ('soilm-100', (136.6062034 ,  40.63990632)),
        ('soilm-200', (239.5253568 ,  65.98121262)),
        ]
'''

static_coeffs = [
        ('lat', (39.0, 8.082823)),
        ('lon', (-96.0, 16.743118)),
        ('m_conus', (0.5048876231527094, 0.4999761105692122)),
        ('int_veg', (5.133024168719212, 4.236818527279419)),
        ('int_soil', (3.9223945504926108, 3.29586169355963)),
        ('pct_sand', (0.3289382889470443, 0.2641296532318484)),
        ('pct_silt', (0.289212592364532, 0.227653693553878)),
        ('pct_clay', (0.12368063038793105, 0.12062372678430285)),
        ('porosity', (0.44338217, 0.020784229)),
        ('fieldcap', (0.35865146, 0.031014577)),
        ('wiltingp', (0.06628392, 0.025511015)),
        ('bparam', (5.54697, 1.5926554)),
        ('matricp', (0.3421833, 0.23236318)),
        ('hydcond', (7.040321e-05, 0.00020800455)),
        ('elev', (717.87305, 669.2913)),
        ('elev_std', (62.56725, 87.51388)),
        ('slope', (0.2663482, 0.35235977)),
        ('aspect', (167.42642, 102.52077)),
        ('vidx', (111.5, 64.66258578188781)),
        ('hidx', (231.5, 133.94495137928865)),
        ]

"""
Resonable histogram bounds based on 12-year min/max gridstats
true bounds are commented out for highly skewed distributions
"""
hist_bounds = {
        "tmp":(230,330),
        "spfh":(0,.03),
        "pres":(60000,105000),
        "ugrd":(-30,30),
        "vgrd":(-30,30),
        "dlwrf":(80,600),
        "pevap":(-1,3),
        #"apcp":(0,150),
        "apcp":(0,15),
        "dswrf":(0,1500),
        #"asnow":(0,120),
        "asnow":(0,15),
        #"arain":(0,50), ## real maxima are >100 but rare
        "arain":(0,5), ## real maxima are >100 but rare
        #"evp":(-.2,2.4),
        "evp":(-.2,0.75),
        "ssrun":(0,110),
        #"bgrun":(0,20),
        "bgrun":(0,2.5),
        #"snom":(0,100),
        "snom":(0,10),
        #"weasd":(0,2000),
        "weasd":(0,250),
        "cnwat":(0,.6),
        "tsoil-10":(230,330),
        "tsoil-40":(230,330),
        "tsoil-100":(230,330),
        "tsoil-200":(230,330),
        #"evcw":(0,1000),
        "evcw":(0,200),
        #"trans":(0,700),
        "trans":(0,300),
        #"evbs":(0,700),
        "evbs":(0,150),
        "lai":(0,7),
        "veg":(0,1),
        "soilm-fc":(40,1000),
        "windmag":(0,45),

        "soilm-10":(2,50),
        "soilm-40":(6,150),
        "soilm-100":(12,300),
        "soilm-200":(20,500),
        "res-soilm-10":(-.5,1.5),
        "res-soilm-40":(-.5,1.5),
        "res-soilm-100":(-.5,1.5),
        "res-soilm-200":(-.5,1.5),
        "res-soilm-fc":(-.5,1.5),
        "err-soilm-10":(-30,30),
        "err-soilm-40":(-60,60),
        "err-soilm-100":(-180,180),
        "err-soilm-200":(-260,260),
        "err-res-soilm-10":(-20,20),
        "err-res-soilm-40":(-20,20),
        "err-res-soilm-100":(-20,20),
        "err-res-soilm-200":(-20,20),

        "rsm-10":(-.2,1.1),
        "rsm-40":(-.2,1.1),
        "rsm-100":(-.2,1.1),
        "rsm-200":(-.2,1.1),
        "rsm-fc":(-.2,1.1),
        "res-rsm-10":(-.15,.25),
        "res-rsm-40":(-.05,.15),
        "res-rsm-100":(-.025,.075),
        "res-rsm-200":(-.01,.05),
        "res-rsm-fc":(-.2,.4),
        "err-rsm-10":(-.6,.6),
        "err-rsm-40":(-.6,.6),
        "err-rsm-100":(-.6,.6),
        "err-rsm-200":(-.6,.6),
        "err-rsm-fc":(-.6,.6),
        "err-res-rsm-10":(-.05,.05),
        "err-res-rsm-40":(-.05,.05),
        "err-res-rsm-100":(-.05,.05),
        "err-res-rsm-200":(-.05,.05),
        "err-res-rsm-fc":(-.05,.05),
        }

"""
Derived features are features that can be calculated on-demand from a
combination of stored dynamic and static features. The specified features
are extracted and provided as arguments (static_args, dynamic_args) to a
transform function specified by a string-encoded lambda object.
"""
derived_feats = {
        ## layerwise relative soil moisture in m^3/m^3
        "rsm-10":(
            ("soilm-10",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.1/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-40":(
            ("soilm-40",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.3/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-100":(
            ("soilm-100",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.6/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-200":(
            ("soilm-200",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/1./1000-s[0])/(s[1]-s[0])",
            ),
        ## full-column relative soil moisture in m^3/m^3
        "rsm-fc":(
            ("soilm-10","soilm-40","soilm-100","soilm-200"),
            ("wiltingp","porosity"),
            "lambda d,s:((d[0]+d[1]+d[2]+d[3])/2000-s[0])/(s[1]-s[0])",
            ),
        ## full-column soil moisture in kg/m^3
        "soilm-fc":(
            ("soilm-10","soilm-40","soilm-100","soilm-200"),
            tuple(),
            "lambda d,s:d[0]+d[1]+d[2]+d[3]",
            ),
        ## Scalar magnitude of wind (rather than directional components)
        "windmag":(
            ("ugrd","vgrd"),
            tuple(),
            "lambda d,s:(d[0]**2+d[1]**2)**(1/2)",
            ),
        #"res-rsm-10":(("rsm-10",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-rsm-40":(("rsm-40",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-rsm-100":(("rsm-100",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-rsm-200":(("rsm-200",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-rsm-fc":(("rsm-fc",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-soilm-10":(("soilm-10",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-soilm-40":(("soilm-40",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-soilm-100":(("soilm-100",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-soilm-200":(("soilm-200",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        #"res-soilm-fc":(("soilm-fc",), tuple(), "lambda d,s:d[1:]-d[:-1]"),
        }

output_conversion_funcs = {
        ## layerwise relative soil moisture in m^3/m^3
        "rsm-10":(
            ("soilm-10",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.1/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-40":(
            ("soilm-40",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.3/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-100":(
            ("soilm-100",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/.6/1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-200":(
            ("soilm-200",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/1./1000-s[0])/(s[1]-s[0])",
            ),
        "rsm-fc":(
            ("soilm-fc",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]/2./1000-s[0])/(s[1]-s[0])",
            ),
        "soilm-10":(
            ("rsm-10",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*.1"
            ),
        "soilm-40":(
            ("rsm-40",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*.3"
            ),
        "soilm-100":(
            ("rsm-100",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*.6"
            ),
        "soilm-200":(
            ("rsm-200",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*1."
            ),
        "soilm-fc":(
            ("rsm-fc",),
            ("wiltingp","porosity"),
            "lambda d,s:(d[0]*(s[1]-s[0])+s[0])*1000*2."
            ),
        }

'''
## potential alternative to derived_feats not currently utilized
transforms = {
        "soilm-10":(
            ("rsm-10","wiltingp","porosity"), "lambda r,w,p:(p-w) * r + w"),
        "soilm-40":(
            ("rsm-40","wiltingp","porosity"), "lambda r,w,p:(p-w) * r + w"),
        "soilm-100":(
            ("rsm-100","wiltingp","porosity"), "lambda r,w,p:(p-w) * r + w"),
        "soilm-200":(
            ("rsm-200","wiltingp","porosity"), "lambda r,w,p:(p-w) * r + w"),
        }
'''
