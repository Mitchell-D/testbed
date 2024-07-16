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
        (10,"apcp"),        ## hourly precip total (kg/m^2)
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

    ('res_soilm-10', (None, None, 0, 0.32)),
    ('res_soilm-40', (None, None, 0, 0.25)),
    ('res_soilm-100', (None, None, 0, 0.15)),
    ('res_soilm-200', (None, None, 0, 0.081)),
    ('res_weasd', (None, None, 0, 0.1)),
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
