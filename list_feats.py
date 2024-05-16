nldas_record_mapping = (
        (1,"tmp"),          ## 2m temperature
        (2,"spfh"),         ## 2m specific humidity
        (3,"pres"),         ## surface pressure
        (4,"ugrd"),         ## 10m zonal wind speed
        (5,"vgrd"),         ## 10m meridional wind speed
        (6,"dlwrf"),        ## downward longwave radiative flux
        (7,"ncrain"),       ##
        (8,"cape"),         ## convective available potential energy
        (9,"pevap"),        ## hourly potential evaporation
        (10,"apcp"),        ## hourly precip total
        (11,"dswrf"),       ## downward shortwave radiative flux
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
