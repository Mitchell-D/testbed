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
        (1,"nswrs"),        ## net shortwave at surface
        (2,"nlwrs"),        ## net longwave at surface
        (3,"lhtfl"),        ## latent heat flux
        (4,"shtfl"),        ## sensible heat flux
        (5,"gflux"),        ## ground heat flux
        (10,"arain"),       ## liquid precipitation
        (11,"evp"),         ## evapotranspiration
        (12,"ssrun"),       ## surface runoff
        (13,"bgrun"),       ## sub-surface runoff
        (19,"tsoil-10"),    ## depth-wise soil temperature
        (20,"tsoil-40"),
        (21,"tsoil-100"),
        (22,"tsoil-200"),
        (26,"soilm-10"),    ## depth-wise soil moisture content
        (27,"soilm-40"),
        (28,"soilm-100"),
        (29,"soilm-200"),
        (30,"lsoil-10"),    ## depth-wise liquid soil moisture
        (31,"lsoil-40"),
        (32,"lsoil-100"),
        (33,"lsoil-200"),
        (34,"mstav-200"),   ## moisture availability 0-200cm
        (35,"mstav-100"),   ## moisture availability 0-100cm
        (36,"evcw"),        ## canopy water evaporation
        (37,"trans"),       ## transpiration
        (38,"evbs"),        ## bare soil evaporation
        (49,"rsmin"),       ## minimal stomatal resistance
        (50,"lai"),         ## leaf area index
        (51,"veg"),         ## vegetation fraction
        )
