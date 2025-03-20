# web visualization meeting

## directions:

 - focus on one time resolution for now (daily)
 - county-level time series rather than pixel-wise (?)
 - pan + zoom full resolution w/ point and click
 - setting data bounds and color map is a high priority

 - HUC or county-wise aggregation for time series
 - prioritize low-bandwidth working prototype with static images

## other examples

 - veda
    - uses mapboxgl framework for rendering, issuing requests, etc
    - mapbox and openstreetmap APIs for satellite/vector basemap
      openveda cloud API for raster tiles.
    - totally cloud-based lambda instances
    - spatial time series not available;
 - farms.etdata.org
    - leaflet tiling webmap
    - echarts visualization
    - day.js for datetime processing
    - geoman for map tools on leaflet (available for MapLibre)
    - lodash provides functional utilities for vanilla JS arrays
 - app.climateengine.org
    - google earth engine
 - cocorahs data explorer dashboard
    - highcharts for 1d data visualizations
 - nws radar.weather.gov
    - web map service protocol; OGC web service
    - uses ArcGIS basemap so probably their wms
 - weather.cod.edu
    - rendering static images in \<canvas/\> per-domain
    - all soundings are also static images referenced by click
      location
    - d3js for menu utilities
 - tropicaltidbits.com
    - static 2d images rendered with \<img/\> rather than \<canvas/\>
    - click-and-drag soundings are generated as PNGs server-side
      on-demand.
    - mousetrap.js for keyboard shortcuts
 - pivotalweather.com
    - 2d graphics from static images, with \<map/\> frame used to
      interact with clickable frame of \<img/\>
    - jvectormap for something, probably tooltips and on-map cursor.
      soundings embedded in <iframe/> subpage
    - soundings generated serverside on-demand
 - products.climate.ncsu.edu/fire/
    - openlayers for tile map rendering on \<canvas/\>, and for
      overlay containers on click.
    - Border polygons are stored as per-tile static PNGs.
    - querying iowa state mesonet wms for radar png tiles, ncsu for
      forcings, esri for basemap, etc.
    - returns data value on click via request to server w/ latlon
 - southeast regional climate center


## utilities

 - **webgl** performance comes from parallelization via Single
   Instruction Multiple Threads (SIMD) processing. Some potential to
   use shaders as a array type for multithreaded processing.
 - **leaflet**
 - **docker**
 - **maplibre**
   https://maplibre.org/maplibre-gl-js/docs/examples/animate-images/

## thoughts

After talking with Chris, Ryan, and Rob, the goal of this project is
ultimately to make a visualization framework that can be applied to
RxBurn, Heat Stress, and NLDAS-3 visualization.

For some time, we discussed pre-calculating county-wise time series,
perhaps to 180 days historically. Rendering and transferring the full
resolution will be much more problematic than storing with this
time range.

Due to the dearth of reasonable array manipulation libraries for
client-side processing, I think the server should take user-specified
value range bounds and saturate/convert to uint8 images prior to
returning, rather than the client doing processing from data coords.
