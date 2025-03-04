# web visualization

Need a visualization system based on python CGI that wraps Nd arrays,
probably stored according to the CFD standard. The

Need to determine appropriate bit depth and resolution hierarchy of
backend data, and framework for rendering it alongside methods for
interactive data identification and analysis with vectorized visuals
like those made with d3js.

## data portal should provide an easy interface for:

 - viewing spatial data at multiple regional bounds and zoom levels
 - animate over spatial data
 - select points and polygons from spatial data
 - building histograms and time series based on a selected point
   or polygon of points.
 - Applying stored polygons for counties, watersheds to bulk data
   (probably based on hard-coded boolean masks in the backend), and
   calculating histograms, time series, integrated totals, etc.

## data needed for the portal:

 - Climatological, annual, monthly rasters at 1 or more spatial
   resolutions to be displayed and animated
 - Near real-time gridded data rasters at same resolutions for
   display and animation, potentially with derived quantities like
   climatological anomaly of DoY, etc.
 - county and watershed boundary vectors (for web display) and grid
   masks (calculated on-demand, by selection or custom shapefile?)
   for extracting a subset of the data.
 - Near real-time hourly, daily data within recent window which can
   be animated as a time series or raster

## minimum requirements:

 - Raster visualization of multiple features over CONUS with default
   coarsened resolution (possibly option to change resolution).
 - select and zoom to region, reset to full grid
 - click on single pixel to retrieve current value of multiple
   features at that location and time.
    - Button to pop-out view the pixel's position in the histograms
      (monthly, annually, climo monthly/anually); prefetch on select.
    - Button to view the pixel's time series (near real-time, daily,
      monthly); prefetch on pixel select.
 -

## implied requirements:

 - CGI system for retrieving data from a series of timegrid-like
   hdf5 files containing static and dynamic features at varying
   spatial and temporal resolutions. These data need to be formatted
   as close to the data needs as possible
 - JavaScript doesn't have fast array operations, or even any clean
   abstractions over Nd arrays, so CGI should probably return only
   a series of float32 TypedArray subclasses based on the currently
   selected spatial region.
    - preload surrounding timestep rasters based on limited window
      for animating like SPoRT web viewer.
    - don't start loading detailed time series
 - d3js-based time series and histogram visualization
 - point-and-click or hover for local values

## user dataset selection

Time series must be requested by constraining:

 1. time resolution (hourly, daily, monthly, anually)
 2. time bounds (initial and final)
 3. spatial bounds (pixel, polygon of pixels, rectangle of pixels)
 4. features to include

Histograms must be requested by constraining:

 1. histogram profile (single/multi month, single/multi year, climo)
 2. spatial bounds (pixel, polygon of pixels, rectangle of pixels)
 3. features to include

Animations must be requested by constraining:

 1. time resolution (hourly, daily, monthly, anually)
 2. time bounds (initial and final)
 3. spatial bounds (rectangle defined by full screen)
 4. color map

## helpful threads:

 - [stdlib][2] offers CDN-delivered JS API subpackages supporting
   numpy array-like operations written in javascript/c++. Could
   be useful for custom array operations (thresholds, combos, etc).
 - [webpack][3] resolves dependencies in a suite of JS modules and
   creates a "bundle" that can be efficiently loaded (perhaps in
   asynchronous chunks).
 - [running ANNs client-side][4] may be prohibitive for large models,
   but supposedly tensorflow's javascript api is capable of it.
 - typescript enables strong typing and a type checking system via
   the `tsc` command, which transpiles to any vanilla JS version.
 - [chunking in hdf5s][5] suggests that the hdf5 chunk cache only
   perists per instance of an hdf5 reader, so a chunk will only be
   retained as long as the same File object persists. Subsequent
   calls to a cgi-bin for example may not benefit from the chunk
   cache, but could still access data stored in the disk cache.
    - [the hdf group][6] confirms object instances must remain open
      for the cache to persist and offers enterprise solution (HSDS),
      also mentions that web caching may be used if a proxy server is
      set up to maintain a cache, and return cached values if the
      same get request is made again.

[2]:https://github.com/stdlib-js/stdlib?tab=readme-ov-file#install_env_builds_umd
[3]:https://github.com/webpack/webpack?tab=readme-ov-file#introduction
[4]:https://frompolandwithdev.com/neural-network-performance/
[5]:https://www.star.nesdis.noaa.gov/jpss/documents/HDF5_Tutorial_201509/2-2-Mastering%20Powerful%20Features.pptx.pdf
[6]:https://www.hdfgroup.org/2022/10/17/improve-hdf5-performance-using-caching/

### real-time mode:

 - Hourly time series to ~3d, daily time series to 1 year
 - Store more contextual time series data (temp, humidity,
   surface fluxes, etc) for plotting alongisde high-frequency data.

## nginx

Server configuration defined by directives in /etc/nginx. Directives
are key-value pairs, where the the value is enclosed in braces, it
is considered a *context* containing more directives.

The global context has username, log locations etc.

The HTTP context includes one or more server contexts distinguished
by the port they listen on. Requests may be written to an acces log.
The server context uses the `location` directive to point to the
location(s) of files to serve, where location targets can be
determined based on regex on the request path.

The request may also route to a separate server on the network or
internet, thereby creating a "reverse proxy" server to handle
caching, anonymity, or load balancing.

## webassembly

Static compiled language with strict type guarantees (obviously).

Previously, wasm modules were loaded in an ArrayBuffer, then
compiled/instantiated as an object. `WebAssembly.instantiate` has
overloads for compiling and returning a promise exposing the
(WA.Module, WA.Instance) objects associated with it, or if it is
provided a compiled WA.Module, then only a WA.Instance

Module objects are stateless but compiled WASM code that can easily
be shared with workers. When a worker recieves the compiled module,
it instantiates it

see [here][1] for streaming module code compilation and passing the
module to a wasm worker to be instantiated by a listener

wasm modules could be used to multiprocess over data transformations
after arrays have been loaded. This may include changing image
resolution, creating gifs, recalculating rendering rasters based on
data with higher bit depth, compiling time series, etc.

[1]:https://developer.mozilla.org/en-US/docs/WebAssembly/Reference/JavaScript_interface/Module

## canvas

canvas 2d context retrieved with `ctx = canvas.getContext("2d")`,
then `ctx.drawImage()` may be used to render onto the canvas.
