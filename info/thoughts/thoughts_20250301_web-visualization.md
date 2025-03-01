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
