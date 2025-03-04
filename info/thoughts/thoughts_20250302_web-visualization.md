# web visualization

It would be really cool to see the pixels' position in multiple
histograms for different features simlutaneously, which each have
unique characteristic scales. Perhaps multiple narrow vertical
violin plots with a line through the selected point on each feat
axis.

## data sizes per type

5-10km seems reasonable for the finest-resolution daily averages,
both of which would be finer-resolution than nldas2. Would be great
to collect histograms at this resolution too.

Perhaps coarser over full CONUS domain, finer in focus regions?

Each feat histogram is an array like `(P, B, 1)` (P pixels, B bins).
Features have independent coordinate (min/max) bounds, and the number
of bins may vay between feats, so the storage size is
`sum(nbits * P * B for B in nbins_per_feat)` where `nbins_per_feat`
contains the unique bounds of the histograms.

Each time series is an array like `(P, S, F)`  given S sequence steps
per pixel over F features. Features may be forward-differenced or
integrated on the client side.

Each raster time series is like `(S,X,Y,F)`, which is probably too
large to transfer over network in data coordinates. Perhaps
there should be optional data bounds provided in the client request,
so the cgi can saturate outside of the provided range and and
discretize the colors within a limited interval.

Make it easy for users to export a JSON configuration for images or
animations they generate (color bounds, datasets, time ranges, etc).

## storage format

It may be reasonable to consider zarr arrays for cgi backend, as
[this][1] report suggests it has a lower error rate and faster than
netcdf api.

[1]:https://ntrs.nasa.gov/api/citations/20200001178/downloads/20200001178.pdf
