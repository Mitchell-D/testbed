# 20250325 web visualization

At this point, I pretty much have the menu selection forms as well as
their initialization and callback methods completed for the prototype
page. Also I have web space on vortex which doesn't have CGI
capability, but which can host static images with links that are
explicitly included in a JSON.

Today I determined normalization bounds for RGBs of a variety of
feature/metric combinations, and started generating daily imagery.
I will start by keeping a consistent color map at least per combo,
and listing 10 years' worth of data statically.

With that data volume, it will still be prescient to use a templated
file naming scheme to establish image URLs client-side.

Also, it will be unreasonable to query all 10 years of each combo on
selection. As such, a default time range will need to be specified
for each time resolution, and date/time selection forms used to
determine the range of images that are buffered.

So the next order of business is setting up image path fetching based
on the menu and time range selection forms, image ticks with
callbacks for stepping forward/back or jumping around, rendering
methods, and methods to clear and reset the image buffer and canvas
when a new menu or time range is selected.
