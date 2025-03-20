# generalizing timegrid access

## coord feat generators

I need to better determine how abstract generators should be. The
challenge is that there are a multitude of ways that someone might
want to access slices and combinations of CFA objects within an
arbitrarily-defined CFD... In the timegrid format (which implies a
`(T,M,N,F)` dynamic array and `(M,N,F)` static array), there is
ensured to be a direct correspondence between the spatial `(M, N)`
axes, which suggests that defining a spatiotemporal constraint on
`(T,M,N)` and an iteration pattern on `(T,)` is sufficient to
determine generator behavior.

This is *not* the case in general, since there is no construct at
this point to define the relationship between coordinate axes. In the
future, perhaps CFA coordinates can be checked for compatibility
(identical name, size, coordinate data) and broad constraints can be
applied on the non-iteration axes, and iteration defined only along
a single commonly-shared coordinate axis...

For now, I think it would be best to just provide as much utility as
possible in the CFA/CFD classes, and determining iteration patterns
and constraints within the generators. The generators, then, will
expect a certain basic CFD structure that they can check on init.

## gen\_timegrid\_series

In the short term, I need a generator that applies a spatial mask
and time constraints, and which can iterate over multiple CFD files
along the time axis. This is very similar to the behavior of
`generators.gen_timegrid_subgrids`, except that it won't break
up the returned data into separate (w,h,s,si,p) subarrays by default.

Breaking up into particular subarrays to suffice a user-defined CFD
configuration would be an ideal goal in the future, but doing so
implies additional complexity in determining the extracted bounds
per sample, which would be very difficult to generalize.

Much of the `gen_timegrid_subgrids` code is dedicated to determining
the time bounds across files, which is a behavior that could be
generalized for iterating over any coordinate axis, although in
practice it will probably only really be useful for generators which
iterate along a temporal axis.

It is also quite handy to determine feat and derived feat ingredient
indeces before looping so that they don't need to be recalculated
every time within the loop. Ultimately this function needs to be part
of the CFD framework where a source and target CFD are provided,
their derived feat dicts combined, and indeces/functions resolved
ahead of time. Not sure the best way to do this yet though...
maybe something like a call stack?

## gen\_timegrid\_subgrids

The subgrids generator yields data in the sequence format
`( (window, horizon, static, static\_int, time), target )`, but
shares many properties with the eneds of `gen_timegrid_series`.

**outside loop**:

 1. Parse attribute dicts from all input timegrid files, and ensure
    consistency between features' ordering
 2. Make slices for the overall spatial bounds
 3. Use `generators._parse_feat_idxs` to determine the locations of
    stored feats, and derived feats' locations, ingredients, etc.
 4. Determine the position & embed size of each static integer feat.
 5. Use the time arrays to establish temporal bounds on each file's
    contents, making sure the gap between adjacent files isn't large.
 6. Restrict files and time arrays to those overlapping the temporal
    bounds specified by the user
 7. Concatenate the times into a single array representing all files,
    and use the user-specified initial and final times to determine
    the index bounds of each sample (window init to last horizon)
    given the times array size and user-provided frequency.
 8. Establish a list of 3-tuples `(file, init_idx, final_idx)` which
    describe the index bounds within each file wrt the full t axis.
 9. Use combined per-file index bounds and per-slice index bounds to
    make a list of lists specifying the file and slice bounds of each
    sample. The first level corresponds to each yielded sample step,
    and the second contains slices across multipl files, if needed.

**inside loop**

(iterating over sample chunks)

 1. Check if sample excludes any currently-open files. If so, close.
 2. Extract all feat data within the sample time bounds and
    user-provided spatial coordinate bounds.
 3. extract `(w,h,s,si,p)` subarrays and use them to calculate
    derived feats with `generators._calc_feat_array`
 4. yield the result
