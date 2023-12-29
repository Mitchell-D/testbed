""" Simple script downloading NoahLSM and NLDAS-2 from GES DISC """
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import pickle as pkl

from krttdkit.acquire import gesdisc
from krttdkit.acquire import grib_tools
#from GeoTimeSeries import GeoTimeSeries as GTS

def get_nldas_noahlsm(init_time, final_time, nldas_dir, noahlsm_dir,
        debug=False, size_thresh_bytes=2e5):
    """
    Download all of the files within the provided time range to the
    corresponding directories
    """
    # Generate strings for each hourly NLDAS2 file in the time range
    nldas_urls = gesdisc.hourly_nldas2_urls(t0=init_time, tf=final_time)
    # Generate strings for each hourly Noah-LSM file in the time range.
    lsm_urls = gesdisc.hourly_noahlsm_urls(t0=init_time, tf=final_time)
    # Download the NLDAS2 files
    dl_nl = gesdisc.gesdisc_curl(nldas_urls, nldas_dir, debug=debug)
    # Download the Noah LSM files
    dl_no = gesdisc.gesdisc_curl(lsm_urls, noahlsm_dir, debug=debug)

    ## Check that all downloaded files exist, and that they meet the
    ## threshold requirement. This is important because curl will quietly
    ## save the error HTML response if a download fails.

    ## You may also find partially-downloaded files. It's harder to check
    ## for these using thresholds since they may be close to the full size.
    ## I've been doing it manually so far with ls -h | grep, but there
    ## must be a better way.

    for f in dl_nl+dl_no:
        if not f.exists():
            print(f"DNE: {f.as_posix()}")
        elif f.stat().st_size < size_thresh_bytes:
            print(f"Size < thresh: {f.as_posix()}")


if __name__=="__main__":
    debug = False
    data_dir = Path("/rstor/mdodson/thesis/")

    years = list(range(2015, 2022))
    #years = [2016, 2018,2019, 2021]
    for y in years:
        nldas_dir = data_dir.joinpath(f"nldas2/{y}")
        noahlsm_dir = data_dir.joinpath(f"noahlsm/{y}")

        '''
        print(y)
        nl = [f.stat().st_size for f in nldas_dir.iterdir()]
        no = [f.stat().st_size for f in noahlsm_dir.iterdir()]
        print(min(nl), max(nl))
        print(min(no), max(no))
        print()
        '''
        get_nldas_noahlsm(
                init_time=datetime(year=y, month=1, day=1),
                final_time=datetime(year=y+1, month=1, day=1),
                nldas_dir=nldas_dir,
                noahlsm_dir=noahlsm_dir,
                debug=debug,
                )
        #'''
