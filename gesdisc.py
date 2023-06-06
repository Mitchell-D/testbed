
import pygrib
import numpy as np
from pathlib import Path
import requests
import multiprocessing as mp
from datetime import datetime as dt
from datetime import timedelta as td
import subprocess
import shlex
from http.cookiejar import CookieJar
import urllib

import grib_tools

gesdisc_url = "https://hydro1.gesdisc.eosdis.nasa.gov"
# URL for GES DISC NLDAS2 data on a 0.125deg resolution grid
nldas2_url = f"{gesdisc_url}/data/NLDAS/NLDAS_FORA0125_H.002"
nldas2_template = "NLDAS_FORA0125_H.A{YYYYmmdd}.{HH}00.002.grb"
# URL for GES DISC run of Noah-LSM on the NLDAS2 domain
noahlsm_url = f"{gesdisc_url}/data/NLDAS/NLDAS_NOAH0125_H.002"
noahlsm_template = "NLDAS_NOAH0125_H.A{YYYYmmdd}.{HH}00.002.grb"


"""
-1:7:12:130
 61:APCP:Precipitation hourly total [kg/m^2]
157:CAPE:180-0 mb above ground Convective Available Potential Energy [J/kg]
153:CONVfrac:Fraction of total precipitation that is convective [unitless]
205:DLWRF:Longwave radiation flux downwards (surface) [W/m^2]
204:DSWRF:Shortwave radiation flux downwards (surface) [W/m^2]
228:PEVAP:Potential evaporation hourly total [kg/m^2]
  1:PRES:Surface pressure [Pa]
 51:SPFH:2-m above ground Specific humidity [kg/kg]
 11:TMP:2-m above ground Temperature [K]
 33:UGRD:10-m above ground Zonal wind speed [m/s]
 34:VGRD:10-m above ground Meridional wind speed [m/s]
"""

def hourly_noahlsm_urls(t0:dt, tf:dt):
    """
    Returns a list of URLs to hourly Noah LSM files in the EOSDIS archive,
    corresponding to the provided inclusive initial and exclusive final times.

    This method only compiles URL strings based on Goddard's HTTP standard,
    so there is no garuntee that the returned URLs actually link to an existing
    data file.

    Sub-hour time bound components are rounded down to the nearest whole hour.

    :@param: inclusive initial time of data range (up to hour precision)
    :@param: exclusive final time of data range (up to hour precision)
    """
    assert tf>t0
    # Round down to the nearest whole hour.
    t0 = dt(year=t0.year, month=t0.month, day=t0.day, hour=t0.hour)
    tf = dt(year=tf.year, month=tf.month, day=tf.day, hour=tf.hour)
    # Iterate over hours in the time range
    fhours = int(((tf-t0).total_seconds()))//3600
    ftimes = [ t0+td(hours=h) for h in range(fhours) ]
    return [f"{noahlsm_url}/{t.year}/{t.strftime('%j')}/"+\
            noahlsm_template.format(YYYYmmdd=t.strftime("%Y%m%d"),
                                    HH=t.strftime("%H"))
            for t in ftimes]

def hourly_nldas2_urls(t0:dt, tf:dt):
    """
    Returns a list of URLs to hourly NLDAS-2 files in the EOSDIS archive,
    corresponding to the provided inclusive initial and exclusive final times.

    This method only compiles URL strings based on Goddard's HTTP standard,
    so there is no garuntee that the returned URLs actually link to an existing
    data file.

    Sub-hour time bound components are rounded down to the nearest whole hour.

    :@param: inclusive initial time of data range (up to hour precision)
    :@param: exclusive final time of data range (up to hour precision)
    """
    assert tf>t0
    # Round down to the nearest whole hour.
    t0 = dt(year=t0.year, month=t0.month, day=t0.day, hour=t0.hour)
    tf = dt(year=tf.year, month=tf.month, day=tf.day, hour=tf.hour)
    # Iterate over hours in the time range
    fhours = int(((tf-t0).total_seconds()))//3600
    ftimes = [ t0+td(hours=h) for h in range(fhours) ]
    return [f"{nldas2_url}/{t.year}/{t.strftime('%j')}/"+\
            nldas2_template.format(YYYYmmdd=t.strftime("%Y%m%d"),
                                   HH=t.strftime("%H"))
            for t in ftimes]

def gesdisc_auth(username,password):
    """
    Initialize a session using a CookieJar in order to maintain authorization.
    """
    # Create a password manager to deal with the 401 reponse login
    pass_man = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pass_man.add_password(None, nldas2_url, username, password)
    # Initialize a session using a CookieJar and install handlers
    opener = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(pass_man),
        urllib.request.HTTPCookieProcessor(CookieJar()),
        #urllib.request.HTTPHandler(debuglevel=1),
        #urllib.request.HTTPSHandler(debuglevel=1),
        )
    urllib.request.install_opener(opener)
    return opener

def gesdisc_curl(urls:list, dl_dir:Path, skip_if_exists:bool=True, debug=False):
    """
    GES DISC authentication is messy with OAuth in Python. Just curl it
    using the invasive GES DISC cookie file requirements :(
    https://uat.gesdisc.eosdis.nasa.gov/information/howto/How%20to%20Generate%20Earthdata%20Prerequisite%20Files

    :@param urls: List of string URLs to downloadable data files
    :@param dl_dir: local directory to dump downloaded files
    :@param skip_if_exists: Don't download existing files.
    """
    curl_command = "curl -n ~/.urs_cookies -b ~/.urs_cookies -LJO --url" + \
            " {url} -o {dl_path}"
    for u in urls:
        dl_path = dl_dir.joinpath(Path(u).name)
        if dl_path.exists() and skip_if_exists:
            continue
        cmd = shlex.split(curl_command.format(url=u, dl_path=dl_path))
        if debug:
            print(f"cmd")
        subprocess.call(cmd)

def gesdisc_download(urls:list, dl_dir:Path, auth=None, letfail:bool=True,
                  skip_if_exists:bool=True, debug=False):
    """
    Make GET requests to the provided URLs and download the response to a file
    in the provided directory named by the leaf of each URL path, which is
    assumed to be a valid data file.

    This relies on the GES DISC cookie requirements outlined here.
    https://uat.gesdisc.eosdis.nasa.gov/information/howto/How%20to%20Generate%20Earthdata%20Prerequisite%20Files

    :@param urls: List of string URLs to downloadable data files
    :@param dl_dir: local directory to dump downloaded files
    :@param auth: (user,pass) string tuple for a site, if necessary.
    :@param letfail: If True, failed requests don't raise an error.
    :@param skip_if_exists: Don't download existing files.
    """
    #auth = None if not auth else requests.auth.HTTPBasicAuth(*auth)
    opener = gesdisc_auth(*auth)
    for u in [ urls[0] ]:
        #request = urllib.request.Request(urls[0])
        # Skip already-downloaded files by default
        dl_path = dl_dir.joinpath(Path(u).name)
        if dl_path.exists() and skip_if_exists:
            continue
        try:
            #urllib.request.urlretrieve(u, dl_path)
            #request = urllib.request.Request(u)
            #response = urllib.request.urlopen(request)
            response = opener.open(u, timeout=15)
        except Exception as E:
            if letfail:
                print(E)
                continue
            raise E
        response.read()
        print(f"Download success: {dl_path}")
    return

def nldas2_to_time(nldas2_path):
    """
    Parse the second and third fields of file names as a datetime

    This method is identical to noahlsm_to_time, but separate since I'm not
    sure whether the entire EOSDISC archive or GES DISC DAAC follow standard.

    :@param: nldas2_path conforming to the GES DISC standard of having the
        2nd and 3rd '.' -separated file name fields correspond to the date,
        for example: "NLDAS_FORA0125_H.A20190901.0000.002.grb"
    """
    return dt.strptime("".join(nldas2_path.name.split(".")[1:3]),"A%Y%m%d%H%M")

def noahlsm_to_time(nldas2_path):
    """
    Parse the second and third fields of file names as a datetime.

    This method is identical to nldas2_to_time, but separate since I'm not
    sure whether the entire EOSDISC archive or GES DISC DAAC follow standard.

    :@param: noahlsm_path conforming to the GES DISC standard of having the
        2nd and 3rd '.' -separated file name fields correspond to the date,
        for example: "NLDAS_NOAH0125_H.A20190901.0000.002.grb"
    """
    return dt.strptime("".join(nldas2_path.name.split(".")[1:3]),"A%Y%m%d%H%M")

if __name__=="__main__":
    debug = True
    data_dir = Path("data/")
    nldas2_dir = data_dir.joinpath("nldas2_2019")
    noahlsm_dir = data_dir.joinpath("noahlsm_2019")
    #init_time = dt(year=2019, month=1, day=1)
    #final_time = dt(year=2020, month=5, day=1)
    #init_time = dt(year=2019, month=5, day=1)
    #final_time = dt(year=2020, month=9, day=1)
    init_time = dt(year=2019, month=9, day=1)
    final_time = dt(year=2020, month=12, day=1)

    '''
    """ Download all of the files within the provided time range """
    # Generate strings for each hourly nldas2 file in the time range
    nldas_urls = hourly_nldas2_urls(t0=init_time, tf=final_time)
    gesdisc_curl(nldas_urls, nldas2_dir, debug=debug)

    # Generate strings for each hourly Noah-LSM file in the time range.
    lsm_urls = hourly_noahlsm_urls(t0=init_time, tf=final_time)
    # Download the Noah LSM files
    gesdisc_curl(lsm_urls, noahlsm_dir, debug=debug)
    '''

    #'''
    """ Print information about a sample file """
    # NLDAS2 LSM forcings
    sample_file = nldas2_dir.joinpath(
            Path("NLDAS_FORA0125_H.A20190101.0000.002.grb"))
    # Noah-LSM run on NLDAS2 domain
    sample_file = noahlsm_dir.joinpath(
            Path("NLDAS_NOAH0125_H.A20190901.0000.002.grb"))
    for w in grib_tools.wgrib(sample_file):
        print(w)
    data, info, geo = grib_tools.get_grib1_data(sample_file)
    #nldas2_to_time(sample_file)
    noahlsm_to_time(sample_file)
    #'''
