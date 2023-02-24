"""
REGRID_CUDEM_ROMS_BATHY

Download CUDEM data from http site and regrid to a given ROMS grid. 

usage:

1) edit relevant variables in the section labeled below.*
2) execute file using something like:
    % conda activate "RELEVANT_EVIRONEMENT"
    % python REGRID_CUDEM_BATHY.py

*- see README
Created by Elias Hunter, hunter@marine.rutgers.edu, 1/13/2023 
"""


#Import Necessary Modules
import xarray as xr
import numpy as np
import requests
import time,os,shutil
import rioxarray
from shapely import geometry
import cartopy.io.shapereader as shpreader
import netCDF4
import xesmf as xe
import zipfile
from dask.distributed import Client
from scipy import signal, datasets,ndimage



########################################################################
#EDIT BETWEEN HERE
########################################################################
# INPUT GRID FILE 
#gfile=r'/home/hunter/roms/NOPP/DEM/SANIBEL_ROMS_FORECAST_11R_JUN2021_MAX3.nc'
# Output GRID FILE 
#newgfile=r'/home/hunter/roms/NOPP/DEM/SANIBEL_ROMS_FORECAST_11R_JUN2021_MAX3_CUDEM.nc'

# INPUT GRID FILE 
gfile=r'/home/hunter/roms/NOPP/grids/michael_grd3.nc'
# Output GRID FILE 
newgfile=r'//home/hunter/roms/NOPP/grids/michael_grd3_CUDEM.nc'


"""
1) Download and uzip the shapefiles contained the DEM tile footprints. The 9th arc-sec shape file is at 
    https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/ in 
    https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip. 
 The 3rd arc-sec shape file is at 
 https://coast.noaa.gov/htdata/raster2/elevation/NCEI_third_Topobathy_2014_8580/ in 
 https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip. Download and unzip both of these files, note the location. 
import zipfile

r = requests.get(url)

with open('test.zip','wb') as f:
    f.write(r.content)
    
with zipfile.ZipFile(r'./test.zip', 'r') as zip_ref:
    zip_ref.extractall(r'./')
    
    
"""


    
# Shape file locations for CUDEM tiles. 
shpfile_9_url=r'https://coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/tileindex_NCEI_ninth_Topobathy_2014.zip'
shpfile_9_dir=r'/home/hunter/roms/NOPP/DEM/tileindex_NCEI_ninth_Topobathy_2014/'
shpfile_9=shpfile_9_dir+r'tileindex_NCEI_ninth_Topobathy_2014.shp'
datadir_9=r'/home/hunter/roms/NOPP/DEM/ncei_nintharcsec_dem_data/'
urlprefix_9=r'https://chs.coast.noaa.gov/htdata/raster2/elevation/NCEI_ninth_Topobathy_2014_8483/'

shpfile_3_url=r'https://coast.noaa.gov/htdata/raster2/elevation/NCEI_third_Topobathy_2014_8580/tileindex_NCEI_third_Topobathy_2014.zip'
shpfile_3_dir=r'/home/hunter/roms/NOPP/DEM/tileindex_NCEI_third_Topobathy_2014/'
shpfile_3=shpfile_3_dir+r'tileindex_NCEI_third_Topobathy_2014.shp'
datadir_3=r'/home/hunter/roms/NOPP/DEM/ncei_thirdarcsec_dem_data/'
urlprefix_3=r'https://coast.noaa.gov/htdata/raster2/elevation/NCEI_third_Topobathy_2014_8580/'

#set a depth cutoff, positive down. 
depthcut=-3.0
# set a datum adjustment to correct CUDEM data. 
datum_adjust=0.0

TMPZIP=r'tmp.zip'


#-------------------------------
#SMOOTH settings
#-------------------------------
filterflag=1#1:regrid and smmoth, #2 Just regrid 3. Just smooth  
smgfile=r'//home/hunter/roms/NOPP/grids/michael_grd3_CUDEM_SMOOTH.nc'

filtertype=1 # 1). Boxcar, 2) Gaussian 

BC_np=5
G_sigma=2
G_order=0
G_mode='reflect'


########################################################################
#AND HERE
########################################################################

############################################################################
#Main Program
############################################################################

def main():
    #Download and extract shapefiles.
    print('Getting and Extracting 9th arc-sec tile shapefile')
    r = requests.get(shpfile_9_url)
    if r.status_code==404:
        print('FILE NOT FOUND: ' + shpfile_9_url)
    with open(shpfile_9_dir+TMPZIP,'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile(shpfile_9_dir+TMPZIP, 'r') as zip_ref:
        zip_ref.extractall(shpfile_9_dir)
            
    
    print('Getting and Extracting 3rd arc-sec tile shapefile')
    r = requests.get(shpfile_3_url)
    if r.status_code==404:
        print('FILE NOT FOUND: ' + shpfile_3_url)
    with open(shpfile_3_dir+TMPZIP,'wb') as f:
        f.write(r.content)    
    with zipfile.ZipFile(shpfile_3_dir+TMPZIP, 'r') as zip_ref:
        zip_ref.extractall(shpfile_3_dir)
    
    
    #Extract Grid Information
    grd = xr.open_dataset(gfile,chunks={'eta_rho':900,'xi_rho':600}) 
    grd=grd.set_coords(('lat_rho','lon_rho'))
    grdH=grd.h.load()
    grdH = grdH.rename({"lon_rho": "lon", "lat_rho": "lat"})
    
    glon=np.concatenate((grdH.lon[0,:].values,grdH.lon[:,-1].values,np.flip(grdH.lon[-1,:].values),np.flip(grdH.lon[:,0].values)))  
    glat=np.concatenate((grdH.lat[0,:].values,grdH.lat[:,-1].values,np.flip(grdH.lat[-1,:].values),np.flip(grdH.lat[:,0].values)))  
    llbounds=np.column_stack((glon,glat))
    
    #Create new grid file
    shutil.copy(gfile,newgfile)
    
    
    # Get CUDEM 9th arc-sec data from coast.noaa.gov 
    #Identify files that overlap the ROMS grid and download them. 
    BBOX=geometry.Polygon(llbounds)
    a=shpreader.Reader(shpfile_9)
    b=a.records()
    GLIST9=[]
    flist9=[]
    for c in b:
        tgeom=c.geometry
        if tgeom.intersects(BBOX):
            GLIST9.append(tgeom)
            location=c.attributes['location']
            tmp=location.split('/')
            image_url=urlprefix_9+location
            print(image_url)
            outfile=datadir_9+tmp[-1]
            flist9.append(outfile)
            
            if os.path.exists(flist9[-1]):
                print('FILE EXISTS SKIPPING:'+outfile)
                continue
                
            print('Getting:'+image_url)
            r = requests.get(image_url)
            if r.status_code==404:
                print('FILE NOT FOUND: ' + image_url)
            
            
            print('Writing:'+tmp[-1])
            
            with open(outfile,'wb') as f:
                f.write(r.content)
                
                
    elements9 = []
    
    for file in flist9:
        print(file)
        tmp=rioxarray.open_rasterio(file)
        tmp=tmp.squeeze() 
        elements9.append(tmp)
    
    if not elements9: 
        print("NO overlapping CUDEM 9th arc-sec files")
    # Get CUDEM 3th arc-sec data from coast.noaa.gov 
    #Identify files that overlap the ROMS grid and download them. 
    
    BBOX=geometry.Polygon(llbounds)
    a=shpreader.Reader(shpfile_3)
    b=a.records()
    GLIST3=[]
    flist3=[]
    for c in b:
        tgeom=c.geometry
     #   print(tgeom)
     #   print(c.attributes['location'])
        if tgeom.intersects(BBOX):
            GLIST3.append(tgeom)
            location=c.attributes['location']
          #  print(location)
          #  flist.append(location)
            tmp=location.split('/')
            image_url=urlprefix_3+location
            print(image_url)
            outfile=datadir_3+tmp[-1]
            flist3.append(outfile)
            
            if os.path.exists(flist3[-1]):
                print('FILE EXISTS SKIPPING:'+outfile)
                continue
                
            print('Getting:'+image_url)
            r = requests.get(image_url)
            if r.status_code==404:
                print('FILE NOT FOUND: ' + image_url)
            
            
            print('Writing:'+tmp[-1])
            
            with open(outfile,'wb') as f:
                f.write(r.content)
                
                
    elements3 = []
    
    for file in flist3:
        print(file)
        tmp=rioxarray.open_rasterio(file)
        tmp=tmp.squeeze() 
        elements3.append(tmp)
    
    if not elements3: 
        print("NO overlapping CUDEM 3rd arc-sec files")
    
    df=grdH.to_dataframe()
    
    
    #Regrid the 9th arc sec CUDEM data and overwetie ROMS grid bathymetry 
    
    for el in elements9:
      #   print(el)
        st = time.time()
        el = el.rename({"x": "lon", "y": "lat"})
        mnlon=el.lon.min().values
        mxlon=el.lon.max().values
        mnlat=el.lat.min().values
        mxlat=el.lat.max().values
        #print([mnlon, mxlon, mnlat, mxlat])
    
        df2=df.loc[(df['lon'] > mnlon) & (df['lon'] < mxlon) & (df['lat'] > mnlat) & (df['lat'] < mxlat)]
        print('Regridding  9th arc sec files')
        regridder = xe.Regridder(el, df2, "bilinear", locstream_out=True)
    
    
        y=regridder(el, keep_attrs=True)
    
        df.loc[(df['lon'] > mnlon) & (df['lon'] < mxlon) & (df['lat'] > mnlat) & (df['lat'] < mxlat),'h']=-y+datum_adjust
    #    get the end time
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
    
    
    #Regrid the 3rd arc sec CUDEM data and overwetie ROMS grid bathymetry 
    for el in elements3:
     #   print(el)
        st = time.time()
        el = el.rename({"x": "lon", "y": "lat"})
        mnlon=el.lon.min().values
        mxlon=el.lon.max().values
        mnlat=el.lat.min().values
        mxlat=el.lat.max().values
        #print([mnlon, mxlon, mnlat, mxlat])
    
        df2=df.loc[(df['lon'] > mnlon) & (df['lon'] < mxlon) & (df['lat'] > mnlat) & (df['lat'] < mxlat)]
        print('Regridding 3rd arc sec file')
        regridder = xe.Regridder(el, df2, "bilinear", locstream_out=True)
    
    
        y=regridder(el, keep_attrs=True)
    
        df.loc[(df['lon'] > mnlon) & (df['lon'] < mxlon) & (df['lat'] > mnlat) & (df['lat'] < mxlat),'h']=-y+datum_adjust
        # get the end time
        et = time.time()
        # get the execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')
        
    #Reshape gridded data    
    newH=df.to_xarray()
    newH=newH.set_coords(('lat','lon'))
    
    #Adjustmenst to bathymetry to accound for
    tmpH=newH['h'].values
    tmpH[tmpH<depthcut]=depthcut
    
    
    #wrtiting data put. 
    nc = netCDF4.Dataset(newgfile, "r+", format="NETCDF4")
    h=nc['h']
    h[:,:]=tmpH
    nc.close()
    
    print("REGRIDING FINISHED")

def smooth_h():
    print('SMOOTHING')
    grd = xr.open_dataset(newgfile,chunks={'eta_rho':900,'xi_rho':600}) 
    grd=grd.set_coords(('lat_rho','lon_rho'))
    grdH=grd.h.load()
    grdH = grdH.rename({"lon_rho": "lon", "lat_rho": "lat"})
    
    grdH_sm=grdH.copy()
    H=grdH[:,:].values
    ###########################################
    if filtertype==1:        
        print('BOXCAR')
        w = signal.windows.boxcar(BC_np)/BC_np
        H_sm = signal.sepfir2d(H, w, w)
    elif filtertype==2:
        print('Gaussian')
        H_sm=ndimage.gaussian_filter(H, G_sigma, order=G_order,mode=G_mode)
    else:
        print('No smoothing')
    ###########################################
    
    grdH_sm[:,:]=H_sm
    #Create new grid file
    shutil.copy(newgfile,smgfile)
    #wrtiting data put. 
    nc = netCDF4.Dataset(smgfile, "r+", format="NETCDF4")
    h=nc['h']
    hraw=nc['hraw']
    hraw[:,:]=h[:,:]
    h[:,:]=grdH_sm.values

    nc.close()


if __name__ == "__main__":
    print('Running')

    if filterflag==1:
        main()
        smooth_h()
    elif filterflag==2:
        main()
    elif filterflag==3:
        smooth_h()
        
        
    print('Finished')