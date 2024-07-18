"""This function will use the results of 04 and 02, reading the GeoTiff images with floe shapes and making a copy that only includes the floe shapes that were classified as likely floes by the logistic regression function OR were manually assigned as true positives based on tracking."""

import numpy as np
import os
import pandas as pd
import rasterio as rio
from scipy.io import loadmat
from skimage.transform import resize

# Folder with MODIS imagery and where the GeoTiffs should be stored
dataloc = '/Volumes/Research/ENG_Wilhelmus_Shared/group/IFT_fram_strait_dataset/'

def get_month_folder(date):
    """Simple utility for navigating file structure"""
    start = '{y}{m}01'.format(y=date.year,
                              m=str(date.month).zfill(2))
    end = '{y}{m}01'.format(y=date.year,
                              m=str(date.month + 1).zfill(2))
    if (date.month == 3) | (date.month == 4):
        start = '{y}0331'.format(y=date.year)
        end = '{y}0501'.format(y=date.year)
    
    return '-'.join(['fram_strait', start, end])

for year in range(2003, 2021):
    
    # Format for the year folders is fram_strait-YYYY
    year_folder = 'fram_strait-{y}'.format(y=year)
    
    # Name for the subfolder with the raw segmented image results
    # Also used for the GeoTiff filenames
    raw_label = 'labeled_raw'
    clean_label = 'labeled_clean'
    
    
# load the labeled image
    
# load the properties table
    
# set shapes marked as false to 0
    
# save image
        month_folder = get_month_folder(info_df.loc[date_idx, 'SOIT time'])
        
        fname = '{d}.{s}.{l}.250m.tiff'.format(d=info_df.loc[date_idx, 'SOIT time'].strftime('%Y%m%d'),
                                               s=info_df.loc[date_idx, 'satellite'],
                                               l=label)
        saveloc = os.path.join(dataloc, year_folder, month_folder, label)
        os.makedirs(saveloc, exist_ok=True)

        # Use rasterIO to save the array as a geotiff with the same georeferencing as the original image
        new_file = rio.open(saveloc + '/' + fname, 'w',
                            driver='GTiff',
                            height=im_ref.meta['height'],
                            width=im_ref.meta['width'],
                            count=1,
                            dtype='uint16',
                            crs=im_ref.meta['crs'],
                            transform=im_ref.meta['transform'])
        new_file.write(segmented_image[0,::-1,:], 1)
        new_file.close()