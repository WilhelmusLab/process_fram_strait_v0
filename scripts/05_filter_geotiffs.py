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
    print(year)
    # Format for the year folders is fram_strait-YYYY
    year_folder = 'fram_strait-{y}'.format(y=year)
    
    # Name for the subfolder with the raw segmented image results
    # Also used for the GeoTiff filenames
    label_raw = 'labeled_raw'
    label_clean = 'labeled_clean'

    filename = 'ift_clean_floe_properties_{y}.csv'.format(y=year)
    info_df = pd.read_csv('../data/matlab_output/{y}/time_data.csv'.format(y=year), index_col=0)
    info_df['SOIT time'] = pd.to_datetime(info_df['SOIT time'])

    all_props = pd.read_csv(os.path.join(dataloc, year_folder, filename), index_col=0)
    all_props.fillna({'init_classification': 'UK'}, inplace=True)
    all_props['datetime'] = pd.to_datetime(all_props['datetime'].values)
    all_props['date_idx'] = -1
    for date_idx in info_df.index:
        all_props.loc[all_props.datetime == info_df.loc[date_idx, 'SOIT time'], 'date_idx'] = date_idx
    

    for date_idx in info_df.index[1:]:
        # Grab the subset matching the date
        df = all_props.loc[all_props.date_idx == date_idx]
        month_folder = get_month_folder(info_df.loc[date_idx, 'SOIT time'])
        
        fname_raw = '{d}.{s}.{l}.250m.tiff'.format(d=info_df.loc[date_idx, 'SOIT time'].strftime('%Y%m%d'),
                                               s=info_df.loc[date_idx, 'satellite'],
                                               l=label_raw)
        fname_clean = fname_raw.replace('raw', 'clean')

        with rio.open(os.path.join(dataloc, year_folder, month_folder, label_raw, fname_raw)) as im_raw:
            data = im_raw.read()
            
            # Set any label that is not present in the cleaned property table to 0.
            for ii in range(1, data.max() + 1):
                if ii not in df['label'].values:
                    data[data == ii] = 0

            # Create a folder for the clean images if one doesn't exist
            saveloc = os.path.join(dataloc, year_folder, month_folder, label_clean)
            os.makedirs(saveloc, exist_ok=True)
    
            # Use rasterIO to save the array as a geotiff with the same georeferencing as the original image
            new_file = rio.open(saveloc + '/' + fname_clean, 'w',
                                driver='GTiff',
                                height=im_raw.meta['height'],
                                width=im_raw.meta['width'],
                                count=1,
                                dtype='uint16',
                                crs=im_raw.meta['crs'],
                                transform=im_raw.meta['transform'])
            new_file.write(data.squeeze(), 1)
            new_file.close()
