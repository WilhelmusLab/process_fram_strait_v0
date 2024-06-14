"""Convert the FLOE_LIBRARY matlab files into GeoTiffs matching the shape of the reference image."""
import numpy as np
import os
import pandas as pd
import rasterio as rio
from scipy.io import loadmat


# Folder with MODIS imagery and where the GeoTiffs should be stored
saveloc = '../../data/floe_tracker/dataset/'

# Set the year to process
year = 2020

# Format for the year folders is fram_strait-YYYY
year_folder = 'fram_strait-{y}'.format(y=year)

# Name for the subfolder with the raw segmented image results
# Also used for the GeoTiff filenames
label = 'labeled_raw'


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

# Load reference image
im_ref = rio.open('../data/floe_tracker/unparsed/NE_Greenland.2017100.terra.250m.tif')
im = im_ref.read()
nlayers = 1
nrows = im.shape[1]
ncols = im.shape[2]

floelib = loadmat('../data/matlab_output/{y}/FLOE_LIBRARY.mat'.format(y=year))
all_props = pd.read_csv('../data/all_floes/ift_floe_properties_{y}.csv'.format(y=year))
info_df = pd.read_csv('../data/matlab_output/{y}/time_data.csv'.format(y=year), index_col=0)
info_df['SOIT time'] = pd.to_datetime(info_df['SOIT time'])
all_props['datetime'] = pd.to_datetime(all_props['datetime'])

# Add date index column to reference the MATLAB indexing
all_props['date_idx'] = -1
for date_idx in info_df.index:
    all_props.loc[all_props.datetime == info_df.loc[date_idx, 'SOIT time'], 'date_idx'] = date_idx

for date_idx in info_df.index:
    
    # Grab the subset matching the date
    df = all_props.loc[all_props.date_idx == date_idx] 

    # Loop through the floe library and add images to array
    # Assign floe label using the original floe index
    count = 1
    segmented_image = np.zeros((nlayers, nrows, ncols))
    for row, data in df.iterrows():
        # this is the centroid - just need to check that we're in the right area
        x = data.x_stere
        y = data.y_stere
        ri, ci = im_new.index(x, y)
        ri = r - ri
    
        floe_image = floelib['FLOE_LIBRARY'][data.orig_idx, date_idx]
        floe_image = np.ma.masked_array(floe_image, floe_image==0)[::-1,:].copy()
        left_x = int(np.floor(data.bbox1))
        right_x = int(left_x + data.bbox3 + 1)
        top_y = r - int(np.floor(data.bbox2))
        bottom_y = int(top_y - data.bbox4 - 1)
        segmented_image[0, bottom_y:top_y, left_x:right_x] += floe_image * data.orig_idx

    month_folder = get_month_folder(info_df.loc[date_idx, 'SOIT time'])
    
    fname = '{d}.{s}.{l}.250m.tiff'.format(d=info_df.loc[date_idx, 'SOIT time'].strftime('%Y%m%d'),
                                           s=info_df.loc[date_idx, 'satellite'],
                                           l=label)
    saveloc = os.path.join(dataloc, year_folder, month_folder, label)
    os.makedirs(saveloc, exist_ok=True)
    new_file = rio.open(saveloc + '/' + fname, 'w',
                        driver='GTiff',
                        height=im_new.meta['height'],
                        width=im_new.meta['width'],
                        count=1,
                        dtype='uint16',
                        crs=im_new.meta['crs'],
                        transform=im_new.meta['transform'])
    new_file.write(segmented_image[0,::-1,:], 1)
    new_file.close()