import os
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from rasterio.plot import reshape_as_image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops_table
import sys
import warnings
import xarray as xr

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

# Data is stored on the shared drive
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

for year in range(2013, 2014):
    ift_loc = '../data/all_floes/' # Output from the interpolate ift file
    #### Load ift data
    ift_df = pd.read_csv(ift_loc + 'ift_floe_properties_{y}.csv'.format(y=year), index_col=0)
    ift_df['datetime'] = pd.to_datetime(ift_df.datetime.values)
    dfs_with_pixel_data = []

    for date, group in ift_df.groupby('datetime'):
        year_folder = 'fram_strait-{y}'.format(y=date.year)
        lb_path = os.path.join(dataloc, year_folder,
                               get_month_folder(date),
                               'labeled_raw',
                               '.'.join([date.strftime('%Y%m%d'),
                                         group.satellite.values[0],
                                         'labeled_raw', '250m', 'tiff']))
        tc_path = os.path.join(dataloc, year_folder,
                               get_month_folder(date),
                               'truecolor',
                               '.'.join([date.strftime('%Y%m%d'),
                                         group.satellite.values[0],
                                         'truecolor', '250m', 'tiff']))
        fc_path = os.path.join(dataloc, year_folder, get_month_folder(date),
                               'falsecolor',
                               '.'.join([date.strftime('%Y%m%d'),
                                         group.satellite.values[0],
                                         'falsecolor', '250m', 'tiff']))
        with rio.open(lb_path) as im:
            lb_im = reshape_as_image(im.read())
        with rio.open(tc_path) as im:
            tc_im = reshape_as_image(im.read())
        with rio.open(fc_path) as im:
            fc_im = reshape_as_image(im.read())
            
        props_tc = pd.DataFrame(regionprops_table(lb_im[:, :, 0], tc_im,
                                                  properties=['label', 
                                                              'intensity_mean']))
        props_fc = pd.DataFrame(regionprops_table(lb_im[:, :, 0], fc_im,
                                                  properties=['label',
                                                              'intensity_mean']))
        props_tc.rename({'intensity_mean-0': 'tc_channel0',
                         'intensity_mean-1': 'tc_channel1',
                         'intensity_mean-2': 'tc_channel2'}, axis=1, inplace=True)
        props_fc.rename({'intensity_mean-0': 'fc_channel0',
                         'intensity_mean-1': 'fc_channel1',
                         'intensity_mean-2': 'fc_channel2'}, axis=1, inplace=True)
        
        props = props_tc.merge(props_fc, left_on='label', right_on='label')
        # props['circularity'] = 4*np.pi*props['area']/props['perimeter']**2
        dfs_with_pixel_data.append(group.merge(props, left_on='orig_idx', right_on='label'))
    
        del lb_im, tc_im, fc_im, props_tc, props_fc, props

    pd.concat(dfs_with_pixel_data).to_csv(
        '../data/all_floes/ift_floe_properties_pixel_brightness_{y}.csv'.format(y=year))
