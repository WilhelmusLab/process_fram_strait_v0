"""
Using the segmented GeoTiffs and the property tables, expand the 
property table with pixel brightness from the satellite imagery, 
re-calculate shape properties using scikit-image for consistency,
add sea ice concentration from the NSIDC climate data record, and
save the resulting CSV files to the repo and to the archive.
"""
import os
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from rasterio.plot import reshape_as_image
from skimage.measure import regionprops_table
import sys
import warnings
import xarray as xr

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter('ignore')

# Satellite image data is stored on the shared drive
image_loc = '/Volumes/Research/ENG_Wilhelmus_Shared/group/IFT_fram_strait_dataset/'

# Property tables are in this repo
props_loc = '../data/all_floes/'

# Sea ice concentration stored locally
sic_loc = '/Users/dwatkin2/Documents/research/data/nsidc_daily_cdr/'

# List of properties to recalculate from images
properties = ['area', 'perimeter', 'solidity', 'bbox', 'centroid', 'orientation',
              'axis_major_length', 'axis_minor_length']

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

def interp_sic(position_data, sic_data):
    """Uses the xarray advanced interpolation to get along-track sic
    via nearest neighbors. Nearest neighbors is preferred because numerical
    flags are used for coasts and open ocean, so interpolation is less meaningful."""
    # Sea ice concentration uses NSIDC NP Stereographic
    crs0 = pyproj.CRS('WGS84')
    crs1 = pyproj.CRS('+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs')
    transformer_stere = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    
    sic = pd.Series(data=np.nan, index=position_data.index)
    
    for date, group in position_data.groupby(position_data.datetime.dt.date):
        x_stere, y_stere = transformer_stere.transform(
            group.longitude, group.latitude)
        
        x = xr.DataArray(x_stere, dims="z")
        y = xr.DataArray(y_stere, dims="z")
        SIC = sic_data.sel(time=date.strftime('%Y-%m-%d'))['sea_ice_concentration'].interp(
            {'x': x,
             'y': y}, method='nearest').data

        sic.loc[group.index] = np.round(SIC.T, 3)
    return sic
    
for year in range(2005, 2021): # rerun 2004
    #### Load ift data
    ift_df = pd.read_csv(props_loc + 'ift_floe_properties_{y}.csv'.format(y=year), index_col=0)
    ift_df['datetime'] = pd.to_datetime(ift_df.datetime.values)

    # Add sea ice concentration column
    with xr.open_dataset(sic_loc + '/aggregate/seaice_conc_daily_nh_' + \
                     str(year) + '_v04r00.nc') as sic_data:
        ds = xr.Dataset({'sea_ice_concentration':
                         (('time', 'y', 'x'), sic_data['cdr_seaice_conc'].data)},
                           coords={'time': (('time', ), sic_data['time'].data),
                                   'x': (('x', ), sic_data['xgrid'].data), 
                                   'y': (('y', ), sic_data['ygrid'].data)})
    
        sic = interp_sic(ift_df, ds)
        ift_df['nsidc_sic'] = np.round(sic, 2)

    rename_vars = {v: v + '_matlab' for v in ift_df.columns if v in properties}
    
    # This might need to happen later
    ift_df.rename(rename_vars, axis=1, inplace=True)
    
    dfs_with_pixel_data = []

    for date, group in ift_df.groupby('datetime'):
        year_folder = 'fram_strait-{y}'.format(y=date.year)
        lb_path = os.path.join(image_loc, year_folder,
                               get_month_folder(date),
                               'labeled_raw',
                               '.'.join([date.strftime('%Y%m%d'),
                                         group.satellite.values[0],
                                         'labeled_raw', '250m', 'tiff']))
        tc_path = os.path.join(image_loc, year_folder,
                               get_month_folder(date),
                               'truecolor',
                               '.'.join([date.strftime('%Y%m%d'),
                                         group.satellite.values[0],
                                         'truecolor', '250m', 'tiff']))
        fc_path = os.path.join(image_loc, year_folder, get_month_folder(date),
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
                                                              'intensity_mean'] + properties))
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
        props['circularity'] = 4*np.pi*props['area']/props['perimeter']**2
        dfs_with_pixel_data.append(group.merge(props, left_on='orig_idx', right_on='label'))
    
        del lb_im, tc_im, fc_im, props_tc, props_fc, props

    data = pd.concat(dfs_with_pixel_data).reset_index(drop=True)

    # rename and reorder variables
    rename = {'centroid-0': 'row_pixel',
              'centroid-1': 'col_pixel',
              'major_axis': 'axis_major_length_matlab',
              'minor_axis': 'axis_minor_length_matlab',
              'bbox-0': 'bbox_min_row',
              'bbox-1': 'bbox_min_col',
              'bbox-2': 'bbox_max_row',
              'bbox-3': 'bbox_max_col'}
              
    order = ['datetime','satellite',  'floe_id', 'label',
             'longitude', 'latitude',
             'x_stere', 'y_stere', 'col_pixel', 'row_pixel',
             'area', 'perimeter', 'solidity', 'orientation',
             'circularity', 'axis_major_length', 'axis_minor_length',
             'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col',
             'area_matlab', 'perimeter_matlab', 'solidity_matlab', 'orientation_matlab',
             'nsidc_sic',
             'tc_channel0', 'tc_channel1', 'tc_channel2',
             'fc_channel0', 'fc_channel1', 'fc_channel2']    
    data.rename(rename, axis=1, inplace=True)
    data = data.sort_values('datetime')
    data.loc[:, order].to_csv(
        '../data/all_floes/ift_floe_properties_with_pixel_brightness_{y}.csv'.format(y=year))

    year_folder = 'fram_strait-{y}'.format(y=year)
    data.to_csv(image_loc + year_folder + '/ift_raw_floe_properties_{y}.csv'.format(y=year))