"""Convert the FLOE_LIBRARY matlab files into GeoTiffs matching the shape of the reference image.
"""
import numpy as np
import os
import pandas as pd
import rasterio as rio
from scipy.io import loadmat
from skimage.transform import resize

# Folder with MODIS imagery and where the GeoTiffs should be stored
dataloc = '/Volumes/Research/ENG_Wilhelmus_Shared/group/IFT_fram_strait_dataset/'

# Set the year to process
for year in range(2003, 2021):
    
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
    im_ref = rio.open('../data/NE_Greenland.2017100.terra.250m.tif')
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
    
    # TBD: test adjustments for the 2020 differences in coordinates
    for date_idx in info_df.index:
        # Grab the subset matching the date
        df = all_props.loc[all_props.date_idx == date_idx] 
    
        # Loop through the floe library and add images to array
        # Assign floe label using the original floe index
        count = 1
        segmented_image = np.zeros((nlayers, nrows, ncols))
        for row, data in df.iterrows():
            ri, ci = im_ref.index(data.x_stere, data.y_stere)
            orig_idx = int(data.orig_idx - 1)
            floe_image = floelib['FLOE_LIBRARY'][orig_idx, date_idx]
            
            if year == 2020:
                # Resize floe image to have correct pixel size
                info_region_pixel_scale_x = 200.298
                info_region_pixel_scale_y = 216.600
                new_size = (int(floe_image.shape[0]*info_region_pixel_scale_y / 256), 
                            int(floe_image.shape[1]*info_region_pixel_scale_x / 256))

                # Define zero padding
                dx = (floe_image.shape[0] - new_size[0])/2
                if dx == int(dx):
                    left_pad = int(dx)
                    right_pad = int(dx)
                else:
                    left_pad = int(np.floor(dx))
                    right_pad = int(np.ceil(dx))
                dy = (floe_image.shape[1] - new_size[1])/2
                if dy == int(dy):
                    top_pad = int(dy)
                    bottom_pad = int(dy)
                else:
                    top_pad = int(np.floor(dy))
                    bottom_pad = int(np.ceil(dy))
                floe_image = np.pad(resize(floe_image, new_size, order=0), ((left_pad, right_pad), (bottom_pad, top_pad)))

            
            floe_image = np.ma.masked_array(
                floe_image, floe_image==0)[::-1,:].copy()

            # Coordinates of top corner
            left_x = int(np.floor(data.bbox1))
            top_y = nrows - int(np.floor(data.bbox2))
            
            if year == 2020:     
                # Use centroid positions to shift into the
                # correct locations
                cx = data.x_pixel
                rx = data.y_pixel
                left_x = int(left_x - cx + ci)
                top_y = nrows - int(np.floor(data.bbox2) - rx + ri)
                top_y = int(top_y)

            # Coordinates of bottom corner found w/ height and width
            right_x = int(left_x + data.bbox3 + 1)
            bottom_y = int(top_y - data.bbox4 - 1)

            # Use the original index as the floe label
            if (ri > 0) & (ci > 0):
                if segmented_image[0, bottom_y:top_y, left_x:right_x].shape == floe_image.shape:
                    segmented_image[0, bottom_y:top_y, left_x:right_x] += floe_image * data.orig_idx
    
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
