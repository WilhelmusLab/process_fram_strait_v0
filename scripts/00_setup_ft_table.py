"""Builds CSV tables with the bounding box, start, and end times for each year."""
import pandas as pd
import pyproj
import numpy as np
import rasterio



# bounding box
# from reference image
im = rasterio.open("../data/NE_Greenland.2017100.terra.250m.tif")
left, bottom, right, top = im.bounds

def make_specification_table(name, left_x, right_x, bottom_y, top_y, startdate, enddate):
    """Given bounding box, start/end dates, and timestep, generate a table with
    a row for each case of length timestep between the start and end dates with
    all the information needed for running IFT. Each row is assigned an identifier
    with the format name-startdate-enddate"""
    start = pd.to_datetime(startdate)
    end = pd.to_datetime(enddate)

    dates = pd.date_range(start, end + pd.DateOffset(months=1), freq='MS')
    start_dates = dates[:-1]
    end_dates = dates[1:]
    
    crs0 = pyproj.CRS('epsg:3413')
    crs1 = pyproj.CRS('WGS84')
    transformer_ll = pyproj.Transformer.from_crs(crs0, crs_to=crs1, always_xy=True)
    
    # get centroid location for soit
    xc = 0.5*(left_x + right_x)
    yc = 0.5*(bottom_y + top_y)
    center_lon, center_lat = transformer_ll.transform(xc, yc)
    
    # convert boundaries to lat/lon        
    top_left_lon, top_left_lat = transformer_ll.transform(left_x, top_y)
    lower_right_lon, lower_right_lat = transformer_ll.transform(right_x, bottom_y)


    # assemble into table
    table = pd.DataFrame({
        'center_lon': np.round(center_lon, 4),
        'center_lat': np.round(center_lat, 4),
        'top_left_lon': np.round(top_left_lon, 4),
        'top_left_lat': np.round(top_left_lat, 4),
        'lower_right_lon': np.round(lower_right_lon, 4),
        'lower_right_lat': np.round(lower_right_lat, 4),
        'left_x': left_x,
        'right_x': right_x,
        'lower_y': bottom_y,
        'top_y': top_y,
        'startdate': start_dates,
        'enddate': end_dates})

    # set the first date to the start date in case it's before the first of the month
    table.loc[table.index[0], 'startdate'] = min(start_dates.min(), start)
    
    # set the last date to the specified end date
    table.loc[table.index[-1], 'enddate'] = min(end_dates.max(), end)
    
    names = ['-'.join([name, start.strftime('%Y%m%d'), end.strftime('%Y%m%d')]) for
                      x, (start, end) in enumerate(zip(table.startdate, table.enddate))]

    table['location'] = names

    order = ['location', 'center_lat', 'center_lon', 'top_left_lat', 'top_left_lon',
             'lower_right_lat', 'lower_right_lon', 'left_x', 'right_x', 'lower_y',
             'top_y', 'startdate', 'enddate']
    
    return table.loc[:, order]

for year in range(2003, 2023):
    startdate = '{y}-03-31'.format(y=year) # start date inclusive
    enddate = '{y}-10-01'.format(y=year) # end date exclusive
    table = make_specification_table('fram_strait', left, right, bottom, top, startdate, enddate)
    table = table.iloc[:-1,:]
    table.to_csv('../data/modis_download_spec_files/location_specs_{y}.csv'.format(y=year))