"""Produce a set of floe trajectories with daily resolution with rotation rates included"""

"""Reads raw IFT csv files from data/floe_tracker/parsed/, resamples to daily resolution,
and adds ERA5 winds. Results for individual years are saved in data/floe_tracker/interpolated
and a single dataframe with wind speed, drift speed ratio, and turning angle for all
the years are saved in data/floe_tracker.

The full files are saved within the floe_tracker/interpolated files. For the analysis, we
apply a filter requiring the ice drift speed to be at least 0.02 m/s and less than 1.5 m/s.
The filtered data is saved as a single CSV file in data/floe_tracker/ft_with_wind.csv.
"""

import numpy as np
import pandas as pd
import pyproj 
import os
from metpy.units import units
import metpy.calc as mcalc
import xarray as xr
import sys
from scipy.interpolate import interp1d
import warnings
 
# Note: Behavior of idxmin in the presence of nans will change in next version
warnings.simplefilter(action='ignore', category=FutureWarning)

saveloc = '../data/floe_tracker/interpolated/'
saveloc_single = '../data/floe_tracker/'

dataloc = '/Volumes/Research/ENG_Wilhelmus_Shared/group/IFT_fram_strait_dataset/'

interp_variables = ['longitude', 'latitude', 'x_stere', 'y_stere',
                    'area', 'perimeter', 'axis_major_length', 'axis_minor_length',
                    'solidity', 'circularity', 'nsidc_sic', 'lr_probability']

# Location of folder with ERA5 data. I saved the ERA5 data with 
# a file structure of era5_dataloc/YYYY/era5_uvmsl_daily_mean_YYYY-MM-01.nc
# era5_dataloc = '../external_data/era5_daily/'
def regrid_floe_tracker(group, datetime_grid, variables=['x_stere', 'y_stere']):
    """Estimate the location at 12 UTC using linear interpolation.
    Group should have datetime index. """
    group = group.sort_index()
    begin = group.index.min()
    end = group.index.max()

    if len(datetime_grid.loc[slice(begin, end)]) > 1:
        t0 = group.index.round('12h').min()
        t1 = group.index.round('12h').max()
        max_extrap = pd.to_timedelta('2h')
        if np.abs(t0 - begin) < max_extrap:
            begin = t0
        if np.abs(t1 - end) < max_extrap:
            end = t1

        X = group[variables].T #.rolling(
            #'1H', center=True).mean().T.values
        t_new = datetime_grid.loc[slice(begin, end)].values
        t_seconds = group['t'].values
        Xnew = interp1d(t_seconds, X,
                        bounds_error=False,
                        kind='linear', fill_value='extrapolate')(t_new)
        idx = ~np.isnan(Xnew.sum(axis=0))

        df_new = pd.DataFrame(data=np.round(Xnew.T, 5), 
                              columns=variables,
                              index=datetime_grid.loc[slice(begin, end)].index)
        return df_new
    
    else:
        df = pd.DataFrame(data = np.nan, columns=group.columns, index=[begin])
        df.drop(['floe_id', 't'], axis=1, inplace=True)
        return df

def estimate_theta(df):
    """Estimate the angle change between each satellite from the orientation
    column from the properties matrix. Units are degrees."""
    df = df.copy()
    df['theta_terra_est'] = np.nan
    df['theta_aqua_est'] = np.nan
    df_t = df.loc[df.satellite=='terra']
    df_a = df.loc[df.satellite=='aqua']
    
    t_theta_est1 = df_t.orientation.shift(-1) - df_t.orientation
    t_theta_est2 = (df_t.orientation % 180).shift(-1) - df_t.orientation
    t_theta_est3 = df_t.orientation.shift(-1) - (df_t.orientation  % 180)
    comp = pd.concat({'t1': t_theta_est1, 't2': t_theta_est2, 't3': t_theta_est3}, axis=1)
    if len(comp) > 0:
        idx = np.abs(comp).idxmin(axis=1, skipna=True).fillna('t1')
        df.loc[comp.index, 'theta_terra_est'] = pd.Series(
            [comp.loc[x, y] for x, y in zip(idx.index, idx.values)], index=comp.index)
    
    a_theta_est1 = df_a.orientation.shift(-1) - df_a.orientation
    a_theta_est2 = (df_a.orientation % 180).shift(-1) - df_a.orientation
    a_theta_est3 = df_a.orientation.shift(-1) - (df_a.orientation  % 180)
    comp = pd.concat({'t1': a_theta_est1, 't2': a_theta_est2, 't3': a_theta_est3}, axis=1)
    if len(comp) > 0:
        idx = np.abs(comp).idxmin(axis=1).fillna('t1')
        df.loc[comp.index, 'theta_aqua_est'] = pd.Series(
            [comp.loc[x, y] for x, y in zip(idx.index, idx.values)], index=comp.index)

    return df
    
def get_daily_angle_estimate(df, interp_df, max_diff):
    """Takes the angle differences from IFT (theta) and from differences in 
    the properties matrix (theta_est), then estimates the angle for the daily
    grid. The angles are converted to zeta by dividing by the time between 
    satellite images. Returns the merged dataset with zeta in units of radians per day.
    """
    
    df['delta_time'] = np.nan
    for sat in ['aqua', 'terra']:
        df_sat = df.loc[df.satellite==sat]
        date = pd.Series(df_sat.index.values, index=df_sat.index)
        dt = date.shift(-1) - date
        df.loc[df_sat.index, 'delta_time'] = dt.dt.total_seconds() / (60*60*24) # Report rates as per day
    
    df['zeta_aqua'] = np.deg2rad(df['theta_aqua']) / df['delta_time']
    df['zeta_terra'] = np.deg2rad(df['theta_terra']) / df['delta_time']
    df['zeta_aqua_est'] = np.deg2rad(df['theta_aqua_est']) / df['delta_time']
    df['zeta_terra_est'] = np.deg2rad(df['theta_terra_est']) / df['delta_time']
    
    # Add the 
    df_aqua = df.loc[df.satellite=='aqua', ['zeta_aqua', 'zeta_aqua_est']].merge(
        interp_df['x_stere'], left_index=True, right_index=True, how='outer').drop('x_stere', axis=1)
    df_terra = df.loc[df.satellite=='terra', ['zeta_terra', 'zeta_terra_est']].merge(
        interp_df['x_stere'], left_index=True, right_index=True, how='outer').drop('x_stere', axis=1)
    df_aqua = df_aqua.interpolate(method='time', limit=1, limit_direction='both').loc[interp_df.index]
    df_terra = df_terra.interpolate(method='time', limit=1, limit_direction='both').loc[interp_df.index]
    df_angles = df_aqua.merge(df_terra, left_index=True, right_index=True)
    
    
    # Fill if one satellite is missing
    df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_aqua'] = df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_terra']
    df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_aqua_est'] = df_angles.loc[df_angles.zeta_aqua.isnull(), 'zeta_terra_est']
    df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_terra'] = df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_aqua']
    df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_terra_est'] = df_angles.loc[df_angles.zeta_terra.isnull(), 'zeta_aqua_est']
    
    df_angles['zeta_diff'] = np.abs(df_angles['zeta_aqua'] - df_angles['zeta_terra'])
    df_angles['zeta_diff_est'] = df_angles['zeta_aqua_est'] - df_angles['zeta_terra_est']
    df_angles['zeta'] = df_angles[['zeta_aqua', 'zeta_terra']].mean(axis=1).where(df_angles['zeta_diff'] < np.deg2rad(max_diff))
    df_angles['zeta_est'] = df_angles[['zeta_aqua_est', 'zeta_terra_est']].mean(axis=1).where(df_angles['zeta_diff_est'] < np.deg2rad(max_diff))
    
    return interp_df.merge(df_angles[['zeta', 'zeta_est']], left_index=True, right_index=True)

def compute_velocity(buoy_df, date_index=True, rotate_uv=False, method='c', xvar='x_stere', yvar='y_stere'):
    """Computes trajectory velocity and (optional) rotates into north and east directions.
    If x and y are not in the columns, projects lat/lon onto stereographic x/y prior
    to calculating velocity. Rotate_uv moves the velocity into east/west. Velocity
    calculations are done on the provided time index. Results will not necessarily 
    be reliable if the time index is irregular. With centered differences, values
    near endpoints are calculated as forward or backward differences.
    
    Options for method
    forward (f): forward difference, one time step
    backward (b): backward difference, one time step
    centered (c): 3-point centered difference
    forward_backward (fb): minimum of the forward and backward differences
    """
    buoy_df = buoy_df.copy()

    if isinstance(date_index, bool):
        if date_index:
            date = pd.Series(pd.to_datetime(buoy_df.index.values), index=pd.to_datetime(buoy_df.index))
    else:
        date = pd.to_datetime(buoy_df[date_index])
        
    delta_t_next = date.shift(-1) - date
    delta_t_prior = date - date.shift(1)
    min_dt = pd.DataFrame({'dtp': delta_t_prior, 'dtn': delta_t_next}).min(axis=1)

    # bwd endpoint means the next expected obs is missing: last data before gap
    bwd_endpoint = (delta_t_prior < delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    fwd_endpoint = (delta_t_prior > delta_t_next) & (np.abs(delta_t_prior - delta_t_next) > 2*min_dt)
    
    if xvar not in buoy_df.columns:
        print('Calculating position with polar stereographic coordinates')
        projIn = 'epsg:4326' # WGS 84 Ellipsoid
        projOut = 'epsg:3413' # NSIDC North Polar Stereographic
        transformer = pyproj.Transformer.from_crs(projIn, projOut, always_xy=True)

        lon = buoy_df.longitude.values
        lat = buoy_df.latitude.values

        x, y = transformer.transform(lon, lat)
        buoy_df[xvar] = x
        buoy_df[yvar] = y
    
    if method in ['f', 'forward']:
        dt = (date.shift(-1) - date).dt.total_seconds().values
        dxdt = (buoy_df[xvar].shift(-1) - buoy_df[xvar])/dt
        dydt = (buoy_df[yvar].shift(-1) - buoy_df[yvar])/dt

    elif method in ['b', 'backward']:
        dt = (date - date.shift(1)).dt.total_seconds()
        dxdt = (buoy_df[xvar] - buoy_df[xvar].shift(1))/dt
        dydt = (buoy_df[yvar] - buoy_df[yvar].shift(1))/dt

    elif method in ['c', 'fb', 'centered', 'forward_backward']:
        fwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='forward')
        bwd_df = compute_velocity(buoy_df.copy(), date_index=date_index, method='backward')

        fwd_dxdt, fwd_dydt = fwd_df['u'], fwd_df['v']
        bwd_dxdt, bwd_dydt = bwd_df['u'], bwd_df['v']
        
        if method in ['c', 'centered']:
            dt = (date.shift(-1) - date.shift(1)).dt.total_seconds()
            dxdt = (buoy_df[xvar].shift(-1) - buoy_df[xvar].shift(1))/dt
            dydt = (buoy_df[yvar].shift(-1) - buoy_df[yvar].shift(1))/dt

        else:
            dxdt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dxdt, 'b':bwd_dxdt})).min(axis=1)
            dydt = np.sign(bwd_dxdt)*np.abs(pd.DataFrame({'f': fwd_dydt, 'b':bwd_dydt})).min(axis=1)

        dxdt.loc[fwd_endpoint] = fwd_dxdt.loc[fwd_endpoint]
        dxdt.loc[bwd_endpoint] = bwd_dxdt.loc[bwd_endpoint]
        dydt.loc[fwd_endpoint] = fwd_dydt.loc[fwd_endpoint]
        dydt.loc[bwd_endpoint] = bwd_dydt.loc[bwd_endpoint]
    
    if rotate_uv:
        # Unit vectors
        buoy_df['Nx'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * -buoy_df[xvar]
        buoy_df['Ny'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * -buoy_df[yvar]
        buoy_df['Ex'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * -buoy_df[yvar]
        buoy_df['Ey'] = 1/np.sqrt(buoy_df[xvar]**2 + buoy_df[yvar]**2) * buoy_df[xvar]

        buoy_df['u'] = buoy_df['Ex'] * dxdt + buoy_df['Ey'] * dydt
        buoy_df['v'] = buoy_df['Nx'] * dxdt + buoy_df['Ny'] * dydt

        # Calculate angle, then change to 360
        heading = np.degrees(np.angle(buoy_df.u.values + 1j*buoy_df.v.values))
        heading = (heading + 360) % 360
        
        # Shift to direction from north instead of direction from east
        heading = 90 - heading
        heading = (heading + 360) % 360
        buoy_df['bearing'] = heading
        buoy_df['speed'] = np.sqrt(buoy_df['u']**2 + buoy_df['v']**2)
        buoy_df.drop(['Nx', 'Ny', 'Ex', 'Ey'], axis=1, inplace=True)
        
    else:
        buoy_df['u'] = dxdt
        buoy_df['v'] = dydt            
        buoy_df['speed'] = np.sqrt(buoy_df['v']**2 + buoy_df['u']**2)    

    return buoy_df
        
####### Apply the functions to the IFT parsed data #########
for year in range(2003, 2021):
    
    # Format for the year folders is fram_strait-YYYY
    year_folder = 'fram_strait-{y}'.format(y=year)
    
    # Name for the subfolder with the raw segmented image results
    # Also used for the GeoTiff filenames
    filename = 'ift_clean_floe_properties_{y}.csv'.format(y=year)
    
    df = pd.read_csv(os.path.join(dataloc, year_folder, filename), index_col=0)
    df['datetime'] = pd.to_datetime(df['datetime'].values).round('1min')
    
    df = df.loc[df.floe_id != 'unmatched'].copy()
    # Convert dates to elapsed time
    ref_time = pd.to_datetime(str(year) + '-01-01 00:00')
    df['t'] = (df['datetime'] - ref_time).dt.total_seconds()

    # Set up a regular grid for interpolation
    date_grid = pd.date_range(str(year) + '-04-01 00:00', str(year) + '-09-30 00:00', freq='1D')
    date_grid += pd.to_timedelta('12h') # Satellite overpass time is close to noon UTC
    t_grid = (date_grid - ref_time).total_seconds()
    datetime_grid = pd.Series(t_grid, index=date_grid)

    # Apply interpolation and angle estimation
    results = {}
    for floe_id, group in df.groupby('floe_id'):
        group = group.loc[~group.datetime.duplicated()].copy()
        df_orig = estimate_theta(group.set_index('datetime'))
        df_regrid = regrid_floe_tracker(group.set_index('datetime'), datetime_grid=datetime_grid, variables=interp_variables)
        if np.any(df_regrid.notnull()):
            df_interp = get_daily_angle_estimate(df_orig, df_regrid, max_diff=30)            
            results[floe_id] = df_interp.copy()
            del df_interp
        del df_orig, df_regrid

    interp_ft_df = pd.concat(results)
    interp_ft_df.index.names = ['floe_id', 'datetime']
    interp_ft_df.reset_index(inplace=True)
    interp_ft_df = interp_ft_df.loc[:, ['datetime', 'floe_id'] + interp_variables + ['zeta', 'zeta_est']]
    # Estimate velocity
    interp_ft_df['u'] = np.nan
    interp_ft_df['v'] = np.nan
    interp_ft_df['bearing'] = np.nan
    interp_ft_df['speed'] = np.nan
    
    for floe_id, group in interp_ft_df.groupby('floe_id'):
        if len(group) > 2:
            vel_df = compute_velocity(group, date_index='datetime', method='f', rotate_uv=True)
            for var in ['u', 'v', 'bearing', 'speed']:
                interp_ft_df.loc[vel_df.index, var] = vel_df[var]

    # Change to kilometers
    interp_ft_df['area_km2'] = (interp_ft_df['area']*0.25**2).round(2)
    interp_ft_df['perimeter_km'] = (interp_ft_df['perimeter']*0.25).round(2)
    interp_ft_df['axis_major_length_km'] = (interp_ft_df['axis_major_length']*0.25).round(2)
    interp_ft_df['axis_minor_length_km'] = (interp_ft_df['axis_minor_length']*0.25).round(2)
    
    interp_ft_df.to_csv(os.path.join(dataloc, year_folder, 'ift_interp_floe_trajectories_{y}.csv'.format(y=year)))



