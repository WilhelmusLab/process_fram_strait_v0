"""Use logistic regression to identify likely true and false positives using circularity, true color, and false color.
Circularity provides information on the floe geometry, while the color helps distinguish cloud and ice."""

import os
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from rasterio.plot import reshape_as_image
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops_table
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
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

def pixel_path_length(floe_df):
    """Calculates distance traversed in units of pixels"""
    delta_r = floe_df['row_pixel'] - floe_df['row_pixel'].shift(-1)
    delta_c = floe_df['col_pixel'] - floe_df['col_pixel'].shift(-1)
    return (np.sqrt(delta_r**2 + delta_c**2)).sum()

def estimated_mean_speed(floe_df):
    """Calculates distance traversed in units of pixels"""
    delta_x = floe_df['x_stere'] - floe_df['x_stere'].shift(-1)
    delta_y = floe_df['y_stere'] - floe_df['y_stere'].shift(-1)
    dt = (floe_df['datetime'].max() - floe_df['datetime'].min()).total_seconds()
    return np.round((np.sqrt(delta_x**2 + delta_y**2)).sum()/dt, 3)

#### Load the dataframes with pixel brightness
pb_dataloc = '../data/all_floes/'

ift_dfs = {}
for year in range(2003, 2021): # re-do to have all years once 03 finishes running
    ift_df = pd.read_csv(pb_dataloc + '/ift_floe_properties_with_pixel_brightness_{y}.csv'.format(y=year))
    ift_df['datetime'] = pd.to_datetime(ift_df['datetime'])
    ift_df['circularity'] = 4*np.pi*ift_df['area']/ift_df['perimeter']**2 # calculated in the new version of the brightness extraction code
    # Scale the pixel brightness data to 0-1
    for var in ['tc_channel0', 'tc_channel1', 'tc_channel2', 'fc_channel0', 'fc_channel1', 'fc_channel2']:
        ift_df[var] = ift_df[var]/255
    
    df_floes = ift_df.loc[ift_df.floe_id != 'unmatched']
    
    # Require a minimum amount of travel (2 pixels per image ~ 1 pixel per day)
    df_floes = df_floes.groupby('floe_id').filter(lambda x: pixel_path_length(x) > 2*len(x))
    
    # Average speed has to be less than 1 m/s
    # Calculated as path length divided by total elapsed time
    df_floes = df_floes.groupby('floe_id').filter(lambda x: estimated_mean_speed(x) < 1)
    
    # Remove SIC=0 and landmasked floes from TP dataset
    df_floes = df_floes.loc[(df_floes.nsidc_sic > 0) & (df_floes.nsidc_sic <= 1)]
    ift_df['classification'] = 'NA'
    ift_df.loc[ift_df.nsidc_sic == 0, 'classification'] = 'FP'
    ift_df.loc[df_floes.index, 'classification'] = 'TP'
    # Mark invalid data as NA. Circularity should always be less than 1 if the true area/perimeter are known, I use 1.2 as an 
    # upper limit to allow for some uncertainty in the calculation.
    ift_df.loc[ift_df.circularity > 1.2, 'classification'] = 'NA'
    ift_df.loc[ift_df.tc_channel0.isnull(), 'classification'] = 'NA'
    ift_dfs[year] = ift_df.copy()

#### Get random sample for training/testing
data_samples = []
for year in ift_dfs:
    for month, group in ift_dfs[year].groupby(ift_dfs[year].datetime.dt.month):
        if month != 3: # Only 1 day in March in any year, so we skip it. Only use full months.
            data_samples.append(group.loc[group.classification != 'NA'].groupby('classification').apply(lambda x: x.sample(min(len(x), 1000), replace=False)))

data = pd.concat(data_samples).reset_index(drop=True)

#### Train logistic regression model
minimal_variables = ['circularity', 'tc_channel0', 'fc_channel0']
# This are the variables that are not closely correlated with each other. Functions of other brightness channels could be useful.

X = data.loc[:, minimal_variables].to_numpy()
y =  (data.classification == 'TP').to_numpy()

# Split data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=1/3, random_state=10)

# Fit the logistic regression model using cross validation
lr_model = LogisticRegressionCV(Cs=10,
                     solver='lbfgs',
                     scoring='accuracy',
                     penalty='l2',
                     cv=10,
                     random_state=5).fit(X_train, y_train)

#### Compute and print skill scores for the model
y_pred = lr_model.predict(X_test)
print('F1 score: ', np.round(f1_score(y_test, y_pred), 3))
print('Recall: ', np.round(recall_score(y_test, y_pred), 3))
print('Precision: ', np.round(precision_score(y_test, y_pred), 3))
print('Confusion matrix:')
print(pd.DataFrame(data=np.round(confusion_matrix(y_test, y_pred)/len(y_test), 2),
             index=['True', 'False'], columns=['PredTrue', 'PredFalse']))

#### Add column with decision function and probability 
order = ['datetime','satellite',  'floe_id', 'label',
         'longitude', 'latitude',
         'x_stere', 'y_stere', 'col_pixel', 'row_pixel',
         'area', 'perimeter', 'solidity', 'orientation',
         'circularity', 'axis_major_length', 'axis_minor_length',
         'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col',
         'area_matlab', 'perimeter_matlab', 'solidity_matlab', 'orientation_matlab',
         'nsidc_sic',
         'tc_channel0', 'tc_channel1', 'tc_channel2',
         'fc_channel0', 'fc_channel1', 'fc_channel2', 'init_classification',
         'lr_probability', 'lr_classification']    

for year in ift_dfs:    
    idx_data = ift_dfs[year].tc_channel1.notnull() & (ift_dfs[year].circularity <= 1.2)
    probs = lr_model.predict_proba(ift_dfs[year].loc[idx_data, minimal_variables].to_numpy())
    p_notfloe, p_floe = probs[:,0], probs[:, 1]
    ift_dfs[year].loc[idx_data, 'lr_probability'] = np.round(p_floe, 3)
    ift_dfs[year].loc[idx_data, 'lr_classification'] = lr_model.predict(ift_dfs[year].loc[idx_data, minimal_variables].to_numpy())
    
    for var in ['tc_channel0', 'tc_channel1', 'tc_channel2', 'fc_channel0', 'fc_channel1', 'fc_channel2']:
        ift_dfs[year][var] = np.round(ift_dfs[year][var]*255, 1) # Re-scale to original brightness values
        
    # ift_dfs[year].to_csv('../data/all_floes/ift_floe_properties_LR_results_{y}.csv'.format(y=year))
    #### Round the numbers down so we aren't artificially increasing precision
    for var in ['x_stere', 'y_stere', 'area', 'perimeter',
                'circularity', 'axis_major_length', 'axis_minor_length',
                'bbox_min_row', 'bbox_min_col', 'bbox_max_row', 'bbox_max_col']:
        ift_dfs[year][var] = ift_dfs[year][var].round(1)

    ift_dfs[year]['latitude'] = ift_dfs[year]['latitude'].round(4)
    ift_dfs[year]['longitude'] = ift_dfs[year]['longitude'].round(4)

    print(year, ift_dfs[year].groupby('classification').count()['floe_id'])
    
    ift_dfs[year].rename({'classification': 'init_classification'}, axis=1, inplace=True)
    
    year_folder = 'fram_strait-{y}'.format(y=year)
    ift_dfs[year].loc[:, order].to_csv(dataloc + year_folder + '/ift_raw_floe_properties_{y}.csv'.format(y=year))

    # Keep the floes tagged as true positives initially as well as any others marked as TPs by the LR model
    idx_keep = (ift_dfs[year]['init_classification'] == 'TP') | ift_dfs[year].lr_classification
    ift_dfs[year].loc[idx_keep, order].dropna(subset='x_stere').to_csv(dataloc + year_folder + '/ift_clean_floe_properties_{y}.csv'.format(y=year))
