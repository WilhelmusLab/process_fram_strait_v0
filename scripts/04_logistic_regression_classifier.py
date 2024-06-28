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
    delta_x = floe_df['x_pixel'] - floe_df['x_pixel'].shift(-1)
    delta_y = floe_df['y_pixel'] - floe_df['y_pixel'].shift(-1)
    return (np.sqrt(delta_x**2 + delta_y**2)).sum()

def estimated_mean_speed(floe_df):
    """Calculates distance traversed in units of pixels"""
    delta_x = floe_df['x_stere'] - floe_df['x_stere'].shift(-1)
    delta_y = floe_df['y_stere'] - floe_df['y_stere'].shift(-1)
    dt = (floe_df['datetime'].max() - floe_df['datetime'].min()).total_seconds()
    return np.round((np.sqrt(delta_x**2 + delta_y**2)).sum()/dt, 3)

#### Load the dataframes with pixel brightness
dataloc = '../data/all_floes/'

ift_dfs = {}
for year in range(2010, 2020): # re-do to have all years once 03 finishes running
    ift_df = pd.read_csv('../data/all_floes/ift_floe_properties_pixel_brightness_{y}.csv'.format(y=year))
    ift_df['datetime'] = pd.to_datetime(ift_df['datetime'])
    ift_df['circularity'] = 4*np.pi*ift_df['area']/ift_df['perimeter']**2 # calculated in the new version of the brightness extraction code
    # Scale the pixel brightness data to 0-1
    for var in ['tc_channel0', 'tc_channel1', 'tc_channel2', 'fc_channel0', 'fc_channel1', 'fc_channel2']:
        ift_df[var] = ift_df[var]/255
    
    df_floes = ift_df.loc[ift_df.floe_id != 'unmatched']
    
    # Require a minimum amount of travel (2 pixels per image ~ 1 pixel per day)
    df_floes = df_floes.groupby('floe_id').filter(lambda x: pixel_path_length(x) > 2*len(x)) # Very simple filter, requires net movement of only 2 pixels per image
    
    # Average speed has to be less than 1 m/s
    # Calculated as path length divided by total elapsed time
    df_floes = df_floes.groupby('floe_id').filter(lambda x: estimated_mean_speed(x) < 1) # Simple filter - average speed can't be more than 1 m/s
    
    # Remove SIC=0 and landmasked floes from TP dataset
    df_floes = df_floes.loc[(df_floes.nsidc_sic > 0) & (df_floes.nsidc_sic <= 1)]
    ift_df['classification'] = 'NA'
    ift_df.loc[ift_df.nsidc_sic == 0, 'classification'] = 'FP'
    ift_df.loc[df_floes.index, 'classification'] = 'TP'
    ift_dfs[year] = ift_df

#### Get random sample for training/testing
data_samples = []
for year in ift_dfs:
    for month, group in ift_dfs[year].groupby(ift_dfs[year].datetime.dt.month):
        if month != 3: # Only 1 day in March in any year, so we skip it. Only use full months.
            data_samples.append(group.loc[group.classification != 'NA'].groupby('classification').apply(lambda x: x.sample(min(len(x), 1000), replace=False)))

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
for year in ift_dfs:
    probs = lr_model.predict_proba(ift_dfs[year].loc[:, minimal_variables].to_numpy())
    p_notfloe, p_floe = probs[:,0], probs[:, 1]
    ift_dfs[year]['lr_probability'] = p_floe
    ift_dfs[year]['lr_classification'] = lr_model.predict(ift_dfs[year].loc[:, minimal_variables].to_numpy())
    
    for var in ['tc_channel0', 'tc_channel1', 'tc_channel2', 'fc_channel0', 'fc_channel1', 'fc_channel2']:
        ift_dfs[year][var] = ift_dfs[year][var]*255 # Re-scale to original brightness values
        
    ift_dfs[year].to_csv('../data/all_floes/ift_floe_properties_LR_results_{y}.csv'.format(y=year))




