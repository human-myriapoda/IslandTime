"""
This module allows us to match time frames and frequency (no NaN values and no extrapolation).
TODO: find time frames with the most time series
TODO: characterise time series (?)

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
from scipy import stats
from pandas.tseries.offsets import MonthEnd
import matplotlib
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1000'

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class PreProcessing:
    def __init__(self, island_info, plot_confounders=False, frequency='monthly', overwrite=False, threshold_zscore=2., window_rolling=6, min_periods_rolling=4):
        self.island_info = island_info
        self.plot_confounders = plot_confounders
        self.frequency = frequency
        self.island_info_path = os.path.join(os.getcwd(), 'data', 'info_islands')
        self.island = self.island_info['general_info']['island']
        self.country = self.island_info['general_info']['country']
        self.overwrite = overwrite
        self.threshold_zscore = threshold_zscore
        self.window_rolling = window_rolling
        self.min_periods_rolling = min_periods_rolling

    def retrieve_available_timeseries(self):

        print('Retrieving available time series')

        list_timeseries = [self.island_info[key]['timeseries'] for key in self.island_info.keys()
                           if 'timeseries' in self.island_info[key]
                           and key not in ['timeseries_WHO', 'timeseries_WorldBank', 'timeseries_disasters',
                                           'timeseries_climate_indices', 'timeseries_nighttime_light',
                                           'timeseries_coastsat', 'timeseries_preprocessing',
                                           'timeseries_analysis']
                           and self.island_info[key]['timeseries'] is not None]

        list_timeseries_socioeconomics = [self.island_info[key]['timeseries'] for key in self.island_info.keys()
                                          if 'timeseries' in self.island_info[key]
                                          and key in ['timeseries_WHO', 'timeseries_WorldBank',
                                                      'timeseries_disasters', 'timeseries_nighttime_light']
                                          and self.island_info[key]['timeseries'] is not None]

        list_confounders = [self.island_info[key]['confounders'] for key in self.island_info.keys()
                            if 'confounders' in self.island_info[key]]

        list_timeseries_climate_indices = [self.island_info[key]['timeseries'] for key in self.island_info.keys()
                                           if 'timeseries' in self.island_info[key]
                                           and key == 'timeseries_climate_indices'
                                           and self.island_info[key]['timeseries'] is not None]

        return list_timeseries, list_timeseries_socioeconomics, list_confounders, list_timeseries_climate_indices

    def utc_datetime_index_and_concat(self, list_timeseries):
        """
        Make sure all time series share the same UTC datetime index.
        """
        
        for timeseries in list_timeseries:
            if timeseries.empty:
                continue
            
            if timeseries.index.tzinfo is None:
                timeseries.index = [pytz.utc.localize(timeseries.index[i]) for i in range(len(timeseries.index))]
        
        # Combine them in one DataFrame
        df_timeseries = pd.concat(list_timeseries, axis=0)
        df_timeseries = df_timeseries.apply(pd.to_numeric)
        
        return df_timeseries

    def replace_outliers_with_nan(self, df_timeseries):

        print('Replacing outliers with NaN using z-score (abnormally high values)')
        
        # Create a copy of the DataFrame
        df_timeseries_remove_outliers = df_timeseries.copy() 

        # Calculate the z-score for each columns
        log_df_timeseries_remove_outliers = np.log(np.abs(df_timeseries_remove_outliers))
        z_scores = stats.zscore(log_df_timeseries_remove_outliers, nan_policy='omit')

        # Outlier mask
        outliers_mask = abs(z_scores) > self.threshold_zscore

        # Replace outliers with NaN for each column
        df_timeseries_remove_outliers[outliers_mask] = np.nan

        return df_timeseries_remove_outliers

    def group_by_frequency(self, df_timeseries_remove_outliers):

        print('Grouping by frequency')

        # Grouper using frequency (e.g. 'M' for monthly)
        df_timeseries_frequency = df_timeseries_remove_outliers.groupby(pd.Grouper(freq=self.frequency[0].capitalize())).mean()

        return df_timeseries_frequency

    def fill_nan_with_rolling_mean(self, df_timeseries_frequency):

        print('Filling NaN with rolling mean (only for time series with very few NaN)')

        # Fill NaN with mean (only for time series with very few NaN)
        df_timeseries_frequency_rolling = df_timeseries_frequency.fillna(df_timeseries_frequency.rolling(window=self.window_rolling, 
                                                                                                         min_periods=self.min_periods_rolling, 
                                                                                                         center=True).mean())

        return df_timeseries_frequency_rolling

    def get_coastline_and_other_timeseries(self, df_timeseries_frequency_rolling):

        print('Retrieving coastline and other time series')

        # Retrieve indices of all time series with 'coastline_position' in their name
        idx_coastline_timeseries = np.argwhere([np.char.startswith(list(df_timeseries_frequency_rolling.columns)[i],
                                                                    'coastline_position')
                                                for i in range(len(df_timeseries_frequency_rolling.columns))]).flatten()
        
        # Create a DataFrame with all coastline time series
        df_coastline_timeseries = df_timeseries_frequency_rolling[df_timeseries_frequency_rolling.columns[
            idx_coastline_timeseries]]
        
        # Retrieve indices for all other time series
        idx_other_timeseries = np.argwhere(
            [~np.char.startswith(list(df_timeseries_frequency_rolling.columns)[i], 'coastline_position')
             for i in range(len(df_timeseries_frequency_rolling.columns))]).flatten()
        
        # Create a DataFrame with all other time series
        df_other_timeseries = df_timeseries_frequency_rolling[df_timeseries_frequency_rolling.columns[idx_other_timeseries]]
        
        return df_coastline_timeseries, df_other_timeseries

    def _get_two_month_period(self, date):
        month = date.month
        year = date.year

        if month in [1, 2]:
            return datetime.datetime(year=year, month=1, day=30).replace(tzinfo=pytz.UTC)#f"{year}-Jan/Feb"
        elif month in [3, 4]:
            return datetime.datetime(year=year, month=3, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Mar/Apr"
        elif month in [5, 6]:
            return datetime.datetime(year=year, month=5, day=30).replace(tzinfo=pytz.UTC) #f"{year}-May/Jun"
        elif month in [7, 8]:
            return datetime.datetime(year=year, month=7, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Jul/Aug"
        elif month in [9, 10]:
            return datetime.datetime(year=year, month=9, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Sep/Oct"
        else:
            return datetime.datetime(year=year, month=11, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Nov/Dec"

    def _get_three_month_period(self, date):
        month = date.month
        year = date.year

        if month in [1, 2, 3]:
            return datetime.datetime(year=year, month=2, day=15).replace(tzinfo=pytz.UTC) #f"{year}-Jan/Feb/Mar"
        elif month in [4, 5, 6]:
            return datetime.datetime(year=year, month=5, day=15).replace(tzinfo=pytz.UTC) #f"{year}-Apr/May/Jun"
        elif month in [7, 8, 9]:
            return datetime.datetime(year=year, month=8, day=15).replace(tzinfo=pytz.UTC) #f"{year}-Jul/Aug/Sep"
        elif month in [10, 11, 12]:
            return datetime.datetime(year=year, month=11, day=15).replace(tzinfo=pytz.UTC) #f"{year}-Oct/Nov/Dec"
        
    def _get_four_month_period(self, date):
        month = date.month
        year = date.year

        if month in [1, 2, 3, 4]:
            return datetime.datetime(year=year, month=2, day=28).replace(tzinfo=pytz.UTC) #f"{year}-Jan/Feb/Mar/Apr"
        elif month in [5, 6, 7, 8]:
            return datetime.datetime(year=year, month=6, day=30).replace(tzinfo=pytz.UTC) #f"{year}-May/Jun/Jul/Aug"
        elif month in [9, 10, 11, 12]:
            return datetime.datetime(year=year, month=10, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Sep/Oct/Nov/Dec"

    def _get_interannual_period(self, date):
        month = date.month
        year = date.year

        if month in [1, 2, 3, 4, 5, 6]:
            return datetime.datetime(year=year, month=3, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Jan/Feb/Mar/Apr/May/Jun"
        elif month in [7, 8, 9, 10, 11, 12]:
            return datetime.datetime(year=year, month=9, day=30).replace(tzinfo=pytz.UTC) #f"{year}-Jul/Aug/Sep/Oct/Nov/Dec"

    def create_dict_timeframes(self, df_coastline_timeseries, df_other_timeseries):

        print('Creating dictionary for different time frames')

        # Create dictionary for different time frames
        dict_timeframes = {'coastline_timeseries': {}, 'other_timeseries': {}}

        # Loop through time frames
        for timeframe in self.list_timeframes:
            # Create copies of the DataFrames
            df_coastline_timeseries_copy = df_coastline_timeseries.copy()
            df_other_timeseries_copy = df_other_timeseries.copy()

            if timeframe == 'monthly':
                dict_timeframes['coastline_timeseries'][timeframe] = df_coastline_timeseries
                dict_timeframes['other_timeseries'][timeframe] = df_other_timeseries

            elif timeframe == 'annual':
                dict_timeframes['coastline_timeseries'][timeframe] = df_coastline_timeseries.groupby(
                    pd.Grouper(freq='Y'), dropna=False).mean()
                dict_timeframes['other_timeseries'][timeframe] = df_other_timeseries.groupby(
                    pd.Grouper(freq='Y'), dropna=False).mean()
                
            else:
                # Use getattr to call the function with the name stored in the variable `timeframe`
                df_coastline_timeseries_copy[timeframe] = [getattr(self, '_get_{}'.format(timeframe))(
                    df_coastline_timeseries.index[i]) for i in range(len(df_coastline_timeseries.index))]
                df_other_timeseries_copy[timeframe] = [getattr(self, '_get_{}'.format(timeframe))(
                    df_other_timeseries.index[i]) for i in range(len(df_other_timeseries.index))]
                
                # Set index to `timeframe`
                df_coastline_timeseries_copy = df_coastline_timeseries_copy.set_index(timeframe)
                df_other_timeseries_copy = df_other_timeseries_copy.set_index(timeframe)

                # Group by time frame
                dict_timeframes['coastline_timeseries'][timeframe] = df_coastline_timeseries_copy.groupby(
                    df_coastline_timeseries_copy.index, dropna=False).mean()
                dict_timeframes['other_timeseries'][timeframe] = df_other_timeseries_copy.groupby(
                    df_other_timeseries_copy.index, dropna=False).mean()

        return dict_timeframes

    def _longest_consecutive_sequence_indices(self, arr):

        max_length = 0  # Initialise the maximum consecutive sequence length
        current_length = 0  # Initialise the current consecutive sequence length
        max_sequence_indices = []  # Initialise the indices of the longest consecutive sequence
        current_sequence_indices = []  # Initialise the indices of the current consecutive sequence

        for i in range(len(arr)):
            if i > 0 and arr[i] == arr[i - 1] + 1:
                current_length += 1
                current_sequence_indices.append(i)
            else:
                current_length = 1
                current_sequence_indices = [i]

            if current_length > max_length:
                max_length = current_length
                max_sequence_indices = current_sequence_indices.copy()

        return max_sequence_indices

    def find_optimal_time_period_coastline(self):

        print('Finding optimal time period for each time frame for each transect')

        # Create empty DataFrame
        df_optimal_time_period_coastline = pd.DataFrame(columns=['transect', 'frequency', 'start_date', 'end_date', 'number of data points'])

        # Loop through time frames
        for timeframe in self.list_timeframes:
            # Retrieve DataFrame for time frame
            df_timeframe = self.dict_timeframes['coastline_timeseries'][timeframe]

            # Loop through transects
            for transect in df_timeframe.columns:
                # Retrieve time series
                array_coastline_position = df_timeframe[transect].values

                # Find indices of non NaN values
                idx_no_nan = np.argwhere(~pd.isna(array_coastline_position)).flatten()

                # Find indices of longest consecutive sequence
                max_sequence_indices = self._longest_consecutive_sequence_indices(idx_no_nan)

                # Add to DataFrame
                if len(max_sequence_indices) > 0:
                    start_date = df_timeframe.index[idx_no_nan[max_sequence_indices][0]]
                    end_date = df_timeframe.index[idx_no_nan[max_sequence_indices][-1]]
                    number_data_points = len(idx_no_nan[max_sequence_indices])
                    df_optimal_time_period_coastline = pd.concat([df_optimal_time_period_coastline, 
                                                                pd.DataFrame([{'transect': transect, 
                                                                                'frequency': timeframe, 
                                                                                'start_date': start_date.replace(tzinfo=pytz.UTC), 
                                                                                'end_date': end_date.replace(tzinfo=pytz.UTC), 
                                                                                'number of data points': number_data_points}])], axis=0) 

        # Set index to 'transect'
        df_optimal_time_period_coastline = df_optimal_time_period_coastline.set_index('transect')

        return df_optimal_time_period_coastline

    def find_optimal_time_period_other(self, df_coastline_timeseries):

        print('Finding optimal time period for each time frame for each other time series')

        # Create empty dictionary
        dict_combined_dataframes_optimal_time_period = {}
        dict_coastline_timeseries = {}

        # Loop through transects
        for transect in df_coastline_timeseries.columns:
            # Create empty dictionary for that transect
            dict_combined_dataframes_optimal_time_period[transect] = {}
            
            # Loop through time frames
            for timeframe in self.list_timeframes:

                # Create empty DataFrame
                df_optimal_time_period_other = pd.DataFrame(columns=['variable', 'frequency', 'start_date', 'end_date', 'number of data points'])

                # Find corresponding row in `df_optimal_time_period_coastline`
                row_coastline = self.df_optimal_time_period_coastline[(self.df_optimal_time_period_coastline.index == transect) & \
                                                                (self.df_optimal_time_period_coastline.frequency == timeframe)]
                
                # If no data for that transect and time frame
                if row_coastline.empty:
                    continue

                # Retrieve start and end dates from `df_optimal_time_period_coastline`
                start_coastline = datetime.datetime.utcfromtimestamp(row_coastline['start_date'].values[0].astype(datetime.datetime)/1e9).replace(tzinfo=pytz.UTC)
                end_coastline = datetime.datetime.utcfromtimestamp(row_coastline['end_date'].values[0].astype(datetime.datetime)/1e9).replace(tzinfo=pytz.UTC)

                # Retrieve DataFrame for time frame (other time series)
                df_timeframe = self.dict_timeframes['other_timeseries'][timeframe]

                # Retrieve time series for that transect
                df_coastline = df_coastline_timeseries

                # Fix date index
                index_date = (df_timeframe.index + MonthEnd(0)).to_numpy()
                index_date_coastline = (df_coastline.index + MonthEnd(0)).to_numpy()

                # Select time period to match coastline time period
                df_timeframe_aligned_period = df_timeframe[(index_date >= start_coastline) & (index_date <= end_coastline)]
                df_timeframe_aligned_period_coastline = df_coastline[(index_date_coastline >= start_coastline) & (index_date_coastline <= end_coastline)]

                if timeframe == 'monthly':
                    dict_coastline_timeseries[transect] = df_timeframe_aligned_period_coastline[transect]
                
                for other_variable in df_timeframe_aligned_period.columns:
                    # Retrieve time series
                    array_other_variable = df_timeframe_aligned_period[other_variable].values

                    # Find indices of non NaN values
                    idx_no_nan = np.argwhere(~pd.isna(array_other_variable)).flatten()

                    # Find indices of longest consecutive sequence
                    max_sequence_indices = self._longest_consecutive_sequence_indices(idx_no_nan)

                    # Add to DataFrame
                    if len(max_sequence_indices) > 0:
                        start_date = df_timeframe_aligned_period.index[idx_no_nan[max_sequence_indices][0]]
                        end_date = df_timeframe_aligned_period.index[idx_no_nan[max_sequence_indices][-1]]
                        number_data_points = len(idx_no_nan[max_sequence_indices])
                        df_optimal_time_period_other = pd.concat([df_optimal_time_period_other, 
                                                                pd.DataFrame([{'variable': other_variable, 
                                                                                'frequency': timeframe, 
                                                                                'start_date': start_date.replace(tzinfo=pytz.UTC), 
                                                                                'end_date': end_date.replace(tzinfo=pytz.UTC), 
                                                                                'number of data points': number_data_points}])], axis=0)

                # Retrieve maximum number of data points and number of data points
                n_dp_max = df_optimal_time_period_other['number of data points'].max()
                n_dp = df_optimal_time_period_other['number of data points']

                if df_optimal_time_period_other.empty:
                    continue

                # Sea level anomaly must be included
                if 'sea_level_anomaly' in df_optimal_time_period_other['variable'].values:
                    int_min_var = df_optimal_time_period_other[df_optimal_time_period_other['variable'] == 'sea_level_anomaly']

                else:
                    cutoff_df_optimal_time_period_other = df_optimal_time_period_other[(abs((n_dp_max - n_dp) / ((n_dp_max + n_dp) / 2)) * 100) < 25.]
                    int_min_var = cutoff_df_optimal_time_period_other[cutoff_df_optimal_time_period_other['number of data points'] == cutoff_df_optimal_time_period_other['number of data points'].min()]
                
                # Final optimal time period
                final_start = datetime.datetime.utcfromtimestamp(int_min_var['start_date'].values[0].astype(datetime.datetime)/1e9).replace(tzinfo=pytz.UTC)
                final_end = datetime.datetime.utcfromtimestamp(int_min_var['end_date'].values[0].astype(datetime.datetime)/1e9).replace(tzinfo=pytz.UTC)

                # Select time period to match optimal time period (other variables)
                df_optimal_period_other = df_timeframe[((df_timeframe.index + MonthEnd(0)).to_numpy() >= final_start) & ((df_timeframe.index + MonthEnd(0)).to_numpy() <= final_end)]

                # Fill NaN for 2-metre temperature
                df_optimal_period_other['2_metre_temperature'] = df_optimal_period_other['2_metre_temperature'].interpolate(method='linear')

                # Drop columns with remaining NaN values
                df_optimal_period_other = df_optimal_period_other.dropna(axis=1, how='any')

                # Select time period to match optimal time period (coastline position)
                df_optimal_period_coastline = self.dict_timeframes['coastline_timeseries'][timeframe]
                df_optimal_period_coastline = df_optimal_period_coastline[((df_optimal_period_coastline.index + MonthEnd(0)).to_numpy() >= final_start) & ((df_optimal_period_coastline.index + MonthEnd(0)).to_numpy() <= final_end)]
                df_optimal_period_coastline_transect = df_optimal_period_coastline[transect]

                # Concatenate DataFrames
                df_optimal_time_period = pd.concat([df_optimal_period_coastline_transect, df_optimal_period_other], axis=1)

                # Save in information to dictionary
                dict_combined_dataframes_optimal_time_period[transect][timeframe] = df_optimal_time_period

        return dict_combined_dataframes_optimal_time_period, dict_coastline_timeseries
    
    def main(self):

        if 'timeseries_preprocessing' in self.island_info.keys() and not self.overwrite:
            return self.island_info

        print('\n-------------------------------------------------------------------')
        print('Time series pre-processing')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # Create dictionary for pre-processed time series
        if 'timeseries_preprocessing' not in self.island_info.keys():
            self.island_info['timeseries_preprocessing'] = {}
        
        # Retrieve all available time series (excluding socioeconomics) from `island_info` dictionary
        list_timeseries, list_timeseries_socioeconomics, list_confounders, list_timeseries_climate_indices = self.retrieve_available_timeseries()        

        # Make sure they all share the same UTC datetime index and combine them in one DataFrame
        df_timeseries = self.utc_datetime_index_and_concat(list_timeseries)
        df_timeseries_socioeconomics = self.utc_datetime_index_and_concat(list_timeseries_socioeconomics)
        df_confounders = self.utc_datetime_index_and_concat(list_confounders)
        try:
            df_timeseries_climate_indices = self.utc_datetime_index_and_concat(list_timeseries_climate_indices)
        except:
            df_timeseries_climate_indices = pd.DataFrame()

        # Plot confounders
        if self.plot_confounders:
            plt.figure()
            df_confounders.plot()
            plt.show()

        # Replace outlier values with NaN using z-score (abnormally high values)
        df_timeseries_remove_outliers = self.replace_outliers_with_nan(df_timeseries)

        # Group by frequency
        df_timeseries_frequency = self.group_by_frequency(df_timeseries_remove_outliers)

        # Fill NaN with mean (only for time series with very few NaN)
        df_timeseries_frequency_rolling = self.fill_nan_with_rolling_mean(df_timeseries_frequency)

        # Retrieve coastline and other time series
        df_coastline_timeseries, df_other_timeseries = self.get_coastline_and_other_timeseries(df_timeseries_frequency_rolling)

        # List of time frames
        self.list_timeframes = ['monthly', 
                                'two_month_period', 
                                'three_month_period', 
                                'four_month_period', 
                                'interannual_period', 
                                'annual']
        
        self.dict_timeframes = self.create_dict_timeframes(df_coastline_timeseries, df_other_timeseries)

        # Find optimal time period for each time frame for each transect
        self.df_optimal_time_period_coastline = self.find_optimal_time_period_coastline()

        # Find optimal time period for each time frame for each other time series
        # This is done within the coastline position time frame
        dict_combined_dataframes_optimal_time_period, dict_coastline_timeseries = self.find_optimal_time_period_other(df_coastline_timeseries)

        # Save information to dictionary
        self.island_info['timeseries_preprocessing']['df_coastline_timeseries'] = df_coastline_timeseries
    
        self.island_info['timeseries_preprocessing']['raw'] = {'df_timeseries_environment': df_timeseries,
                                                              'df_timeseries_socioeconomics': df_timeseries_socioeconomics,
                                                              'df_confounders': df_confounders,
                                                              'df_timeseries_climate_indices': df_timeseries_climate_indices}
        self.island_info['timeseries_preprocessing']['optimal time period'] = {'list_timeframes': self.list_timeframes,
                                                                              'dict_timeseries': dict_combined_dataframes_optimal_time_period,
                                                                              'dict_coastline_timeseries': dict_coastline_timeseries}
        
        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)

        return self.island_info