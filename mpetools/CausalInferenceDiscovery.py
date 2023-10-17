"""
This module allows us to match time frames and frequency (no NaN values and no extrapolation).
TODO: find time frames with the most time series

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
from mpetools import run_all_functions
import numpy as np
import datetime

class MatchTimeFrameFrequency:
    def __init__(self, island, country, date_range=['2010-01-01', '2022-12-31'], frequency='monthly', island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands')):
        self.island = island
        self.country = country
        self.date_range = date_range
        self.frequency = frequency
        self.island_info_path = island_info_path
        
    def find_optimal_date_range(self):
        pass
    
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.info_island = run_all_functions.run_all_functions(self.island, self.country)    

        print('\n-------------------------------------------------------------------')
        print('MATCHING TIME FRAME AND FREQUENCY')
        print('Retrieving all available time series between {} and {} at a {} frequency'.format(self.date_range[0], self.date_range[1], self.frequency))
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # Combine all time series in one DataFrame
        list_timeseries = [self.info_island[key]['timeseries'] for key in self.info_island.keys() if 'timeseries' in self.info_island[key]]
        df_timeseries = pd.concat(list_timeseries, axis=1)
        df_timeseries = df_timeseries.apply(pd.to_numeric)

        # Group by frequency
        df_timeseries_fq = df_timeseries.groupby(pd.Grouper(freq=self.frequency[0].capitalize())).mean()

        # Start and end dates as `datetime`
        start_date = datetime.datetime.strptime(self.date_range[0], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(self.date_range[1], '%Y-%m-%d')

        # Select time frame
        df_timeseries_fq_tf = df_timeseries_fq[(df_timeseries_fq.index >= start_date) & (df_timeseries_fq.index <= end_date)]

        # Drop columns containing NaN
        df_timeseries_fq_tf_cleaned = df_timeseries_fq_tf.dropna(axis=1, how='any')

        return self.info_island, df_timeseries_fq_tf_cleaned