"""
This module contains tools to make some time series analyses.
TODO: find time frames with the most time series
TODO: characterise time series (?)

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
from mpetools import IslandTime
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
from scipy import stats
from pandas.tseries.offsets import MonthEnd
import pymannkendall as mk
from sklearn.preprocessing import LabelEncoder

class TimeSeriesAnalysis:
    def __init__(self, time_series, timeseries_with_nan, time_series_name, time_series_unit, alpha=0.05):
        self.time_series = time_series
        self.timeseries_with_nan = timeseries_with_nan
        self.time_series_name = time_series_name
        self.time_series_unit = time_series_unit
        self.alpha = alpha
    
    def consistency_check(self, results, min_ratio_agreement=0.6):
        # Create label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder to results
        label_encoder_results = label_encoder.fit(results)

        # Check if all tests agree
        if len(label_encoder_results.classes_) == 1:
            print('All tests agree')
            return label_encoder_results.classes_[0]
        
        # Check if tests agree at least min_percentage_agreement
        else:            
            # Get ratio for every category
            ratios = [results.count(category)/len(results) for category in label_encoder_results.classes_]

            if max(ratios) >= min_ratio_agreement:
                print('Tests agree at {}% of the time'.format(np.round(max(ratios)*100), 0))
                return label_encoder_results.classes_[np.argmax(ratios)]
            
            else:
                print('Tests do not agree at least {}% of the time'.format(min_ratio_agreement*100))
                return None

    def test_presence_of_trend(self):
        # Mann-Kendall tests
        # https://pypi.org/project/pymannkendall/
        # https://joss.theoj.org/papers/10.21105/joss.01556
        
        tests = [mk.original_test, 
                 mk.hamed_rao_modification_test, 
                 mk.yue_wang_modification_test,
                 mk.pre_whitening_modification_test, 
                 mk.trend_free_pre_whitening_modification_test,
                 mk.seasonal_test]
        
        trend = []
        h = []
        for test in tests:
            result = test(self.time_series, alpha=self.alpha)
            trend.append(result.trend)
            h.append(result.h)

        # Check if all tests agree
        result_consistent = self.consistency_check(trend)
        if result_consistent is None:
            result_consistent = 'inconsistent'
    
        print('Result: {}'.format(result_consistent))

        # Save for later
        self.result_consistent = result_consistent
  
    def test_presence_of_seasonality(self):
        # Seasonal Mann-Kendall test
        # https://pypi.org/project/pymannkendall/
        pass

        # Update dictionary
        pass

    def test_stationarity(self):
        # Augmented Dickey-Fuller test
        # https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
        #
        pass

        # Update dictionary
        pass

    def plot_qualitative_characterisation(self):
        pass

    def qualitative_characterisation(self):
        
        # Presence of trend
        self.test_presence_of_trend()

        # Presence of seasonality
        self.test_presence_of_seasonality()

        # Stationarity
        self.test_stationarity()

        # Plotting qualitative characterisation
        self.plot_qualitative_characterisation()

    def main(self):
        
        # Step 1: qualitative characterisation
        self.qualitative_characterisation()

        # Step 2: time series decomposition
        pass

        # Step 3: quantitative characterisation
        pass

