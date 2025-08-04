"""
This module contains tools to make time series analysis on island time series.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import warnings
warnings.filterwarnings('ignore')
from IslandTime import plot_shoreline_transects, save_island_info
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymannkendall as mk
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.stattools import acf, adfuller, kpss
from statsmodels.tsa.seasonal import STL
from scipy.signal import find_peaks, argrelextrema
from scipy.fft import fft
from scipy.optimize import curve_fit
import ruptures as rpt
import matplotlib
import matplotlib.style
from scipy.stats import linregress
import datetime
import shapely
from tqdm import tqdm
import pandas as pd
import pytz
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import geopandas as gpd
import Rbeast as rb
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import inspect
import warnings
from functools import lru_cache
import os
import pickle
# os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1000'

warnings.filterwarnings('ignore')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class TimeSeriesAnalysis:
    def __init__(self, island_info, period=12, alpha=0.05, 
                 model_change_detection='l2', width_change_detection=12, z_score_seasonality=2.,
                 verbose=False, debug=False, overwrite=True, plot_results_transect=False, 
                 plot_only=False, threshold_k=0.0009, n_seasonality=2, overwrite_transect=False, transect_to_plot=None, overwrite_all=False):
        self.island_info = island_info
        self.period = period
        self.alpha = alpha
        self.model_change_detection = model_change_detection
        self.width_change_detection = width_change_detection
        self.z_score_seasonality = z_score_seasonality
        self.verbose = verbose
        self.debug = debug
        self.overwrite = overwrite
        self.plot_results_transect = plot_results_transect
        self.threshold_k = threshold_k
        self.plot_only = plot_only
        self.n_seasonality = n_seasonality
        self.overwrite_transect = overwrite_transect
        self.transect_to_plot = transect_to_plot
        self.overwrite_all = overwrite_all
        self.island = self.island_info['general_info']['island']
        self.country = self.island_info['general_info']['country']
        self.reference_shoreline = self.island_info['spatial_reference']['reference_shoreline']
        self.transects = self.island_info['spatial_reference']['transects'].keys()
        self.transects_geo = self.island_info['spatial_reference']['transects']
    
    def _consistency_check(self, results, min_ratio_agreement=0.6):
        # Create label encoder
        label_encoder = LabelEncoder()

        # Fit label encoder to results
        label_encoder_results = label_encoder.fit(results)

        # Check if all tests agree
        if len(label_encoder_results.classes_) == 1:
            if self.verbose:
                print('All tests agree')
            return label_encoder_results.classes_[0]
        
        # Check if tests agree at least min_percentage_agreement
        else:            
            # Get ratio for every category
            ratios = [results.count(category)/len(results) for category in label_encoder_results.classes_]

            if max(ratios) >= min_ratio_agreement:
                if self.verbose:
                    print('Tests agree at {}%'.format(np.round(max(ratios)*100), 0))
                return label_encoder_results.classes_[np.argmax(ratios)]
            
            else:
                if self.verbose:
                    print('Tests do not agree at least {}% of the time'.format(min_ratio_agreement*100))
                return None
            

    def _delta_method(self, COVB, param, x_new, f, x, y, alpha=0.05):
        # - - -
        # Function to calculate the confidence interval and prediction interval
        # for any user-defined regression function using the delta-method
        # as described in Sec 5.1 of the following online statistics lecture:
        # https://jchiquet.github.io/MAP566/docs/regression/map566-lecture-nonlinear-regression.html
        #
        # Greg Pelletier (gjpelletier@gmail.com)
        # - - -
        # INPUT
        # COVB = variance-covariance matrix of the model parameters (e.g. from scipy or lmfit)
        # param = best-fit parameters of the regression function (e.g. from scipy or lmfit)
        # x_new = new x values to evaluate new predicted y_new values (e.g. x_new=linspace(min(x),max(x),100)
        # f = user-defined regression lambda function to predict y given inputs if param and x values (e.g. observed x or x_new)
        # 	For example, if using the 3-parameter nonlinear regression exponential threshold function, then
        # 	f = lambda param,xval : param[0] + param[1] * exp(param[2] * xval)
        # x = observed x
        # y = observed y
        # alpha = significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
        # - - -
        # OUTPUT
        # dict = dictionary of output varlables with the following keys:
        #        'param': best-fit parameter values used as input
        #        'COVB': variance-covariance matrix used as input
        #        'fstr': string of the input lambda function of the regression model
        #        'alpha': input significance level for the confidence/prediction interval (e.g. alpha=0.05 is the 95% confidence/prediction interval)
        #        'x': observed x values used as input
        #        'y': observed y values used as input
        #        'yhat': predicted y at observed x values
        #        'x_new': new x-values used as input to evaluate unew predicted y_new values
        #        'y_new': new predicted y_new values at new x_new values
        #        'lwr_conf': lower confidence interval for each value in x_new
        #        'upr_conf': upper confidence interval for each value in x_new
        #        'lwr_pred': lower prediction interval for each value in x_new
        #        'upr_pred': upper prediction interval for each value in x_new
        #        'grad_new': derivative gradients at x_new (change in f(x_new) per change in each param)
        #        'G_new': variance due to each paramter at x_new
        #        'GS_new': variance due to all parameters combined at x_new
        #        'SST': Sum of Squares Total
        #        'SSR': Sum of Squares Regression
        #        'SSE': Sum of Squares Error
        #        'MSR': Mean Square Regression
        #        'MSE': Mean Square Error of the residuals
        #        'syx': standard error of the estimate
        #        'nobs': number of observations
        #        'nparam': number of parameters
        #        'df': degrees of freedom = nobs-nparam
        #        'qt': 2-tailed t-statistic at alpha
        #        'Fstat': F-statistic = MSR/MSE
        #        'dfn': degrees of freedom for the numerator of the F-test = nparam-1
        #        'dfd': degrees of freedom for the denominator of the F-test = nobs-nparam
        #        'pvalue': signficance level of the regression from the probability of the F-test
        #        'rsquared': r-squared = SSR/SST
        #        'adj_rsquared': adjusted squared
        # - - -

        # calculate predicted y_new at each x_new
        y_new = f(param, x_new)

        # calculate derivative gradients at x_new (change in f(x_new) per change in each param)
        grad_new = np.empty(shape=(np.size(x_new), np.size(param)))

        h = 1e-8       # h = small change for each param to balance truncation error and rounding error of the gradient
        
        for i in range(np.size(param)):
            # make a copy of param
            param2 = np.copy(param)

            # gradient forward
            param2[i] = (1+h) * param[i]
            y_new2 = f(param2, x_new)
            dy = y_new2 - y_new
            dparam = param2[i] - param[i]
            grad_up = dy / dparam

            # gradient backward
            param2[i] = (1-h) * param[i]
            y_new2 = f(param2, x_new)
            dy = y_new2 - y_new
            dparam = param2[i] - param[i]
            grad_dn = dy / dparam

            # centered gradient is the average gradient forward and backward
            grad_new[:,i] = (grad_up + grad_dn) / 2

        # calculate variance in y_new due to each parameter and for all parameters combined
        G_new = np.matmul(grad_new, COVB) * grad_new         # variance in y_new due to each param at each x_new
        GS_new = np.sum(G_new, axis=1)                       # total variance from all param values at each x_new
        
        # - - -
        # lwr_conf and upr_conf are confidence intervals of the best-fit curve
        nobs = np.size(x)
        nparam = np.size(param)
        df = nobs - nparam
        qt = stats.t.ppf(1-alpha/2, df)
        delta_f = np.sqrt(GS_new) * qt
        lwr_conf = y_new - delta_f
        upr_conf = y_new + delta_f

        # - - -
        # lwr_pred and upr_pred are prediction intervals of new observations
        yhat = f(param,x)
        SSE = np.sum((y-yhat) ** 2)                 # sum of squares (residual error)
        MSE = SSE / df                              # mean square (residual error)
        syx = np.sqrt(MSE)                          # std error of the estimate
        delta_y = np.sqrt(GS_new + MSE) * qt
        lwr_pred = y_new - delta_y
        upr_pred = y_new + delta_y

        # - - -
        # optional additional outputs of regression statistics
        SST = np.sum(y **2) - np.sum(y) **2 / nobs  # sum of squares (total)
        SSR = SST - SSE                             # sum of squares (regression model)
        MSR = SSR / (np.size(param)-1)              # mean square (regression model)
        Fstat = MSR / MSE           # F statistic
        dfn = np.size(param) - 1    # df numerator = degrees of freedom for model = number of model parameters - 1
        dfd = df                    # df denomenator = degrees of freedom of the residual = df = nobs - nparam
        pvalue = 1-stats.f.cdf(Fstat, dfn, dfd)      # p-value of F test statistic
        rsquared = SSR / SST                                                        # ordinary rsquared
        adj_rsquared = 1-(1-rsquared)*(np.size(x)-1)/(np.size(x)-np.size(param)-1)  # adjusted rsquared

        # - - -
        # make a string of the lambda function f to save in the output dictionary
        fstr = str(inspect.getsourcelines(f)[0])

        # make the dictionary of output variables from the delta-method
        dict = {
                'param': param,
                'COVB': COVB,
                'fstr': fstr,
                'alpha': alpha,
                'x': x,
                'y': y,
                'yhat': yhat,
                'x_new': x_new,
                'y_new': y_new,
                'lwr_conf': lwr_conf,
                'upr_conf': upr_conf,
                'lwr_pred': lwr_pred,
                'upr_pred': upr_pred,
                'grad_new': grad_new,
                'G_new': G_new,
                'GS_new': GS_new,
                'SST': SST,
                'SSR': SSR,
                'SSE': SSE,
                'MSR': MSR,
                'MSE': MSE,
                'syx': syx,
                'nobs': nobs,
                'nparam': nparam,
                'df': df,
                'qt': qt,
                'Fstat': Fstat,
                'dfn': dfn,
                'dfd': dfd,
                'pvalue': pvalue,
                'rsquared': rsquared,
                'adj_rsquared': adj_rsquared
                }

        return dict

    # @lru_cache(maxsize=128)
    def test_presence_of_trend(self, time_series, time_series_name, res_BEAST_transect):
        
        if self.verbose:
            print('Checking for the presence of a TREND')

        # Mann-Kendall tests
        # https://pypi.org/project/pymannkendall/
        # https://joss.theoj.org/papers/10.21105/joss.01556
        
        # List of tests
        tests = [mk.original_test, 
                 mk.hamed_rao_modification_test, 
                 mk.yue_wang_modification_test,
                 mk.pre_whitening_modification_test, 
                 mk.trend_free_pre_whitening_modification_test,
                 mk.seasonal_test]
        
        # Initialise empty lists to store results
        trends = []
        h = []

        # Iterate over tests
        for test in tests:
            results = test(time_series.values, alpha=self.alpha)
            trends.append(results.trend)
            h.append(results.h)

        # Check if all tests agree
        result_consistent = self._consistency_check(trends)
        if result_consistent is None:
            result_consistent = 'inconsistent'

        if self.verbose:
            print('Result: {}'.format(result_consistent))

        # Decomposition
        res = STL(time_series, period=self.period).fit()
        trend = res.trend

        # BEAST decomposition
        # try:
        # res_BEAST = rb.beast(time_series.values, start=[time_series.index[0].year, time_series.index[0].month, time_series.index[0].day], season='harmonic', deltat='1/12 year', period='1 year', quiet=True, print_progress=True, hasOutliers=False)
        trend_BEAST = res_BEAST_transect.trend.Y
        
        # except:
        #     trend_BEAST = trend

        # Assign consistent result to trend_result
        trend_result = result_consistent

        # Point-change detection
        # C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. Signal Processing, 167:107299, 2020
        algo_trend = rpt.Window(width=self.width_change_detection, model=self.model_change_detection).fit(trend.values)
        algo_time_series = rpt.Window(width=self.width_change_detection, model=self.model_change_detection).fit(time_series.values)

        # Predict change points
        try:
            result_bkp_trend = algo_trend.predict(n_bkps=12)

        except:
            try:
                result_bkp_trend = algo_trend.predict(n_bkps=6)
            
            except:
                result_bkp_trend = algo_trend.predict(n_bkps=3)

        try:
            result_bkp_time_series = algo_time_series.predict(n_bkps=12)

        except:
            try:
                result_bkp_time_series = algo_time_series.predict(n_bkps=6)
            
            except:
                result_bkp_time_series = algo_time_series.predict(n_bkps=3)

        # Concatenate results from trend and time series and remove duplicates
        result_bkp = np.concatenate((result_bkp_trend[:-1], result_bkp_time_series[:-1]))
        result_bkp = np.unique(result_bkp)

        # Identify change points that result in a change in the direction of the trend
        # Concatenate 0 at the beginning of the result_bkp array
        result_bkp_with_0 = np.concatenate(([0], result_bkp))

        # Initialise an empty list to store the Mann-Kendall test results
        result_bkp_mk = []

        # Iterate over the indices of result_bkp_with_0
        for idx_bkp in range(len(result_bkp_with_0) - 1):
            # Extract the selected time series
            sel_ts = res.trend[result_bkp_with_0[idx_bkp]:result_bkp_with_0[idx_bkp + 1]]

            # Perform the Mann-Kendall test with a significance level of alpha
            res_mk = mk.original_test(sel_ts, alpha=self.alpha)

            # Append the Mann-Kendall test result to the list
            result_bkp_mk.append(res_mk[0])

        # Initialise empty lists to store breakpoints and corresponding Mann-Kendall test results
        result_bkp_significant = []
        result_mk_significant = []

        # Iterate over the range of the length of result - 1
        for res in range(len(result_bkp) - 1):
            # Check if consecutive Mann-Kendall test results are equal
            if result_bkp_mk[res] == result_bkp_mk[res + 1]:
                continue

            else:
                # Append the breakpoint and corresponding Mann-Kendall result to the lists
                result_bkp_significant.append(result_bkp[res])
                result_mk_significant.append(result_bkp_mk[res])   

        # Dates of change points
        result_bkp_dates = time_series.index[result_bkp]
        result_bkp_dates_significant = time_series.index[result_bkp_significant]     

        # Fit linear regression to trend
        numeric_dates = np.array([(date - time_series.index[0]).days for date in time_series.index])
        trend_slope, intercept, r_value, p_value, std_err = linregress(numeric_dates, time_series.values)
        trend_slope_BEAST, intercept_BEAST, r_value_BEAST, p_value_BEAST, std_err_BEAST = linregress(numeric_dates, trend_BEAST)

        # Fit line equation: y = mx + b
        line = trend_slope * numeric_dates + intercept
        line_BEAST = trend_slope_BEAST * numeric_dates + intercept_BEAST

        # Save in island_info
        results_trend_dict = {'trend_result': trend_result,
                                'trend_component_STL': trend,
                                'trend_slope': trend_slope,
                                'trend_fitted_line': line,
                                'trend_bkp_dates': result_bkp_dates,
                                'trend_bkp_dates_significant': result_bkp_dates_significant,
                                'trend_mk_significant': result_mk_significant,
                                'trend_component_BEAST': trend_BEAST,
                                'trend_slope_BEAST': trend_slope_BEAST,
                                'trend_fitted_line_BEAST': line_BEAST}
        
        self.island_info['timeseries_analysis'][time_series_name]['trend'] = results_trend_dict
        self.island_info['timeseries_analysis'][time_series_name]['full_results_BEAST'] = res_BEAST_transect

        return results_trend_dict

    def _relation_indian_monsoon(self, time_series, time_series_name):
        
        list_tests = ['t-test', 'mannwhitneyu', 'clustering']

        # Initialise empty list to store results
        results_indian_monsoon = []

        # Create SW Indian Monsoon period DataFrame
        for yr in range(2012, 2023):
            idx_im = pd.date_range(datetime.datetime(yr, 5, 1), datetime.datetime(yr, 9, 30), freq='M')
            df_im = pd.DataFrame(index=idx_im)
            
            if yr == 2012:
                df_im_t = df_im
            
            else:
                df_im_t = pd.concat([df_im_t, df_im])

        # Convert index to localized datetime
        df_im_t.index = [pytz.utc.localize(df_im_t.index[i]) for i in range(len(df_im_t.index))]

        # Condition for Indian Monsoon and data masks
        condition_im = time_series.index.isin(df_im_t.index)
        mask_im = pd.Series(condition_im, index=time_series.index)

        # Masked time series (index within Indian Monsoon period and outside Indian Monsoon period)
        masked_df_ts = time_series.dropna().where(mask_im, np.nan)
        other_df_ts = time_series.dropna().where(~mask_im, np.nan)

        # Retrieve trend component from STL decomposition
        trend_component_STL = self.island_info['timeseries_analysis'][time_series_name]['trend']['trend_component_STL']

        # De-trended masked time series
        detrended_masked_df_ts = (masked_df_ts - trend_component_STL).dropna()
        detrended_other_df_ts = (other_df_ts - trend_component_STL).dropna()

        # Iterate over tests
        for test in list_tests:
            # Test 1: Student t-test
            if test == 't-test':
                # Perform Student t-test
                t_statistic, p_value = stats.ttest_ind(detrended_masked_df_ts, detrended_other_df_ts)

                # Significant difference between the two groups
                if p_value < self.alpha:
                    # Check if mean of masked time series is higher than mean of other time series
                    if np.mean(detrended_masked_df_ts) > np.mean(detrended_other_df_ts):
                        results_indian_monsoon.append('significant increase during SW Monsoon')
                    
                    else:
                        results_indian_monsoon.append('significant decrease during SW Monsoon')
                
                # No significant difference between the two groups
                else:
                    results_indian_monsoon.append('not significant')

            # Test 2: Mann-Whitney U test
            elif test == 'mannwhitneyu':
                # Perform Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(detrended_masked_df_ts, detrended_other_df_ts)

                # Significant difference between the two groups
                if p_value < self.alpha:
                    # Check if mean of masked time series is higher than mean of other time series
                    if np.mean(detrended_masked_df_ts) > np.mean(detrended_other_df_ts):
                        results_indian_monsoon.append('significant increase during SW Monsoon')
                    
                    else:
                        results_indian_monsoon.append('significant decrease during SW Monsoon')
                
                # No significant difference between the two groups
                else:
                    results_indian_monsoon.append('not significant')

            # Test 3: clustering (TODO)
            elif test == 'clustering':
                pass

        # Check if all tests agree
        result_consistent = self._consistency_check(results_indian_monsoon)

        return result_consistent

    def _behaviour_indian_monsoon(self, time_series, time_series_name, peaks_seasonal_STL, seasonal_fit_STL, k):

        def find_closest_date(reference_date, year):

            # Define labels
            labels = ['sw_monsoon', 'ne_monsoon', 'transition_ne_sw', 'transition_sw_ne']
            
            # Define dates (middle of monsoon and transition periods)
            dates = [
                datetime.datetime(year, 7, 15, tzinfo=pytz.utc),
                datetime.datetime(year, 1, 15, tzinfo=pytz.utc),
                datetime.datetime(year, 4, 1, tzinfo=pytz.utc),
                datetime.datetime(year, 11, 1, tzinfo=pytz.utc),
            ]

            # Find argmin
            idx_closest_date = np.argmin([abs(date - reference_date) for date in dates])
            idx_second_closest_date = np.argsort([abs(date - reference_date) for date in dates])[1]
            
            # Return label
            return labels[idx_closest_date], labels[idx_second_closest_date]
        
        # Peaks of seasonal component
        dates_peaks_seasonal_STL = time_series.index[peaks_seasonal_STL]

        # Minima of seasonal component
        minima = argrelextrema(seasonal_fit_STL, np.less)[0]
        dates_minima_seasonal_STL = time_series.index[minima]

        closest_dates_peaks = []
        second_closest_dates_peaks = []
        for peak in dates_peaks_seasonal_STL:
            yr = peak.year
            closest_date, second_closest_date = find_closest_date(peak, yr)
            closest_dates_peaks.append(closest_date)
            second_closest_dates_peaks.append(second_closest_date)

        closest_dates_minima = []
        second_closest_dates_minima = []
        for minimum in dates_minima_seasonal_STL:
            yr = minimum.year
            closest_date, second_closest_date = find_closest_date(minimum, yr)
            closest_dates_minima.append(closest_date)
            second_closest_dates_minima.append(second_closest_date)

        # Find most common label for peaks
        if len(closest_dates_peaks) == 1:
            most_common_label_peaks = closest_dates_peaks[0]
        
        elif len(closest_dates_peaks) == 0:
            most_common_label_peaks = None
        
        else:
            most_common_label_peaks = Counter(closest_dates_peaks).most_common(1)[0][0]
        
        # Find second most common label for peaks
        if len(closest_dates_peaks) == 0:
            second_most_common_label_peaks = None
        
        else:
            if len(np.unique(closest_dates_peaks)) == 1:
                second_most_common_label_peaks = Counter(second_closest_dates_peaks).most_common(1)[0][0]
            
            else:
                second_most_common_label_peaks = Counter(closest_dates_peaks).most_common(2)[1][0]
        
        # Find most common label for minima
        if len(closest_dates_minima) == 1:
            most_common_label_minima = closest_dates_minima[0]
        
        elif len(closest_dates_minima) == 0:
            most_common_label_minima = None
        
        else:
            most_common_label_minima = Counter(closest_dates_minima).most_common(1)[0][0]

        # Find second most common label for minima
        if len(closest_dates_minima) == 0:
            second_most_common_label_minima = None
        
        else:
            if len(np.unique(closest_dates_minima)) == 1:
                second_most_common_label_minima = Counter(second_closest_dates_minima).most_common(1)[0][0]
            
            else:
                second_most_common_label_minima = Counter(closest_dates_minima).most_common(2)[1][0]

        plott = False

        if abs(k) > self.threshold_k:
            behaviour_indian_monsoon = 'undetermined'
        
        else:
            if most_common_label_peaks == 'sw_monsoon' and most_common_label_minima == 'ne_monsoon':
                behaviour_indian_monsoon = 'Peak of accretion during SW Monsoon and peak of erosion during NE Monsoon'
            
            elif most_common_label_peaks == 'ne_monsoon' and most_common_label_minima == 'sw_monsoon':
                behaviour_indian_monsoon = 'Peak of erosion during SW Monsoon and peak of accretion during NE Monsoon'
            
            elif most_common_label_peaks == 'transition_ne_sw' and most_common_label_minima == 'transition_sw_ne':
                behaviour_indian_monsoon = 'Strictly accreting during SW Monsoon and strictly eroding during NE Monsoon'

            elif most_common_label_peaks == 'transition_sw_ne' and most_common_label_minima == 'transition_ne_sw':
                behaviour_indian_monsoon = 'Strictly eroding during SW Monsoon and strictly accreting during NE Monsoon'
        
            elif most_common_label_peaks == 'transition_ne_sw' and most_common_label_minima == 'sw_monsoon':
                behaviour_indian_monsoon = 'Mostly eroding during SW Monsoon and mostly accreting during NE Monsoon'
            
            elif most_common_label_peaks == 'transition_sw_ne' and most_common_label_minima == 'sw_monsoon':
                behaviour_indian_monsoon = 'Mostly accreting during SW Monsoon and mostly eroding during NE Monsoon'
            
            elif most_common_label_peaks == 'transition_ne_sw' and most_common_label_minima == 'ne_monsoon':
                behaviour_indian_monsoon = 'Mostly eroding during SW Monsoon and mostly accreting during NE Monsoon'
            
            elif most_common_label_peaks == 'transition_sw_ne' and most_common_label_minima == 'ne_monsoon':
                behaviour_indian_monsoon = 'Mostly accreting during SW Monsoon and mostly eroding during NE Monsoon'
            
            elif most_common_label_peaks == 'ne_monsoon' and most_common_label_minima == 'transition_ne_sw':
                behaviour_indian_monsoon = 'Mostly accreting during SW Monsoon and mostly eroding during NE Monsoon'
            
            elif most_common_label_peaks == 'ne_monsoon' and most_common_label_minima == 'transition_sw_ne':
                behaviour_indian_monsoon = 'Mostly eroding during SW Monsoon and mostly accreting during NE Monsoon'
            
            elif most_common_label_peaks == 'sw_monsoon' and most_common_label_minima == 'transition_ne_sw':
                behaviour_indian_monsoon = 'Mostly accreting during SW Monsoon and mostly eroding during NE Monsoon'
            
            elif most_common_label_peaks == 'sw_monsoon' and most_common_label_minima == 'transition_sw_ne':
                behaviour_indian_monsoon = 'Mostly eroding during SW Monsoon and mostly accreting during NE Monsoon'

            else:
                behaviour_indian_monsoon = 'undetermined'

        if plott:
            fig, ax = plt.subplots()
            # Plot fitting sinusoidal function
            if peaks_seasonal_STL is not None:
                for peak in peaks_seasonal_STL:
                    ax.axvline(time_series.index[peak], color='r', linestyle='--')
                
            if minima is not None:
                for minimum in minima:
                    ax.axvline(time_series.index[minimum], color='pink', linestyle='--')
            for year in range(2010, 2023):
                if year == 2010:
                    ax.axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2, label='NE Monsoon')
                    ax.axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2, label='SW Monsoon')
                else:
                    ax.axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2)
                    ax.axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2)
            ax.plot(time_series.index, seasonal_fit_STL, color='g')
            ax.legend(fontsize=10)
            ax.set_xlim(time_series.index[0], time_series.index[-1])
            ax.set_title('Name: {}'.format(time_series_name))
            plt.show()
            # plt.close(fig)
        
        return behaviour_indian_monsoon, most_common_label_peaks, most_common_label_minima

    # @lru_cache(maxsize=128)
    def test_presence_of_seasonality(self, time_series, time_series_name, res_BEAST_transect):

        if self.verbose:
            print('Checking for the presence of SEASONALITY')

        # Define sinusoidal function (for fitting)
        def seasonal_function_fitting(x, A, k, period, phi, offset):
            return A * np.exp(k * x) * np.sin(2 * np.pi / period * x + phi) + offset

        def seasonal_function_skewed_fitting(x, A, k, sp, period, phi, offset):
            inside_sin = 2 * np.pi / period * x + phi
            skewed_sin = np.sin(inside_sin) / np.sqrt((sp + np.cos(inside_sin))**2 + np.sin(inside_sin)**2)
            return A * np.exp(k * x) * skewed_sin + offset
        
        # List of tests
        tests = ['acf', 'fourier_analysis', 'sinus_fitting']

        # STL decomposition on original time series and filtered time series
        res = STL(time_series, period=self.period).fit()
        
        # BEAST decomposition
        # try:
        # print(time_series.values)
        # print([time_series.index[0].year, time_series.index[0].month, time_series.index[0].day])
        # res_BEAST = rb.beast(time_series.values, start=[time_series.index[0].year, time_series.index[0].month, time_series.index[0].day], season='harmonic', deltat='1/12 year', period='1 year', quiet=False, print_progress=True, maxMissingRate=0.30)
        res_BEAST = res_BEAST_transect
        seasonal_BEAST =  res_BEAST.season.Y

        # Iterate over tests
        for test in tests:
            if self.verbose:
                print('Test: {}'.format(test))

            # Autocorrelation function
            # https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
            if test == 'acf':
                # Define lags and compute ACF
                lags_acf = acf(time_series - res_BEAST.trend.Y, nlags=len(time_series)-1)
                lags_acf_fit = acf(time_series - res_BEAST.trend.Y, nlags=50)

                if len(time_series) <= 50:
                    lags_acf = lags_acf_fit

                # Define range of lags
                range_lags_acf = np.arange(0, len(time_series))

                if len(time_series) <= 50:
                    range_lags_acf_fit = range_lags_acf
                
                else:
                    range_lags_acf_fit = np.arange(0, 51)

                # Initial guess for A, k, phi, offset
                initial_guess_acf = [1, 0, 12, 0, 0]  

                # Fitting sinusoidal function to ACF
                # try:
                params_acf, pcov_acf = curve_fit(seasonal_function_fitting, xdata=range_lags_acf_fit, ydata=lags_acf_fit, p0=initial_guess_acf, maxfev=1000000)
                
                # except:
                #     x_new_acf = None
                #     y_new_acf = None
                #     rsquared_ = None
                #     pvalue_ = None
                #     condition_1 = False
                #     condition_2 = False
                #     first_peak_acf = None
                #     continue

                # Define function for delta method
                f = lambda param,x : param[0] * np.exp(param[1] * x) * np.sin(2 * np.pi / param[2] * x + param[3]) + param[4] 

                # New x values
                x_new_acf = np.linspace(0, 50, 100)

                # Calculate confidence intervals
                d_ = self._delta_method(pcov_acf, params_acf, x_new_acf, f, range_lags_acf_fit, lags_acf_fit)

                # Statistical parameters
                y_new_acf = d_['y_new']
                rsquared_ = d_['rsquared']
                pvalue_ = d_['pvalue']

                # Find peaks
                peaks_acf, _ = find_peaks(y_new_acf, height=0)

                # Conditions for seasonality
                condition_1 = False # R-squared > 0.6
                condition_2 = False # first peak at lag 12 +- 1 month

                # Check for condition 1
                if rsquared_ > 0.6:
                    condition_1 = True

                # First peak
                if len(peaks_acf) > 0:
                    first_peak_acf = int(x_new_acf[peaks_acf[0]]) 

                    if first_peak_acf == 0 and len(peaks_acf) > 1:
                        first_peak_acf = int(x_new_acf[peaks_acf[1]])
                    
                    else:
                        first_peak_acf = int(x_new_acf[peaks_acf[0]]) 

                    # Check for condition 2
                    if 11 <= first_peak_acf <= 13:
                        condition_2 = True
                
                else:
                    peaks_acf_fl, _ = find_peaks(lags_acf, height=0.)

                    if len(peaks_acf_fl) > 0:
                        first_peak_acf = peaks_acf_fl[0]

                        # Check for condition 2
                        if 11 <= first_peak_acf <= 13:
                            condition_2 = True
                    
                    else:
                        first_peak_acf = None

                if self.verbose:
                    if lags_acf[first_peak_acf] > 0.3:
                        print('Result: seasonality present')
                        print('First peak at lag {}'.format(first_peak_acf))
                    
                    else:
                        print('Result: seasonality not present')

            # Fourier Analysis
            elif test == 'fourier_analysis':
                
                # Fourier transform of time series
                fft_result = fft(time_series.values)

                # Get frequencies
                frequencies = np.fft.fftfreq(len(time_series.values), 1/self.period)

                # fft_result without frequency 0
                fft_result_n = fft_result[1:]

                # Remove frequencies with amplitude below a threshold
                #print(all(element is False for element in (stats.zscore(np.abs(fft_result_n)) > self.z_score_seasonality)))
                score_list = stats.zscore(np.abs(fft_result_n)) > self.z_score_seasonality
                if all(not element for element in score_list):
                    fft_result = fft_result
                else:
                    threshold = np.min(np.abs(fft_result_n[stats.zscore(np.abs(fft_result_n)) > self.z_score_seasonality]))
                    fft_result[np.abs(fft_result) < threshold] = 0

                # Inverse Fourier transform (filetered data)
                filtered_data_fourier = np.fft.ifft(fft_result).real

                # Find peaks
                peaks_fourier, _ = find_peaks(filtered_data_fourier, height=0)

                # Find period
                period_fourier = stats.mode(np.diff(peaks_fourier)).mode

                # Condition 3
                condition_3 = False # period of 12 +- 1 month

                # Check for condition 3
                if 11 <= period_fourier <= 13:
                    condition_3 = True

                if self.verbose:
                    if len(peaks_fourier) > 0:
                        print('Result: seasonality present')
                        print('First peak at lag {}'.format(period_fourier))
                    
                    else:
                        print('Result: seasonality not present')

            # Fitting sinusoidal function to seasonal component
            elif test == 'sinus_fitting':

                res_fourier = STL(filtered_data_fourier, period=self.period).fit()

                    # rb.plot(res_BEAST)
                # except:
                #     res_BEAST = res
                #     seasonal_BEAST = res.seasonal

                # try:
                # res_fourier_BEAST = rb.beast(filtered_data_fourier, start=[time_series.index[0].year, time_series.index[0].month, time_series.index[0].day], season='harmonic', deltat='1/12 year', period='1 year', quiet=True, print_progress=False)
                # seasonal_BEAST_fourier =  res_fourier_BEAST.season.Y
                
                # except:
                res_fourier_BEAST = res_fourier
                seasonal_BEAST_fourier = res_fourier.seasonal

                # Extract seasonal component and trend
                seasonal_STL = res.seasonal
                seasonal_STL_fourier = res_fourier.seasonal
                trend_fourier = res_fourier.trend

                # Numerical dates for fitting sinusoidal function
                numeric_dates = np.array([(date - time_series.index[0]).days for date in time_series.index])

                # Initial guess for A, k, s, T, phi, offset
                initial_guess = [7, 0.001, 0, 365, 90, 0]  

                # Fitting sinusoidal function to seasonal component from original time series
                try:
                    params, _ = curve_fit(seasonal_function_skewed_fitting, xdata=numeric_dates, ydata=seasonal_STL, p0=initial_guess, maxfev=10000)
                
                except:
                    params = None

                # Fit sinusoidal function to seasonal component from BEAST decomposition
                try:
                    params_BEAST, _ = curve_fit(seasonal_function_skewed_fitting, xdata=numeric_dates, ydata=seasonal_BEAST, p0=initial_guess, maxfev=10000)
                
                except:
                    params_BEAST = None
                
                if params is None:
                    seasonal_fit_STL = None
                    peaks_seasonal_STL = None
                    minima_seasonal_STL = None
                    period_seasonal_STL = None
                    k = None
                    period_seasonal_STL_fourier = None
                    peaks_seasonal_STL_fourier = None
                    minima_seasonal_STL_fourier = None
                    amplitude_seasonal_STL = None
                    amplitude_seasonal_STL_absrange = None
                    A_fit = k_fit = s_fit = period_fit = phi_fit = offset_fit = None
                
                else:
                    # Amplitude of seasonal component
                    amplitude_seasonal_STL = params[0]
                    k = params[1]
                    A_fit, k_fit, s_fit, period_fit, phi_fit, offset_fit = params
                    seasonal_fit_STL = seasonal_function_skewed_fitting(numeric_dates, *params)            

                    # Find peaks
                    peaks_seasonal_STL, _ = find_peaks(seasonal_fit_STL, height=0)
                    peaks_seasonal_STL_fourier, _ = find_peaks(seasonal_STL_fourier, height=0)
                    minima_seasonal_STL = argrelextrema(seasonal_fit_STL, np.less)[0]
                    minima_seasonal_STL_fourier = argrelextrema(seasonal_STL_fourier, np.less)[0]

                    # Find periods
                    period_seasonal_STL = stats.mode(np.diff(peaks_seasonal_STL)).mode
                    period_seasonal_STL_fourier = stats.mode(np.diff(peaks_seasonal_STL_fourier)).mode

                    # Absolute range of amplitude
                    if abs(amplitude_seasonal_STL) > 1e6:
                        amplitude_seasonal_STL_absrange = 0.
                    else:
                        amplitude_seasonal_STL_absrange = 2*np.abs(amplitude_seasonal_STL)
                
                if params_BEAST is None:
                    seasonal_fit_BEAST = None
                    peaks_seasonal_BEAST = None
                    period_seasonal_BEAST = None
                    k_BEAST = None
                    period_seasonal_BEAST_fourier = None
                    peaks_seasonal_BEAST_fourier = None
                    minima_seasonal_BEAST = None
                    minima_seasonal_BEAST_fourier = None
                    amplitude_seasonal_BEAST = None
                    amplitude_seasonal_BEAST_absrange = None
                    A_fit_BEAST = k_fit_BEAST = s_fit_BEAST = period_fit_BEAST = phi_fit_BEAST = offset_fit_BEAST = None
                
                else:
                    # Amplitude of seasonal component
                    amplitude_seasonal_BEAST = params_BEAST[0]
                    k_BEAST = params_BEAST[1]
                    A_fit_BEAST, k_fit_BEAST, s_fit_BEAST, period_fit_BEAST, phi_fit_BEAST, offset_fit_BEAST = params_BEAST
                    seasonal_fit_BEAST = seasonal_function_skewed_fitting(numeric_dates, *params_BEAST)       

                    # Find peaks
                    peaks_seasonal_BEAST, _ = find_peaks(seasonal_fit_BEAST, height=0)
                    peaks_seasonal_BEAST_fourier, _ = find_peaks(seasonal_BEAST, height=0)
                    minima_seasonal_BEAST = argrelextrema(seasonal_fit_BEAST, np.less)[0]
                    minima_seasonal_BEAST_fourier = argrelextrema(seasonal_BEAST, np.less)[0]

                    # Find periods
                    period_seasonal_BEAST = stats.mode(np.diff(peaks_seasonal_BEAST)).mode
                    period_seasonal_BEAST_fourier = stats.mode(np.diff(peaks_seasonal_BEAST_fourier)).mode

                    # Absolute range of amplitude
                    if 2*abs(amplitude_seasonal_BEAST) > 250:
                        amplitude_seasonal_BEAST_absrange = 0.
                    
                    elif 2*abs(amplitude_seasonal_BEAST) < 5:
                        amplitude_seasonal_BEAST_absrange = 0.
                        condition_1, condition_2, condition_3 = False, False, False

                    else:
                        amplitude_seasonal_BEAST_absrange = 2*np.abs(amplitude_seasonal_BEAST)
        
        if k is None:
            relation_indian_monsoon = 'undetermined'
            behaviour_indian_monsoon = 'undetermined'
            most_common_label_peaks = None
            most_common_label_minima = None
        
        else:
            # Find relation with Indian Monsoon
            relation_indian_monsoon = self._relation_indian_monsoon(time_series, time_series_name)

            # Find behaviour with Indian Monsoon
            behaviour_indian_monsoon, most_common_label_peaks, most_common_label_minima = self._behaviour_indian_monsoon(time_series, time_series_name, peaks_seasonal_STL, seasonal_fit_STL, k=k)
        
        if k_BEAST is None:
            relation_indian_monsoon_BEAST = 'undetermined'
            behaviour_indian_monsoon_BEAST = 'undetermined'
            most_common_label_peaks_BEAST = None
            most_common_label_minima_BEAST = None
        
        else:
            # Find relation with Indian Monsoon
            relation_indian_monsoon_BEAST = self._relation_indian_monsoon(time_series, time_series_name)

            # Find behaviour with Indian Monsoon
            behaviour_indian_monsoon_BEAST, most_common_label_peaks_BEAST, most_common_label_minima_BEAST = self._behaviour_indian_monsoon(time_series, time_series_name, peaks_seasonal_BEAST, seasonal_fit_BEAST, k=k_BEAST)
        
        # Save in island_info
        results_seasonality_dict = {'lags_acf': lags_acf,
                                    'range_lags_acf': range_lags_acf,
                                    'first_peak_acf': first_peak_acf,
                                    'filtered_data_fourier': filtered_data_fourier,
                                    'peaks_fourier': peaks_fourier,
                                    'period_fourier': period_fourier,
                                    'seasonal_component_STL': seasonal_STL,
                                    'seasonal_component_STL_fourier': seasonal_STL_fourier,
                                    'trend_fourier': trend_fourier,
                                    'amplitude_seasonal_STL': amplitude_seasonal_STL,
                                    'amplitude_seasonal_STL_absrange': amplitude_seasonal_STL_absrange,
                                    'seasonal_fit_STL': seasonal_fit_STL,
                                    'peaks_seasonal_STL': peaks_seasonal_STL,
                                    'peaks_seasonal_STL_fourier': peaks_seasonal_STL_fourier,
                                    'minima_seasonal_STL': minima_seasonal_STL,
                                    'minima_seasonal_STL_fourier': minima_seasonal_STL_fourier,
                                    'period_seasonal_STL': period_seasonal_STL,
                                    'period_seasonal_STL_fourier': period_seasonal_STL_fourier,
                                    'relation_indian_monsoon': relation_indian_monsoon,
                                    'behaviour_indian_monsoon': behaviour_indian_monsoon,
                                    'seasonality_indian_monsoon_peaks': most_common_label_peaks,
                                    'seasonality_indian_monsoon_minima': most_common_label_minima,
                                    'fit_params_STL': {'A': A_fit, 'k': k_fit, 's': s_fit, 'period': period_fit, 'phi': phi_fit, 'offset': offset_fit},
                                    'seasonal_component_BEAST': seasonal_BEAST,
                                    'seasonal_component_BEAST_fourier': seasonal_BEAST_fourier,
                                    'amplitude_seasonal_BEAST': amplitude_seasonal_BEAST,
                                    'amplitude_seasonal_BEAST_absrange': amplitude_seasonal_BEAST_absrange,
                                    'seasonal_fit_BEAST': seasonal_fit_BEAST,
                                    'peaks_seasonal_BEAST': peaks_seasonal_BEAST,
                                    'peaks_seasonal_BEAST_fourier': peaks_seasonal_BEAST_fourier,
                                    'minima_seasonal_BEAST': minima_seasonal_BEAST,
                                    'minima_seasonal_BEAST_fourier': minima_seasonal_BEAST_fourier,
                                    'period_seasonal_BEAST': period_seasonal_BEAST,
                                    'period_seasonal_BEAST_fourier': period_seasonal_BEAST_fourier,
                                    'relation_indian_monsoon_BEAST': relation_indian_monsoon_BEAST,
                                    'behaviour_indian_monsoon_BEAST': behaviour_indian_monsoon_BEAST,
                                    'seasonality_indian_monsoon_peaks_BEAST': most_common_label_peaks_BEAST,
                                    'seasonality_indian_monsoon_minima_BEAST': most_common_label_minima_BEAST,
                                    'conditions_seasonality': {'condition_1': condition_1, 'condition_2': condition_2, 'condition_3': condition_3}, 
                                    'fit_curve_acf': {'x_new': x_new_acf, 'y_new': y_new_acf, 'rsquared': rsquared_, 'pvalue': pvalue_},                                                                             
                                    'fit_params_BEAST': {'A': A_fit_BEAST, 'k': k_fit_BEAST, 's': s_fit_BEAST, 'period': period_fit_BEAST, 'phi': phi_fit_BEAST, 'offset': offset_fit_BEAST}
                                    }

        self.island_info['timeseries_analysis'][time_series_name]['seasonality'] = results_seasonality_dict
        
        return results_seasonality_dict

    def test_stationarity(self, time_series, time_series_name):

        if self.verbose:
            print('Checking for STATIONARITY')

        # List of tests (Augmented Dickey-Fuller and KPSS)
        tests = [adfuller, kpss]
        name_tests = ['Augmented Dickey-Fuller', 'KPSS']

        # Initialise empty list to store results
        stationarity = []

        # Iterate over tests
        for test, name_test in zip(tests, name_tests):
            if self.verbose:
                print('Test: {}'.format(name_test))
            
            # Perform test
            results = test(time_series.values)
            p_value = results[1]

            if name_test == 'KPSS':
                p_value = 1 - p_value

            if p_value > self.alpha:
                stationarity.append('not stationary')
            else:
                stationarity.append('stationary')
        
        # Check if all tests agree
        result_consistent = self._consistency_check(stationarity)

        # If tests do not agree, assign 'inconsistent' to result_consistent
        if result_consistent is None:
            result_consistent = 'inconsistent'
        
            if stationarity[0] == 'stationary':
                result_consistent = 'difference stationary'

            else:
                result_consistent = 'trend stationary'

        if self.verbose:
            print('Result: {}'.format(result_consistent))

        # Assign consistent result to stationarity_result
        stationarity_result = result_consistent

        # Save in island_info
        results_stationarity_dict = {'stationarity_result': stationarity_result}
        self.island_info['timeseries_analysis'][time_series_name]['stationarity'] = results_stationarity_dict

        return results_stationarity_dict

    def plot_characterisation(self, time_series, time_series_name, transect):
        
        # Create figure
        fig, ax = plt.subplots(3, 2)
        #fig_temp, ax_temp = plt.subplots()

        # Shortcut for result dictionary
        results_analysis = self.island_info['timeseries_analysis'][time_series_name]

        # Plot transect
        plot_shoreline_transects(self.island_info, ax=ax[0, 0], transect_plot=transect)
        ax[0, 0].set_title('Transect {} location'.format(transect))

        # Plot trend analysis
        ax[0, 1].plot(time_series.index, time_series.values, color='k')
        ax[0, 1].plot(results_analysis['trend']['trend_component_STL'].index, results_analysis['trend']['trend_component_STL'].values, color='gold', label=results_analysis['trend']['trend_result'])
        for bkp in results_analysis['trend']['trend_bkp_dates']:
            ax[0, 1].axvline(bkp, color='b', linestyle='--', alpha=0.3)
        for bkp in results_analysis['trend']['trend_bkp_dates_significant']:
            ax[0, 1].axvline(bkp, color='b', linestyle='--')
        ax[0, 1].plot(time_series.index, results_analysis['trend']['trend_fitted_line'], color='g')
        ax[0, 1].legend(fontsize=10)
        ax[0, 1].set_title('Trend analysis')

        # Plot trend analysis
        # ax_temp.plot(time_series.index, time_series.values, color='k')
        # ax_temp.plot(results_analysis['trend']['trend_component_STL'].index, results_analysis['trend']['trend_component_STL'].values, color='gold', label=results_analysis['trend']['trend_result'])
        # for bkp in results_analysis['trend']['trend_bkp_dates']:
        #     ax_temp.axvline(bkp, color='b', linestyle='--', alpha=0.3)
        # for bkp in results_analysis['trend']['trend_bkp_dates_significant']:
        #     ax_temp.axvline(bkp, color='b', linestyle='--')
        # ax_temp.plot(time_series.index, results_analysis['trend']['trend_fitted_line'], color='g')
        # ax_temp.legend(fontsize=10)
        # ax_temp.set_title('Trend analysis: {}'.format(time_series_name))

        # Plot ACF
        ax[1, 0].stem(results_analysis['seasonality']['range_lags_acf'], results_analysis['seasonality']['lags_acf'], linefmt='k', basefmt='k', markerfmt='k.')
        if results_analysis['seasonality']['fit_curve_acf']['x_new'] is not None:
            ax[1, 0].plot(results_analysis['seasonality']['fit_curve_acf']['x_new'], results_analysis['seasonality']['fit_curve_acf']['y_new'], color='g', label='R-squared: {:.2f}'.format(results_analysis['seasonality']['fit_curve_acf']['rsquared']))
        if results_analysis['seasonality']['first_peak_acf'] is not None:
            ax[1, 0].axvline(results_analysis['seasonality']['first_peak_acf'], color='r', linestyle='--', label='period {}'.format(results_analysis['seasonality']['first_peak_acf']))
        ax[1, 0].legend(fontsize=10)
        ax[1, 0].set_title('Autocorrelation function')

        # Plot Fourier analysis
        ax[1, 1].plot(time_series.index, time_series.values, color='k')
        ax[1, 1].plot(time_series.index, results_analysis['seasonality']['filtered_data_fourier'], color='g', label='period {}'.format(results_analysis['seasonality']['period_fourier']))
        if results_analysis['seasonality']['peaks_fourier'] is not None:
            for peak in results_analysis['seasonality']['peaks_fourier']:
                ax[1, 1].axvline(time_series.index[peak], color='r', linestyle='--')
        ax[1, 1].legend(fontsize=10)
        ax[1, 1].set_title('Fourier reconstruction')

        # Plot fitting sinusoidal function
        ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_component_STL'].values, color='purple', label='seasonal component')
        if results_analysis['seasonality']['peaks_seasonal_STL'] is not None:
            for peak in results_analysis['seasonality']['peaks_seasonal_STL']:
                ax[2, 0].axvline(time_series.index[peak], color='r', linestyle='--')
        
        if results_analysis['seasonality']['minima_seasonal_STL'] is not None:
            for minimum in results_analysis['seasonality']['minima_seasonal_STL']:
                ax[2, 0].axvline(time_series.index[minimum], color='pink', linestyle='--')        
        for year in range(2010, 2023):
            if year == 2010:
                ax[2, 0].axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2, label='Northeast Monsoon')
                ax[2, 0].axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2, label='Southwest Monsoon')
            else:
                ax[2, 0].axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2)
                ax[2, 0].axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2)
        if results_analysis['seasonality']['seasonal_fit_STL'] is not None:
            ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_fit_STL'], color='g', label='sinusoidal fit (period {})'.format(results_analysis['seasonality']['period_seasonal_STL']))
        ax[2, 0].legend(fontsize=10)
        ax[2, 0].set_xlim(time_series.index[0], time_series.index[-1])
        ax[2, 0].set_title('Seasonal component from original data')

        # Plot Fourier filtered seasonal component
        ax[2, 1].plot(time_series.index, results_analysis['seasonality']['seasonal_component_STL_fourier'], color='purple', label='period {}'.format(results_analysis['seasonality']['period_seasonal_STL_fourier']))
        if results_analysis['seasonality']['peaks_seasonal_STL_fourier'] is not None:
            for peak in results_analysis['seasonality']['peaks_seasonal_STL_fourier']:
                ax[2, 1].axvline(time_series.index[peak], color='r', linestyle='--')
        ax[2, 1].legend(fontsize=10)
        ax[2, 1].set_title('Seasonal component from filtered data')

        # Add title with conditions
        fig.suptitle(results_analysis['seasonality']['conditions_seasonality'])

        plt.show()
        # plt.close(fig)

        # Plot with BEAST results
        # Create figure
        fig, ax = plt.subplots(3, 2)
        #fig_temp, ax_temp = plt.subplots()

        # Shortcut for result dictionary
        results_analysis = self.island_info['timeseries_analysis'][time_series_name]

        # Plot transect
        plot_shoreline_transects(self.island_info, ax=ax[0, 0], transect_plot=transect)
        ax[0, 0].set_title('Transect location')

        # Plot trend analysis
        ax[0, 1].plot(time_series.index, time_series.values, color='k')
        ax[0, 1].plot(results_analysis['trend']['trend_component_STL'].index, results_analysis['trend']['trend_component_BEAST'], color='gold', label=results_analysis['trend']['trend_result'])
        ax[0, 1].plot(time_series.index, results_analysis['trend']['trend_fitted_line_BEAST'], color='g')
        ax[0, 1].legend(fontsize=10)
        ax[0, 1].set_title('Trend analysis BEAST')

        # Plot ACF
        ax[1, 0].stem(results_analysis['seasonality']['range_lags_acf'], results_analysis['seasonality']['lags_acf'], linefmt='k', basefmt='k', markerfmt='k.')
        if results_analysis['seasonality']['first_peak_acf'] is not None:
            ax[1, 0].axvline(results_analysis['seasonality']['first_peak_acf'], color='r', linestyle='--', label='period {}'.format(results_analysis['seasonality']['first_peak_acf']))
        ax[1, 0].legend(fontsize=10)
        ax[1, 0].set_title('Autocorrelation function')

        # Plot Fourier analysis
        ax[1, 1].plot(time_series.index, time_series.values, color='k')
        ax[1, 1].plot(time_series.index, results_analysis['seasonality']['filtered_data_fourier'], color='g', label='period {}'.format(results_analysis['seasonality']['period_fourier']))
        if results_analysis['seasonality']['peaks_fourier'] is not None:
            for peak in results_analysis['seasonality']['peaks_fourier']:
                ax[1, 1].axvline(time_series.index[peak], color='r', linestyle='--')
        ax[1, 1].legend(fontsize=10)
        ax[1, 1].set_title('Fourier reconstruction')

        # Plot fitting sinusoidal function
        ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_component_BEAST'], color='purple', label='seasonal component')
        if results_analysis['seasonality']['seasonal_fit_BEAST'] is not None:
            ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_component_BEAST'], color='purple', label='seasonal component')
        if results_analysis['seasonality']['peaks_seasonal_BEAST'] is not None:
            for peak in results_analysis['seasonality']['peaks_seasonal_BEAST']:
                ax[2, 0].axvline(time_series.index[peak], color='r', linestyle='--')
        if results_analysis['seasonality']['minima_seasonal_BEAST'] is not None:
            for minimum in results_analysis['seasonality']['minima_seasonal_BEAST']:
                ax[2, 0].axvline(time_series.index[minimum], color='pink', linestyle='--')        
        for year in range(2010, 2023):
            if year == 2010:
                ax[2, 0].axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2, label='Northeast Monsoon')
                ax[2, 0].axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2, label='Southwest Monsoon')
            else:
                ax[2, 0].axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2)
                ax[2, 0].axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2)
        if results_analysis['seasonality']['seasonal_fit_BEAST'] is not None:
            ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_fit_BEAST'], color='g', label='sinusoidal fit (period {})'.format(results_analysis['seasonality']['period_seasonal_BEAST']))
        ax[2, 0].legend(fontsize=10)
        ax[2, 0].set_xlim(time_series.index[0], time_series.index[-1])
        ax[2, 0].set_title('Seasonal component from original data')

        # Plot Fourier filtered seasonal component
        ax[2, 1].plot(time_series.index, results_analysis['seasonality']['seasonal_component_STL_fourier'], color='purple', label='period {}'.format(results_analysis['seasonality']['period_seasonal_STL_fourier']))
        if results_analysis['seasonality']['peaks_seasonal_STL_fourier'] is not None:
            for peak in results_analysis['seasonality']['peaks_seasonal_STL_fourier']:
                ax[2, 1].axvline(time_series.index[peak], color='r', linestyle='--')
        ax[2, 1].legend(fontsize=10)
        ax[2, 1].set_title('Seasonal component from filtered data')

        plt.show()
        # plt.close(fig)

    def make_plots(self):

        # Create figure
        fig, ax = plt.subplots(nrows=2, ncols=2)

        # Polygon of the reference shoreline
        polygon_reference_shoreline = shapely.geometry.Polygon(self.reference_shoreline)
        
        # Short cut for time series analysis results
        ts_analysis_results = self.island_info['timeseries_analysis']

        # Get transect keys 
        key_transects = [int((key).split('_')[3]) for key in ts_analysis_results.keys()]

        # Plot reference shoreline in all subplots
        for i in range(2):
            for j in range(2):
                ax[i, j].plot(self.reference_shoreline[:, 0], self.reference_shoreline[:, 1], 'k-', zorder=1)
                ax[i, j].set_xlabel('Longitude', fontsize=15)
                ax[i, j].set_ylabel('Latitude', fontsize=15)

        # Get intersections between transects and reference shoreline
        intersections = [polygon_reference_shoreline.exterior.intersection(shapely.geometry.LineString(self.transects_geo[key_transect])) for key_transect in key_transects]

        # x and y coordinates of intersections
        x_intersections = []
        y_intersections = []
        for intersection in intersections:
            if type(intersection) == shapely.geometry.MultiPoint:
                # Take the first point of the MultiPoint
                x_intersections.append(intersection.geoms[0].x)
                y_intersections.append(intersection.geoms[0].y)
            
            elif type(intersection) == shapely.geometry.collection.GeometryCollection:
                x_intersections.append(None)
                y_intersections.append(None)
            
            elif type(intersection) == shapely.geometry.LineString:
                x_intersections.append(None)
                y_intersections.append(None)

            else:
                x_intersections.append(intersection.x)
                y_intersections.append(intersection.y)

        #x_intersections = [intersection.x for intersection in intersections]
        #y_intersections = [intersection.y for intersection in intersections]

        # Trend splot and results (colorbar = trend slope and symbol = trend result)
        c_trend = [ts_analysis_results[val]['trend']['trend_slope'] for val in ts_analysis_results.keys()]
        symbols_trend = [ts_analysis_results[val]['trend']['trend_result'] for val in ts_analysis_results.keys()]

        # Seasonality amplitude results (colorbar = seasonality amplitude)
        #c_seasonality_amplitude = [ts_analysis_results[val]['seasonality']['amplitude_seasonal_STL'] for val in ts_analysis_results.keys()]
        c_seasonality_amplitude = [ts_analysis_results[val]['seasonality']['amplitude_seasonal_STL_absrange'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= self.n_seasonality else 0 for val in ts_analysis_results.keys()]
        c_seasonality_amplitude_BEAST = [ts_analysis_results[val]['seasonality']['amplitude_seasonal_BEAST_absrange'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= self.n_seasonality else 0 for val in ts_analysis_results.keys()]

        # Seasonality peaks and minima results
        c_seasonality_peaks = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_peaks'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= self.n_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
        c_seasonality_minima = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_minima'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= self.n_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
        c_seasonality_peaks_BEAST = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_peaks_BEAST'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= self.n_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
        c_seasonality_minima_BEAST = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_minima_BEAST'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= self.n_seasonality else 'undetermined' for val in ts_analysis_results.keys()]

        # Seasonality peak months
        # c_seasonality_peak_months = [ts_analysis_results[val]['seasonality']['seasonality_peak_months'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= 1 else 'undetermined' for val in ts_analysis_results.keys()]

        # DataFrame for trend
        df_trend = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'linear regression slope': c_trend, 'statistical significance': symbols_trend, 's': 100})

        # DataFrame for seasonality
        df_seasonality_amplitude = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'amplitude': c_seasonality_amplitude, 's': 100})
        df_seasonality_peaks = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'peak period': c_seasonality_peaks, 's': 100})
        df_seasonality_minima = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'minimum period': c_seasonality_minima, 's': 100})
        df_seasonality_amplitude_BEAST = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'amplitude': c_seasonality_amplitude_BEAST, 's': 100})
        df_seasonality_peaks_BEAST = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'peak period': c_seasonality_peaks_BEAST, 's': 100})
        df_seasonality_minima_BEAST = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'minimum period': c_seasonality_minima_BEAST, 's': 100})

        # Trend plot
        sns.scatterplot(data=df_trend, x='x', y='y', hue='linear regression slope', style='statistical significance', s=100, ax=ax[0, 0], palette=sns.color_palette("RdYlGn", as_cmap=True), zorder=2, edgecolor='k')

        # Seasonality amplitude plot
        sns.scatterplot(data=df_seasonality_amplitude, x='x', y='y', hue='amplitude', s=100, ax=ax[0, 1], palette=sns.color_palette("viridis"), zorder=2, edgecolor='k')

        # Seasonality peaks plot
        sns.scatterplot(data=df_seasonality_peaks, x='x', y='y', hue='peak period', s=100, ax=ax[1, 0], palette=sns.color_palette("Set2"), zorder=2, edgecolor='k')

        # Seasonality minima plot
        sns.scatterplot(data=df_seasonality_minima, x='x', y='y', hue='minimum period', s=100, ax=ax[1, 1], palette=sns.color_palette("Set2"), zorder=2, edgecolor='k')

        # Set titles
        ax[0, 0].set_title('Trend amplitude', fontsize=15)
        ax[0, 1].set_title('Seasonality amplitude', fontsize=15)
        ax[1, 0].set_title('Seasonality peak period', fontsize=15)
        ax[1, 1].set_title('Seasonality minimum period', fontsize=15)
        fig.suptitle('Island: {}, {}'.format(self.island_info['general_info']['island'], self.island_info['general_info']['country']), fontsize=20)

        #plt.show()

        # BEAST plot
        fig, ax = plt.subplots(nrows=2, ncols=2)
        for i in range(2):
            for j in range(2):
                ax[i, j].plot(self.reference_shoreline[:, 0], self.reference_shoreline[:, 1], 'k-', zorder=1)
                ax[i, j].set_xlabel('Longitude', fontsize=15)
                ax[i, j].set_ylabel('Latitude', fontsize=15)

        # Trend plot
        sns.scatterplot(data=df_trend, x='x', y='y', hue='linear regression slope', style='statistical significance', s=100, ax=ax[0, 0], palette=sns.color_palette("RdYlGn", as_cmap=True), zorder=2, edgecolor='k')

        # Seasonality amplitude plot
        sns.scatterplot(data=df_seasonality_amplitude_BEAST, x='x', y='y', hue='amplitude', s=100, ax=ax[0, 1], palette=sns.color_palette("viridis"), zorder=2, edgecolor='k')

        # Seasonality peaks plot
        sns.scatterplot(data=df_seasonality_peaks_BEAST, x='x', y='y', hue='peak period', s=100, ax=ax[1, 0], palette=sns.color_palette("Set2"), zorder=2, edgecolor='k')

        # Seasonality minima plot
        sns.scatterplot(data=df_seasonality_minima_BEAST, x='x', y='y', hue='minimum period', s=100, ax=ax[1, 1], palette=sns.color_palette("Set2"), zorder=2, edgecolor='k')

        # Set titles
        ax[0, 0].set_title('Trend amplitude', fontsize=15)
        ax[0, 1].set_title('Seasonality amplitude BEAST', fontsize=15)
        ax[1, 0].set_title('Seasonality peak period BEAST', fontsize=15)
        ax[1, 1].set_title('Seasonality minimum period BEAST', fontsize=15)
        fig.suptitle('Island: {}, {}'.format(self.island_info['general_info']['island'], self.island_info['general_info']['country']), fontsize=20)

        plt.show()
        # plt.close(fig)

        fig_temp, ax_temp = plt.subplots(1, 3, figsize=(20, 5))
        # fig_temp, ax_temp = plt.subplots(3, 1, figsize=(12, 20))

        # Plot reference shoreline
        for ax in ax_temp:
            ax.plot(self.reference_shoreline[:, 0], self.reference_shoreline[:, 1], 'k-', zorder=1)
            ax.axis('off')
        
        # Plot seasonality minimum period (BEAST)
        # Polygon of the reference shoreline
        polygon_reference_shoreline = shapely.geometry.Polygon(self.reference_shoreline)

        # For legend
        dict_colours_labels = {'sw_monsoon': '#58508d',
                                'ne_monsoon': '#bc5090', 
                                'transition_ne_sw': '#ff6361',
                                'transition_sw_ne': '#ffa600',
                                'undetermined': '#003f5c'}
        
        dict_names_labels = {'sw_monsoon': 'SW Monsoon',
                                'ne_monsoon': 'NE Monsoon', 
                                'transition_ne_sw': 'Transition from NE to SW Monsoon',
                                'transition_sw_ne': 'Transition from SW to NE Monsoon',
                                'undetermined': 'Undetermined / No seasonality'}    
        
        # for idx_cs, cs in enumerate(c_seasonality_amplitude_BEAST):
        spa = ax_temp[2].scatter(x_intersections, y_intersections, s=100, c=c_seasonality_amplitude_BEAST, cmap='viridis', edgecolor='k')
        # add colorbar
        cbar = fig_temp.colorbar(spa, ax=ax_temp)
        cbar.ax.tick_params(labelsize=15)
        cbar.set_label('Amplitude (m)', fontsize=15)

        # for idx_cs, cs in enumerate(c_seasonality_peaks_BEAST):
        #     ax_temp[1].scatter(x_intersections[idx_cs], y_intersections[idx_cs], s=100, c=dict_colours_labels[cs], edgecolor='k')
        
        # for idx_cs, cs in enumerate(c_seasonality_minima_BEAST):
        #     ax_temp[0].scatter(x_intersections[idx_cs], y_intersections[idx_cs], s=100, c=dict_colours_labels[cs], edgecolor='k')
        
        # We create a colormar from our list of colors
        cm = mcolors.ListedColormap(dict_colours_labels.values())
        custom_palette = {label: color for label, color in dict_colours_labels.items()}

        sns.scatterplot(data=df_seasonality_peaks, x='x', y='y', hue='peak period', s=100, ax=ax_temp[1], palette=custom_palette, zorder=2, edgecolor='k', legend=False)
        sns.scatterplot(data=df_seasonality_minima, x='x', y='y', hue='minimum period', s=100, ax=ax_temp[0], palette=custom_palette, zorder=2, edgecolor='k', legend=False)

        # Create custom legend patches
        handles = [mpatches.Patch(color=dict_colours_labels[key], label=dict_names_labels[key]) for key in dict_colours_labels.keys()]

        # Add legend to the plot
        # ax_temp[0].legend(handles=handles, loc='center', fontsize=12)
        ax_temp[1].legend(handles=handles, bbox_to_anchor=(-1.00, 1), loc='upper right', fontsize=12)

        # Set titles
        ax_temp[0].set_title('Minimum period', fontsize=20)
        ax_temp[1].set_title('Peak period', fontsize=20)  
        ax_temp[2].set_title('Differential Amplitude', fontsize=20)
        
        fig_temp.savefig('figures//seasonality_BEAST//{}_{}_seasonality_BEAST.png'.format(self.island, self.country), dpi=300, bbox_inches='tight')
        # plt.close(fig_temp)

    def _create_aggretation_dict(self, dict_results):

        # Initialise variables
        current_group = 0
        dict_results_agg = {}
        sorted_transect = []
        last_transect = max(dict_results.keys())

        # Iterate over dictionary
        for key, value in dict_results.items():

            # First key for the current group
            agg_temp = [key]

            # Skip if key is already in sorted_transect
            if key in sorted_transect:
                continue
            
            # Add key to sorted_transect
            sorted_transect.append(key)

            # List of keys after the current key
            keys_after = [k for k in dict_results.keys() if k > key]

            # Iterate over keys after the current key
            for k in keys_after:

                # If same behaviour, add to agg_temp
                if value == dict_results[k]:
                    # Add to agg_temp and sorted_transect
                    agg_temp.append(k)
                    sorted_transect.append(k)

                    # If last key, add to dict_results_agg
                    if k == last_transect:
                        current_group += 1
                        dict_results_agg[current_group] = {'behaviour': value, 'transects': agg_temp}
                        break
                
                # If different behaviour, add group to dict_results_agg
                else:
                    current_group += 1
                    dict_results_agg[current_group] = {'behaviour': value, 'transects': agg_temp}
                    break
        
        # Check if first and last group are the same
        if dict_results_agg[1]['behaviour'] == dict_results_agg[current_group]['behaviour']:
            dict_results_agg[1]['transects'] = dict_results_agg[current_group]['transects'] + dict_results_agg[1]['transects']
            del dict_results_agg[current_group]
        
        return dict_results_agg
    
    def _plot_aggregation(self, dict_results_agg, label, ax):

        ax.set_title(label, fontsize=20)
        ax.set_xlabel('Longitude', fontsize=20)
        ax.set_ylabel('Latitude', fontsize=20)

        # Plot reference shoreline
        ax.plot(self.reference_shoreline[:, 0], self.reference_shoreline[:, 1], 'k-', zorder=1)

        # Polygon of the reference shoreline
        polygon_reference_shoreline = shapely.geometry.Polygon(self.reference_shoreline)

        # For legend
        list_legend = []
        dict_colours_labels = {'sw_monsoon': 'blue',
                                'ne_monsoon': 'orange', 
                                'transition_ne_sw': 'green',
                                'transition_sw_ne': 'red',
                                'undetermined': 'purple'}
        
        dict_names_labels = {'sw_monsoon': 'SW Monsoon',
                                'ne_monsoon': 'NE Monsoon', 
                                'transition_ne_sw': 'Transition from NE to SW Monsoon',
                                'transition_sw_ne': 'Transition from SW to NE Monsoon',
                                'undetermined': 'Undetermined'}    
            
        # Create LineString for each group
        for key, value in dict_results_agg.items():

            # Skip short groups
            if len(value['transects']) < 5:
                continue

            key_transects = value['transects']

            # Get intersections between transects and reference shoreline
            intersections = [polygon_reference_shoreline.exterior.intersection(shapely.geometry.LineString(self.transects_geo[key_transect])) for key_transect in key_transects]

            # x and y coordinates of intersections
            x_intersections = []
            y_intersections = []
            for intersection in intersections:
                if type(intersection) == shapely.geometry.MultiPoint:
                    # Take the first point of the MultiPoint
                    x_intersections.append(intersection.geoms[0].x)
                    y_intersections.append(intersection.geoms[0].y)
                
                elif type(intersection) == shapely.geometry.LineString:
                    if intersection.is_empty:
                        continue
                        
                    else:
                        x, y = intersection.xy
                        x_intersections.append(x[0])
                        y_intersections.append(y[0])
                else:
                    x_intersections.append(intersection.x)
                    y_intersections.append(intersection.y)

            #x_intersections = [intersection.x for intersection in intersections]
            #y_intersections = [intersection.y for intersection in intersections]

            # LineString
            try:
                line = shapely.geometry.LineString(zip(x_intersections, y_intersections))
            
            except:
                continue

            if line.is_empty:
                continue
                
            if line is None:
                continue

            # Plot LineString
            if value['behaviour'] not in list_legend:
                try:
                    gpd.GeoSeries(line).plot(ax=ax, color=dict_colours_labels[value['behaviour']], zorder=2, linewidth=5, label=dict_names_labels[value['behaviour']])
                    list_legend.append(value['behaviour'])
                except:
                    continue
            else:
                try:
                    gpd.GeoSeries(line).plot(ax=ax, color=dict_colours_labels[value['behaviour']], zorder=2, linewidth=5)
                except:
                    continue

    def aggregate_results(self):

        # Shortcut for time series analysis results
        ts_analysis_results = self.island_info['timeseries_analysis']

        # List of minima and peaks results
        minima_results = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_minima'] for val in ts_analysis_results.keys()]
        peaks_results = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_peaks'] for val in ts_analysis_results.keys()]

        # List of transects
        transects = [int((key).split('_')[3]) for key in ts_analysis_results.keys()]

        # Create dictionary for results/transects
        dict_minima_results = {transect: minima_results[i] for i, transect in enumerate(transects)}
        dict_peaks_results = {transect: peaks_results[i] for i, transect in enumerate(transects)}
        
        dict_results_agg_minima = self._create_aggretation_dict(dict_minima_results)
        dict_results_agg_peaks = self._create_aggretation_dict(dict_peaks_results)

        # Plot aggregation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        self._plot_aggregation(dict_results_agg_minima, label='Peak of erosion', ax=ax1)
        self._plot_aggregation(dict_results_agg_peaks, label='Peak of accretion', ax=ax2)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        if len(labels1) > len(labels2):
            handles, labels = handles1, labels1
        else:
            handles, labels = handles2, labels2

        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), fontsize=15)
        fig.suptitle('Island: {}, {}'.format(self.island_info['general_info']['island'], self.island_info['general_info']['country']), fontsize=20)
        plt.show()
        # plt.close(fig)

        # Save in island_info
        self.island_info['timeseries_aggregation'] = {'dict_results_agg_minima': dict_results_agg_minima, 'dict_results_agg_peaks': dict_results_agg_peaks}

    def characterisation(self, time_series, time_series_name, transect, res_BEAST_transect):

        # Presence of trend
        results_trend_dict = self.test_presence_of_trend(time_series, time_series_name, res_BEAST_transect)

        # Presence of seasonality
        results_seasonality_dict = self.test_presence_of_seasonality(time_series, time_series_name, res_BEAST_transect)

        # Stationarity
        results_stationarity_dict = self.test_stationarity(time_series, time_series_name)
        
        # Save results in island_info
        dict_all_results = {'trend': results_trend_dict, 'seasonality': results_seasonality_dict, 'stationarity': results_stationarity_dict}

        # Plotting characterisation for this transect
        if self.plot_results_transect:
            # if transect in [10, 11]:
            print('Plotting characterisation')
            self.plot_characterisation(time_series, time_series_name, transect)
        
        return dict_all_results

    def initialise_time_series_analysis(self):
        if 'timeseries_analysis' not in self.island_info.keys() or self.overwrite_all:
            self.island_info['timeseries_analysis'] = {}

    def main(self):

        # Initialise time series analysis
        self.initialise_time_series_analysis()

        if self.plot_only and not self.overwrite:
            # Make plots for the whole island
            self.make_plots()

            return self.island_info

        # else:
        
        if 'timeseries_analysis' in self.island_info.keys() and not self.overwrite:
            return self.island_info

        print('\n-------------------------------------------------------------------')
        print('Time series analysis')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        path_res_BEAST_only = os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(self.island, self.country), '{}_{}_results_BEAST_only.data'.format(self.island, self.country))
        if os.path.exists(path_res_BEAST_only):
            with open(path_res_BEAST_only, 'rb') as file:
                res_BEAST_dict_only = pickle.load(file)

        if self.transect_to_plot is not None:

            time_series_name = 'coastline_position_transect_{}_waterline'.format(self.transect_to_plot)

            # if 'monthly' in self.island_info['timeseries_preprocessing']['optimal time period']['dict_timeseries'][time_series_name].keys():
                # time_series = self.island_info['timeseries_preprocessing']['optimal time period']['dict_timeseries'][time_series_name]['monthly'][time_series_name]
                # print(time_series)
            time_series = self.island_info['timeseries_preprocessing']['optimal time period']['dict_coastline_timeseries'][time_series_name]
            res_BEAST_transect = res_BEAST_dict_only[time_series_name]
            self.plot_results_transect = True

            _ = self.characterisation(time_series, time_series_name, self.transect_to_plot, res_BEAST_transect)
            save_island_info(self.island_info)
            return self.island_info

        # If file with results already exists, load it
        path_res_BEAST = os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(self.island, self.country), '{}_{}_results_BEAST.data'.format(self.island, self.country))
        
        if os.path.exists(path_res_BEAST):
            with open(path_res_BEAST, 'rb') as file:
                try:
                    res_BEAST_dict = pickle.load(file)
                except:
                    os.remove(path_res_BEAST)
                    res_BEAST_dict = {}

            # self.island_info['timeseries_analysis'] = res_BEAST_dict
        
        # Empty dictionary for results BEAST
        else:
            # if 'timeseries_analysis' in self.island_info.keys():
            #     res_BEAST_dict = self.island_info['timeseries_analysis']
            
            # else:
            res_BEAST_dict = {}
        


        self.island_info['timeseries_analysis'] = self.island_info.get('timeseries_analysis', {})
        self.island_info['timeseries_analysis'] =  res_BEAST_dict

        # Iterate over transects
        for transect in tqdm(self.transects, desc='Transect'):
        # for transect in self.transects:
        # for transect in tqdm(np.arange(37, list(self.transects)[-1]+1)):

            time_series_name = 'coastline_position_transect_{}_waterline'.format(transect)
            if time_series_name not in self.island_info['timeseries_preprocessing']['optimal time period']['dict_coastline_timeseries'].keys():
                print('Transect {} not in dictionary'.format(transect))
                continue

            time_series = self.island_info['timeseries_preprocessing']['optimal time period']['dict_coastline_timeseries'][time_series_name]

            # Time series quality check
            if len(time_series) < 25:
                continue
            
            # Initialise empty dictionary for time series analysis for this transect
            if time_series_name not in self.island_info['timeseries_analysis'].keys():
                self.island_info['timeseries_analysis'][time_series_name] = {}
            
            if not self.overwrite_transect:
                if time_series_name in res_BEAST_dict.keys() and res_BEAST_dict[time_series_name] != {}:
                        continue
                else:
                    self.island_info['timeseries_analysis'][time_series_name] = {}
            else:
                self.island_info['timeseries_analysis'][time_series_name] = {}
            
            # BEAST results for this transect
            if time_series_name in res_BEAST_dict_only.keys():
                result_BEAST_transect = res_BEAST_dict_only[time_series_name]
            
            else:
                continue
            
            # Characterisation of the time series
            dict_all_results = self.characterisation(time_series, time_series_name, transect, result_BEAST_transect)

            # Populate dictionary with results BEAST
            res_BEAST_dict[time_series_name] = dict_all_results

            # Save dictionary
            with open(path_res_BEAST, 'wb') as file:
                pickle.dump(res_BEAST_dict, file)

        self.island_info['timeseries_analysis'] = res_BEAST_dict
        save_island_info(self.island_info)

        # Make plots for the whole island
        self.make_plots()

        # Aggregate results
        # self.aggregate_results()

        return self.island_info