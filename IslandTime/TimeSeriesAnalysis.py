"""
This module contains tools to make time series analysis on island time series.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import warnings
warnings.filterwarnings('ignore')
from IslandTime import plot_shoreline_transects
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

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class TimeSeriesAnalysis:
    def __init__(self, island_info, period=12, alpha=0.05, 
                 model_change_detection='l2', width_change_detection=12, z_score_seasonality=1.5,
                 verbose=False, debug=False, overwrite=True, plot_results_transect=False, 
                 plot_only=False, threshold_k=0.0009):
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

    def test_presence_of_trend(self, time_series, time_series_name):
        
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
            results = test(time_series, alpha=self.alpha)
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

        # Fit line equation: y = mx + b
        line = trend_slope * numeric_dates + intercept

        # Save in island_info
        self.island_info['timeseries_analysis'][time_series_name]['trend'] = {'trend_result': trend_result,
                                                                              'trend_component_STL': trend,
                                                                              'trend_slope': trend_slope,
                                                                              'trend_fitted_line': line,
                                                                              'trend_bkp_dates': result_bkp_dates,
                                                                              'trend_bkp_dates_significant': result_bkp_dates_significant,
                                                                              'trend_mk_significant': result_mk_significant}

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
            for peak in peaks_seasonal_STL:
                ax.axvline(time_series.index[peak], color='r', linestyle='--')
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
        
        return behaviour_indian_monsoon, most_common_label_peaks, most_common_label_minima

    def test_presence_of_seasonality(self, time_series, time_series_name):

        if self.verbose:
            print('Checking for the presence of SEASONALITY')

        # Define sinusoidal function (for fitting)
        def seasonal_function_fitting(x, A, k, period, phi, offset):
            return A * np.exp(k * x) * np.sin(2 * np.pi / period * x + phi) + offset
        
        # List of tests
        tests = ['acf', 'fourier_analysis', 'sinus_fitting']

        # Iterate over tests
        for test in tests:
            if self.verbose:
                print('Test: {}'.format(test))

            # Autocorrelation function
            # https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.acf.html
            if test == 'acf':
                # Define lags and compute ACF
                lags_acf = acf(time_series, nlags=len(time_series)-1)

                # Define range of lags
                range_lags_acf = np.arange(1, len(time_series)+1)

                # Find peaks
                peaks, _ = find_peaks(lags_acf, height=0)

                # First peak
                if len(peaks) > 0:
                    first_peak_acf = peaks[0] + 1
                
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
                period_fourier = stats.mode(np.diff(peaks)).mode

                if self.verbose:
                    if len(peaks) > 0:
                        print('Result: seasonality present')
                        print('First peak at lag {}'.format(period_fourier))
                    
                    else:
                        print('Result: seasonality not present')

            # Fitting sinusoidal function to seasonal component
            elif test == 'sinus_fitting':

                # STL decomposition on original time series and filtered time series
                res = STL(time_series, period=self.period).fit()
                res_fourier = STL(filtered_data_fourier, period=self.period).fit()

                # Extract seasonal component and trend
                seasonal_STL = res.seasonal
                seasonal_STL_fourier = res_fourier.seasonal
                trend_fourier = res_fourier.trend

                # Numerical dates for fitting sinusoidal function
                numeric_dates = np.array([(date - time_series.index[0]).days for date in time_series.index])

                # Initial guess for A, k, phi, offset
                initial_guess = [7, 0.001, 365, 90, 0]  

                # Fitting sinusoidal function to seasonal component from original time series
                try:
                    params, _ = curve_fit(seasonal_function_fitting, xdata=numeric_dates, ydata=seasonal_STL, p0=initial_guess)
                
                except:
                    params = None
                
                if params is None:
                    seasonal_fit_STL = None
                    peaks_seasonal_STL = None
                    period_seasonal_STL = None
                    k = None
                    period_seasonal_STL_fourier = None
                    peaks_seasonal_STL_fourier = None
                    amplitude_seasonal_STL = None
                
                else:
                    # Amplitude of seasonal component
                    amplitude_seasonal_STL = params[0]
                    k = params[1]
                    seasonal_fit_STL = seasonal_function_fitting(numeric_dates, *params)            

                    # Find peaks
                    peaks_seasonal_STL, _ = find_peaks(seasonal_fit_STL, height=0)
                    peaks_seasonal_STL_fourier, _ = find_peaks(seasonal_STL_fourier, height=0)

                    # Find periods
                    period_seasonal_STL = stats.mode(np.diff(peaks_seasonal_STL)).mode
                    period_seasonal_STL_fourier = stats.mode(np.diff(peaks_seasonal_STL_fourier)).mode
        
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
        
        # Save in island_info
        self.island_info['timeseries_analysis'][time_series_name]['seasonality'] = {'lags_acf': lags_acf,
                                                                                    'range_lags_acf': range_lags_acf,
                                                                                    'first_peak_acf': first_peak_acf,
                                                                                    'filtered_data_fourier': filtered_data_fourier,
                                                                                    'peaks_fourier': peaks_fourier,
                                                                                    'period_fourier': period_fourier,
                                                                                    'seasonal_component_STL': seasonal_STL,
                                                                                    'seasonal_component_STL_fourier': seasonal_STL_fourier,
                                                                                    'trend_fourier': trend_fourier,
                                                                                    'amplitude_seasonal_STL': amplitude_seasonal_STL,
                                                                                    'seasonal_fit_STL': seasonal_fit_STL,
                                                                                    'peaks_seasonal_STL': peaks_seasonal_STL,
                                                                                    'peaks_seasonal_STL_fourier': peaks_seasonal_STL_fourier,
                                                                                    'period_seasonal_STL': period_seasonal_STL,
                                                                                    'period_seasonal_STL_fourier': period_seasonal_STL_fourier,
                                                                                    'relation_indian_monsoon': relation_indian_monsoon,
                                                                                    'behaviour_indian_monsoon': behaviour_indian_monsoon,
                                                                                    'seasonality_indian_monsoon_peaks': most_common_label_peaks,
                                                                                    'seasonality_indian_monsoon_minima': most_common_label_minima}

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
        self.island_info['timeseries_analysis'][time_series_name]['stationarity'] = {'stationarity_result': stationarity_result}

        return stationarity_result

    def plot_characterisation(self, time_series, time_series_name, transect):
        
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
        ax[1, 0].axvline(results_analysis['seasonality']['first_peak_acf'], color='r', linestyle='--', label='period {}'.format(results_analysis['seasonality']['first_peak_acf']))
        ax[1, 0].legend(fontsize=10)
        ax[1, 0].set_title('Autocorrelation function')

        # Plot Fourier analysis
        ax[1, 1].plot(time_series.index, time_series.values, color='k')
        ax[1, 1].plot(time_series.index, results_analysis['seasonality']['filtered_data_fourier'], color='g', label='period {}'.format(results_analysis['seasonality']['period_fourier']))
        for peak in results_analysis['seasonality']['peaks_fourier']:
            ax[1, 1].axvline(time_series.index[peak], color='r', linestyle='--')
        ax[1, 1].legend(fontsize=10)
        ax[1, 1].set_title('Fourier reconstruction')

        # Plot fitting sinusoidal function
        ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_component_STL'].values, color='purple', label='seasonal component')
        for peak in results_analysis['seasonality']['peaks_seasonal_STL']:
            ax[2, 0].axvline(time_series.index[peak], color='r', linestyle='--')
        for minimum in results_analysis['seasonality']['peaks_seasonal_STL']:
            ax[2, 0].axvline(time_series.index[minimum], color='pink', linestyle='--')        
        for year in range(2010, 2023):
            if year == 2010:
                ax[2, 0].axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2, label='Northeast Monsoon')
                ax[2, 0].axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2, label='Southwest Monsoon')
            else:
                ax[2, 0].axvspan(datetime.datetime(year=year, month=12, day=1), datetime.datetime(year=year+1, month=2, day=28), color='yellow', alpha=0.2)
                ax[2, 0].axvspan(datetime.datetime(year=year, month=5, day=1), datetime.datetime(year=year, month=9, day=30), color='blue', alpha=0.2)
        ax[2, 0].plot(time_series.index, results_analysis['seasonality']['seasonal_fit_STL'], color='g', label='sinusoidal fit (period {})'.format(results_analysis['seasonality']['period_seasonal_STL']))
        ax[2, 0].legend(fontsize=10)
        ax[2, 0].set_xlim(time_series.index[0], time_series.index[-1])
        ax[2, 0].set_title('Seasonal component from original data')

        # Plot Fourier filtered seasonal component
        ax[2, 1].plot(time_series.index, results_analysis['seasonality']['seasonal_component_STL_fourier'], color='purple', label='period {}'.format(results_analysis['seasonality']['period_seasonal_STL_fourier']))
        for peak in results_analysis['seasonality']['peaks_seasonal_STL_fourier']:
            ax[2, 1].axvline(time_series.index[peak], color='r', linestyle='--')
        ax[2, 1].legend(fontsize=10)
        ax[2, 1].set_title('Seasonal component from filtered data')

        plt.show()

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
        c_seasonality_amplitude = [ts_analysis_results[val]['seasonality']['amplitude_seasonal_STL'] for val in ts_analysis_results.keys()]

        # Seasonality peaks and minima results
        c_seasonality_peaks = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_peaks'] for val in ts_analysis_results.keys()]
        c_seasonality_minima = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_minima'] for val in ts_analysis_results.keys()]

        # DataFrame for trend
        df_trend = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'linear regression slope': c_trend, 'statistical significance': symbols_trend, 's': 100})

        # DataFrame for seasonality
        df_seasonality_amplitude = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'amplitude': c_seasonality_amplitude, 's': 100})
        df_seasonality_peaks = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'peak period': c_seasonality_peaks, 's': 100})
        df_seasonality_minima = pd.DataFrame({'x': x_intersections, 'y': y_intersections, 'minimum period': c_seasonality_minima, 's': 100})

        # Trend plot
        sns.scatterplot(data=df_trend, x='x', y='y', hue='linear regression slope', style='statistical significance', s=100, ax=ax[0, 0], palette=sns.color_palette("RdYlGn", as_cmap=True), zorder=2, edgecolor='k')

        # Seasonality amplitude plot
        sns.scatterplot(data=df_seasonality_amplitude, x='x', y='y', hue='amplitude', s=100, ax=ax[0, 1], palette=sns.color_palette("viridis", as_cmap=True), zorder=2, edgecolor='k')

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

    def characterisation(self, time_series, time_series_name, transect):
        
        # Presence of trend
        self.test_presence_of_trend(time_series, time_series_name)

        # Presence of seasonality
        self.test_presence_of_seasonality(time_series, time_series_name)

        # Stationarity
        self.test_stationarity(time_series, time_series_name)

        # Plotting characterisation for this transect
        if self.plot_results_transect:
            self.plot_characterisation(time_series, time_series_name, transect)

    def main(self):

        if self.plot_only and not self.overwrite:
            # Make plots for the whole island
            self.make_plots()

            return self.island_info

        else:
        
            if 'timeseries_analysis' in self.island_info.keys() and not self.overwrite:
                return self.island_info
            
            else:
                self.island_info['timeseries_analysis'] = {}

            print('\n-------------------------------------------------------------------')
            print('Time series analysis')
            print('Island:', ', '.join([self.island, self.country]))
            print('-------------------------------------------------------------------\n')

            # Iterate over transects
            for transect in tqdm(self.transects, desc='Transect'):

                time_series_name = 'coastline_position_transect_{}_waterline'.format(transect)
                time_series = self.island_info['timeseries_preprocessing']['optimal time period']['dict_timeseries'][time_series_name]['monthly'][time_series_name]
                
                # Time series quality check
                if len(time_series) < 30:
                    continue
                
                # Initialise empty dictionary for time series analysis for this transect
                if time_series_name not in self.island_info['timeseries_analysis'].keys():
                    self.island_info['timeseries_analysis'][time_series_name] = {}
                
                # Characterisation of the time series
                self.characterisation(time_series, time_series_name, transect)
            
            # Make plots for the whole island
            self.make_plots()

            return self.island_info