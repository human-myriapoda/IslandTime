# Import modules
import numpy as np
import pandas as pd
import itertools
import os
from scipy import stats, signal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, coint
import statsmodels.api as sm
import seaborn as sns
import Rbeast as rb
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity
import pyinform
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
# from PyIF import te_compute as te

# TODO
# - Spatial correlation for nearby transects or islands
# - Fix Transfer Entropy

class TimeSeriesConnections:
    def __init__(self, dict_time_series: dict):
        self.dict_time_series = dict_time_series

    def evaluate_granger_causality(self, ts_df_subset, ax, invert=False):

        print('--- Evaluating Granger causality ---')

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # Perform Granger causality test
        # Null hypothesis: ts2 does not Granger cause ts1
        max_lag = range(-12, 13, 1)
        max_lag = 12
        if invert:
            # Null hypothesis: ts1 does not Granger cause ts2
            results = grangercausalitytests(np.column_stack((ts2, ts1)), max_lag, verbose=False)
        
        else:
            # Null hypothesis: ts2 does not Granger cause ts1
            results = grangercausalitytests(np.column_stack((ts1, ts2)), max_lag, verbose=False)

        # Extract p-values
        p_values = [round(results[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]

        # Find p-values less than 0.05
        p_values_significant = np.where(np.array(p_values) < 0.05)[0]

        # Extract F-statistics
        f_statistics = [round(results[i+1][0]['ssr_ftest'][0], 4) for i in range(max_lag)]

        # Plot p-values and F-statistics on twin axes
        ax.plot(range(1, max_lag+1), p_values, color='darkorange', label='p-value')
        ax.set_xlabel('Lag (months)')
        ax.set_ylabel('p-value')

        ax2 = ax.twinx()
        ax2.plot(range(1, max_lag+1), f_statistics, color='darkviolet', label='F-statistic')
        ax2.set_ylabel('F-statistic')

        ax.axhline(0.05, color='grey', linestyle='--', label='5% Significance Level')
        ax.axhline(0.01, color='grey', linestyle=':', label='1% Significance Level')
        ax.axhline(0.1, color='grey', linestyle='-.', label='10% Significance Level')
        if len(p_values_significant) > 0:
            for idx_pvals, pvals in enumerate(p_values_significant):
                if idx_pvals == 0:
                    ax.axvline(pvals + 1, color='k', linestyle='-', label='Significant Lag', alpha=0.5)
                else:
                    ax.axvline(pvals + 1, color='k', linestyle='-', alpha=0.5)
        if invert:
            ax.set_title('Granger Causality {} -> {}'.format(columns[0], columns[1]))
        else:
            ax.set_title('Granger Causality {} -> {}'.format(columns[1], columns[0]))
        ax.legend()

        # Save results in dictionary
        dict_results_granger = {
            'p_values': p_values,
            'f_statistics': f_statistics,
            'p_values_significant': p_values_significant
        }

        return ax, dict_results_granger

    def _ccf_values(self, ts1, ts2):
        p = (ts1 - np.mean(ts1)) / (np.std(ts1) * len(ts1))
        q = (ts2 - np.mean(ts2)) / (np.std(ts2))  
        c = np.correlate(p, q, 'full')
        return c

    def evaluate_cross_correlation(self, ts_df_subset, ax):
        print('--- Evaluating cross-correlation ---')

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # Calculate cross-correlation
        ccf = self._ccf_values(ts1, ts2)

        # Lag values
        lags_ccf = signal.correlation_lags(len(ts1), len(ts2))

        # Find peaks and minima in cross-correlation
        peaks = signal.find_peaks(ccf)[0]
        minima = signal.find_peaks(-ccf)[0]
        lags_peaks = lags_ccf[peaks]
        lags_minima = lags_ccf[minima]

        # Lags greater than 5% confidence interval
        lags_significant = lags_ccf[np.where(np.abs(ccf) > 2/np.sqrt(23))[0]]
        ccf_significant = ccf[np.where(np.abs(ccf) > 2/np.sqrt(23))[0]]

        # Remove lags that are not in lags_peak or lags_minima
        mask = np.isin(lags_significant, lags_peaks) | np.isin(lags_significant, lags_minima)
        lags_significant = lags_significant[mask]
        ccf_significant = ccf_significant[mask]

        # Significant positive correlations
        lags_significant_pos = lags_significant[ccf_significant > 0]
        ccf_significant_pos = ccf_significant[ccf_significant > 0]

        # Significant negative correlations
        lags_significant_neg = lags_significant[ccf_significant < 0]
        ccf_significant_neg = ccf_significant[ccf_significant < 0]

        # First significant positive and negative lags
        if len(lags_significant_pos) == 0:
            first_lag_significant_pos = np.nan
            first_ccf_significant_pos = np.nan
        
        if len(lags_significant_neg) == 0:
            first_lag_significant_neg = np.nan
            first_ccf_significant_neg = np.nan
        
        else:
            first_lag_significant_pos = lags_significant_pos[np.argmin(np.abs(lags_significant_pos))]
            first_lag_significant_neg = lags_significant_neg[np.argmin(np.abs(lags_significant_neg))]
            first_ccf_significant_pos = ccf_significant_pos[np.argmin(np.abs(lags_significant_pos))]
            first_ccf_significant_neg = ccf_significant_neg[np.argmin(np.abs(lags_significant_neg))]
        
        # Save results in dictionary
        dict_results_ccf = {
            'lags_ccf': lags_ccf,
            'ccf': ccf,
            'lags_significant': lags_significant,
            'ccf_significant': ccf_significant,
            'lags_significant_pos': lags_significant_pos,
            'lags_significant_neg': lags_significant_neg,
            'ccf_significant_pos': ccf_significant_pos,
            'ccf_significant_neg': ccf_significant_neg,
            'first_significant_pos': first_lag_significant_pos,
            'first_significant_neg': first_lag_significant_neg,
            'first_ccf_significant_pos': first_ccf_significant_pos,
            'first_ccf_significant_neg': first_ccf_significant_neg
        }

        # Plot cross-correlation
        ax.plot(lags_ccf, ccf, color='k')
        ax.axhline(-2/np.sqrt(23), color='red', label='5% Confidence Interval')
        ax.axhline(2/np.sqrt(23), color='red')
        ax.axvline(x = 0, color = 'grey', lw = 1)
        ax.axhline(y = 0, color = 'grey', lw = 1)
        ax.axhline(y = np.max(ccf), color = 'blue', lw = 1, linestyle='--', label = 'Highest +/- Correlation')
        ax.axhline(y = np.min(ccf), color = 'blue', lw = 1, linestyle='--')
        ax.axvline(x = first_lag_significant_pos, color = 'gold', lw = 1, linestyle='--', label = 'First + Correlation: Lag {}'.format(first_lag_significant_pos))
        ax.axvline(x = first_lag_significant_neg, color = 'green', lw = 1, linestyle='--', label = 'First - Correlation: Lag {}'.format(first_lag_significant_neg))
        ax.set(xlim = [-24, 24], ylim = [-1, 1])
        ax.set_title('Cross Correlation')
        ax.set_ylabel('Correlation Coefficients')
        ax.set_xlabel('Lag (months)')
        # ax.tick_params(axis='both', which='major', labelsize=15)
        ax.legend(loc='upper left')

        return ax, dict_results_ccf

    def evaluate_correlation(self, ts_df_subset, ax, lag=0):
        print('--- Evaluating correlation at lag {} ---'.format(lag))

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # If lag is not 0, shift time series
        if lag != 0:
            ts2 = ts2.shift(lag)

        # Handle NaN values introduced by shifting
        valid_indices = ts1.dropna().index.intersection(ts2.dropna().index)
        ts1 = ts1.loc[valid_indices]
        ts2 = ts2.loc[valid_indices]

        # Shifted time series
        ts_df_subset = pd.DataFrame({columns[0]: ts1, columns[1]: ts2})

        # Pearson correlation
        corr_pearson, pvalue_pearson = stats.pearsonr(ts1, ts2)

        # Spearman correlation
        corr_spearman, pvalue_spearman = stats.spearmanr(ts1, ts2)

        # Kendall correlation
        corr_kendall, pvalue_kendall = stats.kendalltau(ts1, ts2)
        
        # Store results in dictionary
        dict_results_eval = {
            'pearson': {'correlation': corr_pearson, 'p-value': pvalue_pearson},
            'spearman': {'correlation': corr_spearman, 'p-value': pvalue_spearman},
            'kendall': {'correlation': corr_kendall, 'p-value': pvalue_kendall}
        }

        # Add interpretation
        if pvalue_pearson < 0.05:
            label_pearson = r'Significant Pearson correlation ($R = {}$)'.format(np.round(corr_pearson, 2))
        
        else:
            label_pearson = r'No significant Pearson correlation ($R = {}$)'.format(np.round(corr_pearson, 2))

        # Plot correlation
        sns.regplot(x=columns[0], y=columns[1], data=ts_df_subset, ci=95, ax=ax, scatter_kws={"s": 50, "color": "grey"}, line_kws={"color": "forestgreen"}, label=label_pearson)

        # Add aesthetics
        ax.set_title('Correlation at Lag {}'.format(lag))
        ax.legend(loc='upper left')
        # ax.tick_params(axis='both', which='major', labelsize=15)
        
        return ax, dict_results_eval

    def evaluate_cointegration(self, ts_df_subset_trend, ax):
        print('--- Evaluating cointegration on trend components ---')

        # Extract time series
        columns = list(ts_df_subset_trend.columns)
        ts1 = ts_df_subset_trend[columns[0]]
        ts2 = ts_df_subset_trend[columns[1]]

        # Co-integration test (Engle-Granger Test)
        # Null hypothesis: there is no cointegration
        t_statistic, p_val, critical_p_val = coint(ts1, ts2, trend='c', autolag='BIC')

        # Get OLS regression
        X = sm.add_constant(ts2)  # Add intercept
        model = sm.OLS(ts1, X).fit()

        # Predictions and residuals
        predictions = model.predict(X)
        residuals = ts1 - predictions

        # Plot time series and OLS regression
        ax.plot(ts_df_subset_trend.index, self._normalise_data_plotting(ts1), label=columns[0], color='firebrick')
        ax.plot(ts_df_subset_trend.index, self._normalise_data_plotting(ts2), label=columns[1], color='c')
        ax.plot(ts_df_subset_trend.index, self._normalise_data_plotting(predictions), '--', color='black', label=f'OLS Regression (Co-int: pval={p_val:.3f}, t-stat={t_statistic:.3f})')
        ax.set_title('Cointegration Test on Trend Components')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalised Value')
        ax.legend()

        # Store results in dictionary
        dict_results_coint = {
            't_statistic': t_statistic,
            'p_val': p_val,
            'critical_p_val': critical_p_val
        }

        return ax, dict_results_coint

    def evaluate_transfer_entropy(self, ts_df_subset, ax):
        print('--- Evaluating transfer entropy ---')

        # Binning function
        def bin_data(data, num_bins):
            return np.digitize(data, bins=np.linspace(np.min(data), np.max(data), num_bins))

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # Bin data into 10 bins
        num_bins = 10
        ts1_binned = bin_data(ts1, num_bins)
        ts2_binned = bin_data(ts2, num_bins)

        # Range of lags to evaluate
        lags = range(1, 8)

        # Loop through all lags
        res_TE_ts1_to_ts2, res_TE_ts2_to_ts1 = {}, {}
        for lag in lags:
            # Compute transfer entropy (1 -> 2)
            te_ts1_to_ts2 = pyinform.transferentropy.transfer_entropy(ts2_binned, ts1_binned, k=lag)
            # te_ts1_to_ts2 = te.te_compute(ts2.values, ts1.values, k=lag, embedding=1, safetyCheck=True, GPU=False)

            # Compute transfer entropy (2 -> 1)
            # te_ts2_to_ts1 = te.te_compute(ts1.values, ts2.values, k=lag, embedding=1, safetyCheck=True, GPU=False)
            te_ts2_to_ts1 = pyinform.transferentropy.transfer_entropy(ts1_binned, ts2_binned, k=lag)

            # Store results in dictionary
            res_TE_ts1_to_ts2[lag] = te_ts1_to_ts2
            res_TE_ts2_to_ts1[lag] = te_ts2_to_ts1
        
        # Plot transfer entropy
        ax.plot(lags, list(res_TE_ts1_to_ts2.values()), label=f'{columns[0]} -> {columns[1]}', color='firebrick')
        ax.plot(lags, list(res_TE_ts2_to_ts1.values()), label=f'{columns[1]} -> {columns[0]}', color='c')
        ax.set_title('Transfer Entropy')
        ax.set_xlabel('Lag (months)')
        ax.set_ylabel('Transfer Entropy')
        ax.legend()

        return ax

    def evaluate_mutual_information(self, ts_df_subset):
        print('--- Evaluating mutual information ---')

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # Compute mutual information
        mi = pyinform.mutualinfo.mutual_info(ts1, ts2)

        print('Mutual information', mi)

        # Store results in dictionary
        dict_results_mi = {
            'mutual_information': mi,
        }

        return dict_results_mi

    def evaluate_dynamic_time_warping(self, ts_df_subset, ax, seasonal=True):
        print('--- Evaluating dynamic time warping ---')

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # Dynamic Time Warping
        dtw_s, path_s = fastdtw(ts1.values, ts2.values, dist=2)
        
        # Scaled DTW
        dtw, path = fastdtw(self._normalise_data_plotting(ts1.values), self._normalise_data_plotting(ts2.values), dist=2)
        
        # Path indices
        path_indices_x = [point[0] for point in path]
        path_indices_y = [point[1] for point in path]
    
        # Measure the lag from DTW
        lag = np.mean(np.abs(np.array(path_indices_x) - np.array(path_indices_y)))

        print('Lag from DTW: {}'.format(lag))
        print('Similarity score: {}'.format(dtw / len(path_s)))

        # Plot DTW
        ax.plot(ts_df_subset.index, self._normalise_data_plotting(ts1.values), label=columns[0], color='firebrick')
        ax.plot(ts_df_subset.index, self._normalise_data_plotting(ts2.values), label=columns[1], color='c')
        for point in path:
            ax.plot([ts_df_subset.index[point[0]], ts_df_subset.index[point[1]]], [self._normalise_data_plotting(ts1.values)[point[0]], self._normalise_data_plotting(ts2.values)[point[1]]], color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time')
        if seasonal:
            ax.set_ylabel('Normalised Seasonal Component')
            ax.set_title('Dynamic Time Warping - Seasonal Components (Lag: {})'.format(lag))
        else:
            ax.set_ylabel('Normalised Value')
            ax.set_title('Dynamic Time Warping (Lag: {})'.format(lag))
        ax.legend()

        # Store results in dictionary
        dict_results_dtw = {
            'dtw': dtw,
            'dtw_scaled': dtw_s,
            'lag': lag,
            'path': path,
            'path_scaled': path_s
        }

        # Shortest Path Plot (Cost Matrix with the path)
        # In this example, only the path is plotted, not the entire cost matrix.
        # ax2 = plt.subplot2grid((2, 2), (0, 1))
        # ax2.plot(np.array(path_s)[:, 0], np.array(path_s)[:, 1], 'green', marker='o', linestyle='-')
        # ax2.set_title('Shortest Path (Best Path)')
        # ax2.set_xlabel(columns[0])
        # ax2.set_ylabel(columns[1])
        # ax2.grid(True)

        return ax, dict_results_dtw

    def evaluate_spectral_similarity(self, ts_df_subset, ax):
        print('--- Evaluating spectral similarity ---')

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

        # Compute power spectral density with monthly resolution
        f1, Pxx1 = welch(ts1, fs=1, nperseg=ts1.shape[0])
        f2, Pxx2 = welch(ts2, fs=1, nperseg=ts2.shape[0])

        # Convert frequency to cycles per year
        f1 = f1 * 12
        f2 = f2 * 12

        # Euclidean distance between power spectral densities
        euclidean_distance = np.linalg.norm(Pxx1 - Pxx2)

        # Cosine similarity between power spectral densities
        cosine_similarity_res = cosine_similarity(Pxx1.reshape(1, -1), Pxx2.reshape(1, -1))[0][0]

        # Plot power spectral densities
        ax.plot(f1, self._normalise_data_plotting(Pxx1), label=columns[0], color='firebrick')
        ax.plot(f2, self._normalise_data_plotting(Pxx2), label=columns[1], color='c')
        ax.set_title('Spectral Similarity')
        ax.set_xlabel('Frequency (cycles/year)')
        ax.set_ylabel('Power Spectral Density (normalised)')
        ax.legend()

        # Save results in dictionary
        dict_results_spectral = {
            'euclidean_distance': euclidean_distance,
            'cosine_similarity': cosine_similarity_res
        }

        return ax, dict_results_spectral

    def _normalise_data_plotting(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    def time_series_decomposition_BEAST(self, ts_df_subset):
        
        # Empty dictionary to store results
        ts_df_trend, ts_df_seasonal, ts_df_residual, ts_df_diff = {}, {}, {}, {}

        # Loop through all time series
        for idx_ts in range(len(ts_df_subset.columns)):
            ts = ts_df_subset[ts_df_subset.columns[idx_ts]]

            # Run BEAST
            o = rb.beast(ts, start=[ts.index[0].year, ts.index[0].month, ts.index[0].day], season='harmonic', deltat='1/12 year', period='1 year', quiet=True, print_progress=False)
            
            # Extract trend, seasonal and residual components
            trend = o.trend.Y
            seasonal = o.season.Y
            residual = ts - trend - seasonal

            # Get differential results
            diff = ts.diff().dropna()

            # Store results in dictionary
            ts_df_trend[ts_df_subset.columns[idx_ts]] = trend
            ts_df_seasonal[ts_df_subset.columns[idx_ts]] = seasonal
            ts_df_residual[ts_df_subset.columns[idx_ts]] = residual
            ts_df_diff[ts_df_subset.columns[idx_ts]] = diff

        return pd.DataFrame(ts_df_trend), pd.DataFrame(ts_df_seasonal), pd.DataFrame(ts_df_residual), pd.DataFrame(ts_df_diff)

    def main(self):
        
        print('\n-------------------------------------------------------------------')
        print('Evaluating time series connections')
        print('-------------------------------------------------------------------\n')

        # Create DataFrame with time series
        ts_df = pd.DataFrame(self.dict_time_series)
        ts_df_trend, ts_df_seasonal, ts_df_residual, ts_df_diff = self.time_series_decomposition_BEAST(ts_df)

        # Create all possible combinations of time series
        combinations = list(itertools.combinations(self.dict_time_series.keys(), 2))

        # Create dictionary to store results
        dict_results = {}

        # Loop through all combinations
        for idx_comb in range(len(combinations)):

            print('Evaluating connections between time series: {} & {}'.format(combinations[idx_comb][0], combinations[idx_comb][1]))
            dict_results_temp = {}

            # Create panel of plots
            fig, ax = plt.subplots(4, 3, figsize=(20, 15))
            axs = ax.ravel()

            # Subset of DataFrame
            ts_df_subset = ts_df[[combinations[idx_comb][0], combinations[idx_comb][1]]]
            ts_df_trend_subset = ts_df_trend[[combinations[idx_comb][0], combinations[idx_comb][1]]]
            ts_df_seasonal_subset = ts_df_seasonal[[combinations[idx_comb][0], combinations[idx_comb][1]]]
            ts_df_residual_subset = ts_df_residual[[combinations[idx_comb][0], combinations[idx_comb][1]]]
            ts_df_diff_subset = ts_df_diff[[combinations[idx_comb][0], combinations[idx_comb][1]]]          

            # Plot time series
            axs[0].plot(ts_df_subset.index, self._normalise_data_plotting(ts_df_subset[combinations[idx_comb][0]]), label=combinations[idx_comb][0], color='firebrick') 
            axs[0].plot(ts_df_subset.index, self._normalise_data_plotting(ts_df_subset[combinations[idx_comb][1]]), label=combinations[idx_comb][1], color='c')
            axs[0].set_title('Time Series')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Normalised Value')
            axs[0].legend()

            # Evaluate correlation
            index_fig = 1
            axs[index_fig ], dict_results_temp['correlation_lag_0'] = self.evaluate_correlation(ts_df_subset, axs[index_fig], lag=0)
            index_fig += 1

            # Evaluate cross-correlation
            axs[index_fig], dict_results_temp['cross_correlation'] = self.evaluate_cross_correlation(ts_df_subset, axs[index_fig])
            index_fig += 1

            # Plot lagged correlation using ccf
            if dict_results_temp['cross_correlation']['first_significant_pos'] is not np.nan:
                axs[index_fig], dict_results_temp['correlation_lag_{}'.format(dict_results_temp['cross_correlation']['first_significant_pos'])] = self.evaluate_correlation(ts_df_subset, axs[index_fig], lag=dict_results_temp['cross_correlation']['first_significant_pos'])
            index_fig += 1

            if dict_results_temp['cross_correlation']['first_significant_neg'] is not np.nan:
                axs[index_fig], dict_results_temp['correlation_lag_{}'.format(dict_results_temp['cross_correlation']['first_significant_neg'])] = self.evaluate_correlation(ts_df_subset, axs[index_fig], lag=dict_results_temp['cross_correlation']['first_significant_neg'])
            index_fig += 1

            # Evaluate cointegration on the trend components
            axs[index_fig], dict_results_temp['cointegration_trends'] = self.evaluate_cointegration(ts_df_trend_subset, axs[index_fig])
            index_fig += 1

            # Evaluate Granger causality (2 on 1)
            axs[index_fig], dict_results_temp['grangercausality_{} -> {}'.format(combinations[idx_comb][1], combinations[idx_comb][0])] = self.evaluate_granger_causality(ts_df_diff_subset, axs[index_fig])
            index_fig += 1

            # Evaluate Granger causality (1 on 2)
            axs[index_fig], dict_results_temp['grangercausality_{} -> {}'.format(combinations[idx_comb][0], combinations[idx_comb][1])] = self.evaluate_granger_causality(ts_df_residual_subset, axs[index_fig], invert=True)
            index_fig += 1

            # Evaluate spectral similarity
            axs[index_fig], dict_results_temp['spectral_similarity'] = self.evaluate_spectral_similarity(ts_df_subset, axs[index_fig])
            index_fig += 1

            # Evaluate transfer entropy
            # TODO: Fix transfer entropy
            axs[index_fig] = self.evaluate_transfer_entropy(ts_df_subset, axs[index_fig])
            index_fig += 1

            # Evaluate mutual information
            dict_results_temp['mutual_information'] = self.evaluate_mutual_information(ts_df_subset)

            # Evaluate dynamic time warping
            axs[index_fig], dict_results_temp['dynamic_time_warping_seasonal'] = self.evaluate_dynamic_time_warping(ts_df_seasonal_subset, axs[index_fig])
            index_fig += 1

            # Evaluate dynamic time warping (raw)
            axs[index_fig], dict_results_temp['dynamic_time_warping'] = self.evaluate_dynamic_time_warping(ts_df_subset, axs[index_fig], seasonal=False)

            # Save results 
            dict_results['{}___{}'.format(combinations[idx_comb][0], combinations[idx_comb][1])] = dict_results_temp

            # Add suptitle
            fig.suptitle('Connections between time series: {} & {}'.format(combinations[idx_comb][0], combinations[idx_comb][1]), fontsize=16)

            # Tight layout
            plt.tight_layout()
        
        print(dict_results.keys())



        return ts_df_diff, ts_df_residual, ts_df