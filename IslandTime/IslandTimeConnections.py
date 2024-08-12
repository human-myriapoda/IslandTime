# Import modules
import numpy as np
import pandas as pd
import itertools
import os
from scipy import stats, signal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf, grangercausalitytests, coint, adfuller
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import seaborn as sns

class TimeSeriesConnections:
    def __init__(self, dict_time_series: dict):
        self.dict_time_series = dict_time_series

    def evaluate_granger_causality(self, ts_df_subset, ax):
        return
        print('--- Evaluating Granger causality ---')

        # Perform Granger causality test
        max_lag = 24
        results = grangercausalitytests(np.column_stack((ts1, ts2)), max_lag, verbose=False)

        # Extract p-values
        p_values = [round(results[i+1][0]['ssr_ftest'][1], 4) for i in range(max_lag)]

        # Extract F-statistics
        f_statistics = [round(results[i+1][0]['ssr_ftest'][0], 4) for i in range(max_lag)]

        # Plot p-values and F-statistics on twin axes
        ax.plot(range(1, max_lag+1), p_values, color='blue')
        ax.set_xlabel('Lag')
        ax.set_ylabel('P-value')

        ax2 = ax.twinx()
        ax2.plot(range(1, max_lag+1), f_statistics, color='red')
        ax2.set_ylabel('F-statistic')

        ax.set_title('Granger causality test')

        return ax

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
        ax.tick_params(axis='both', which='major', labelsize=15)
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
        sns.regplot(x=columns[0], y=columns[1], data=ts_df_subset, ci=95, ax=ax, scatter_kws={"s": 50, "color": "grey"}, line_kws={"color": "red"}, label=label_pearson)

        # Add aesthetics
        ax.set_title('Correlation at Lag {}'.format(lag))
        ax.legend(loc='upper left')
        ax.tick_params(axis='both', which='major', labelsize=15)
        
        return ax, dict_results_eval

    def evaluate_cointegration(self, ts_df_subset, ax):
        return
        print('--- Evaluating cointegration ---')

        t_statistic, p_val, critical_p_val = coint(ts2, ts1, trend='ct', autolag='BIC')
        print(f' t statistic: {np.round(t_statistic, 2)} \n p value: {np.round(p_val,2)} \n critical p values [1%, 5%, 10%] {critical_p_val}')
        
        # X, Y = ts1, ts2
        # X_train = sm.add_constant(X)
        # model = sm.OLS(Y, X_train).fit()

        # # Get the model residuals
        # residuals = model.resid

        # # Plot residuals
        # ax.plot(residuals, label='Residuals')


        # # Perform Johansen cointegration test
        # ccj = coint_johansen(np.column_stack((ts1, ts2)), det_order=1, k_ar_diff=12)
        # print('Trace statistic: ', ccj.lr1)
        # print('Critical values (90%, 95%, 99%): ', ccj.cvt)

        # # Extract the co-integrating vectors
        # coint_vector = ccj.evec[:, 0]

        # # Calculate the co-integrating relationship
        # coint_relationship = np.dot(np.column_stack((ts1, ts2)), coint_vector)

        # # Plot cointegration (residuals of the co-integrating relationship)
        # ax.plot(coint_relationship, label='Co-integrating Relationship')
        # ax.axhline(0, color='black', linestyle='--')
        # ax.legend()
        # ax.set_title('Co-integrating Relationship (Johansen Test)')

        # Test 1: are trends co-integrated?

        # Test 2: are residuals co-integrated?

        # Test 3: are seasonal components co-integrated?

    def evaluate_transfer_entropy(self):
        pass

    def evaluate_mutual_information(self):
        pass

    def evaluate_dynamic_time_warping(self):
        pass

    def evaluate_spatial_correlation(self):
        pass

    def evaluate_spectral_similarity(self):
        pass

    def evaluate_lagged_regression(self):
        pass

    def main(self):
        
        print('\n-------------------------------------------------------------------')
        print('Evaluating time series connections')
        print('-------------------------------------------------------------------\n')

        # Create DataFrame with time series
        ts_df = pd.DataFrame(self.dict_time_series)

        # Create all possible combinations of time series
        combinations = list(itertools.combinations(self.dict_time_series.keys(), 2))

        # Create dictionary to store results
        dict_results = {}

        # Loop through all combinations
        for idx_comb in range(len(combinations)):

            print('Evaluating connections between time series: {} & {}'.format(combinations[idx_comb][0], combinations[idx_comb][1]))
            dict_results_temp = {}

            # Create panel of plots
            fig, ax = plt.subplots(3, 3, figsize=(15, 15))
            axs = ax.ravel()
            
            # ts1 = self.dict_time_series[combinations[idx_comb][0]]
            # ts2 = self.dict_time_series[combinations[idx_comb][1]]

            # Subset of DataFrame
            ts_df_subset = ts_df[[combinations[idx_comb][0], combinations[idx_comb][1]]]

            # Evaluate correlation
            axs[0], dict_results_temp['correlation_lag_0'] = self.evaluate_correlation(ts_df_subset, axs[0], lag=0)

            # Evaluate cross-correlation
            axs[1], dict_results_temp['cross_correlation'] = self.evaluate_cross_correlation(ts_df_subset, axs[1])

            # Plot lagged correlation using ccf
            lag_pos = dict_results_temp['cross_correlation']['first_significant_pos']
            lag_neg = dict_results_temp['cross_correlation']['first_significant_neg']
            axs[2], dict_results_temp['correlation_lag_{}'.format(lag_pos)] = self.evaluate_correlation(ts_df_subset, axs[2], lag=lag_pos)
            axs[3], dict_results_temp['correlation_lag_{}'.format(lag_neg)] = self.evaluate_correlation(ts_df_subset, axs[3], lag=lag_neg)

            # # Evaluate Granger causality
            # axs[2] = self.evaluate_granger_causality(ts_df_subset, axs[2])

            # # Evaluate cointegration
            # axs[3] = self.evaluate_cointegration(ts_df_subset, axs[3])

            # # Evaluate transfer entropy
            # self.evaluate_transfer_entropy()

            # # Evaluate mutual information
            # self.evaluate_mutual_information()

            # # Evaluate dynamic time warping
            # self.evaluate_dynamic_time_warping()

            # # Evaluate spatial correlation
            # self.evaluate_spatial_correlation()

            # # Evaluate spectral similarity
            # self.evaluate_spectral_similarity()

            # # Evaluate lagged regression
            # self.evaluate_lagged_regression()

            # Save results 
            dict_results['{}___{}'.format(combinations[idx_comb][0], combinations[idx_comb][1])] = dict_results_temp

            # Add suptitle
            fig.suptitle('Connections between time series: {} & {}'.format(combinations[idx_comb][0], combinations[idx_comb][1]), fontsize=16)
        
        print(dict_results.keys())

        return 