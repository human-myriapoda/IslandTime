# Import modules
import numpy as np
import pandas as pd
import itertools
import os
from scipy import stats
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

    def evaluate_cross_correlation(self, ts_df_subset, ax):
        return
        print('--- Evaluating cross-correlation ---')

        # Perform cross-correlation
        backwards = ccf(ts1[::-1], ts2[::-1], adjusted=False, nlags=24)[::-1]
        forwards = ccf(ts1, ts2, adjusted=False, nlags=24)
        ccf_output = np.r_[backwards[:-1], forwards]

        # Identify the lag with the maximum cross-correlation
        max_lag = np.argmax(ccf_output) - len(ccf_output)//2
        print('Maximum cross-correlation at lag: ', max_lag)

        # Plot cross-correlation
        ax.plot(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, color='green')
        ax.axvline(max_lag, color='red', linestyle='--')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross-correlation')
        ax.set_title('Cross-correlation between time series 1 and time series 2')

        return ax

    def evaluate_correlation(self, ts_df_subset, ax):
        print('--- Evaluating correlation at lag 0 ---')

        # Extract time series
        columns = list(ts_df_subset.columns)
        ts1 = ts_df_subset[columns[0]]
        ts2 = ts_df_subset[columns[1]]

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
            label_pearson = 'Significant Pearson correlation'
        
        else:
            label_pearson = 'No significant Pearson correlation'

        # Plot correlation
        sns.regplot(x=columns[0], y=columns[1], data=ts_df_subset, ci=95, ax=ax, scatter_kws={"s": 50, "color": "grey"}, line_kws={"color": "red"}, label=label_pearson)

        # Add aesthetics
        ax.set_title('Correlation at lag 0')
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

            # Evaluate Granger causality
            axs[0] = self.evaluate_granger_causality(ts_df_subset, axs[0])

            # Evaluate cross-correlation
            axs[1] = self.evaluate_cross_correlation(ts_df_subset, axs[1])

            # Evaluate correlation
            axs[2], dict_results_temp['correlation_lag_0'] = self.evaluate_correlation(ts_df_subset, axs[2])

            # Evaluate cointegration
            axs[3] = self.evaluate_cointegration(ts_df_subset, axs[3])

            # Evaluate transfer entropy
            self.evaluate_transfer_entropy()

            # Evaluate mutual information
            self.evaluate_mutual_information()

            # Evaluate dynamic time warping
            self.evaluate_dynamic_time_warping()

            # Evaluate spatial correlation
            self.evaluate_spatial_correlation()

            # Evaluate spectral similarity
            self.evaluate_spectral_similarity()

            # Evaluate lagged regression
            self.evaluate_lagged_regression()

            # Save results 
            dict_results['{}___{}'.format(combinations[idx_comb][0], combinations[idx_comb][1])] = dict_results_temp

            # Add suptitle
            fig.suptitle('Connections between time series: {} & {}'.format(combinations[idx_comb][0], combinations[idx_comb][1]), fontsize=16)
        
        print(dict_results)

        return 