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

class TimeSeriesConnections:
    def __init__(self, dict_time_series: dict):
        self.dict_time_series = dict_time_series

    def evaluate_granger_causality(self, ts1, ts2, ax):
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

    def evaluate_cross_correlation(self, ts1, ts2, ax):
        print('--- Evaluating cross-correlation ---')

        # Perform cross-correlation
        backwards = ccf(ts1[::-1], ts2[::-1], adjusted=False, nlags=24)[::-1]
        forwards = ccf(ts1, ts2, adjusted=False, nlags=24)
        ccf_output = np.r_[backwards[:-1], forwards]

        # Plot cross-correlation
        ax.plot(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output, color='green')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross-correlation')
        ax.set_title('Cross-correlation between time series 1 and time series 2')

        return ax

    def evaluate_correlation(self, ts1, ts2, ax, test_correlation='pearson'):
        print('--- Evaluating correlation ---')

        if test_correlation == 'pearson':
            corr = stats.pearsonr(ts1, ts2)
            print('Pearson correlation coefficient: ', corr[0])

        elif test_correlation == 'spearman':
            corr = stats.spearmanr(ts1, ts2)
            print('Spearman correlation: ', corr[0])

        # Plot correlation
        ax.scatter(ts1, ts2, color='blue')
        ax.set_xlabel('Time series 1')
        ax.set_ylabel('Time series 2')
        ax.set_title('Correlation between time series 1 and time series 2')
        return ax

    def evaluate_cointegration(self, ts1, ts2, ax):
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

        # Create all possible combinations of time series
        combinations = list(itertools.combinations(self.dict_time_series.keys(), 2))
        print(combinations)

        # Loop through all combinations
        for idx_comb in range(len(combinations)):

            # Create panel of plots
            fig, ax = plt.subplots(3, 3, figsize=(15, 15))
            axs = ax.ravel()
            
            ts1 = self.dict_time_series[combinations[idx_comb][0]]
            ts2 = self.dict_time_series[combinations[idx_comb][1]]

            # Evaluate Granger causality
            axs[0] = self.evaluate_granger_causality(ts1, ts2, axs[0])

            # Evaluate cross-correlation
            axs[1] = self.evaluate_cross_correlation(ts1, ts2, axs[1])

            # Evaluate correlation
            axs[2] = self.evaluate_correlation(ts1, ts2, axs[2])

            # Evaluate cointegration
            axs[3] = self.evaluate_cointegration(ts1, ts2, axs[3])

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

        return 