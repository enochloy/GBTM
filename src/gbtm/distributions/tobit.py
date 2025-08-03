import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import statsmodels.api as sm
from gbtm.distributions.base import DistributionModel


class CensoredNormalModel(DistributionModel):
    def __init__(self, sigma=0.05, lower_bound=-np.inf, upper_bound=np.inf):
        assert sigma > 0, "Sigma must be greater than 0 for CensoredNormalModel."
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dist = "cnorm"
        self.ylabel = "Value"

    def init_params(self, y, clusters, K):
        # calculate no. covariates
        beta = np.zeros((K, self.X.shape[1]))

        for k in range(K):
            nan_mask = np.isnan(y)
            cluster_mask = clusters == k
            if np.sum(cluster_mask) > 0:
                Y_cluster = y[~nan_mask & cluster_mask]
                X_cluster = self.X.loc[~nan_mask & cluster_mask].to_numpy()

                try:
                    model_ols = sm.OLS(Y_cluster, X_cluster).fit()
                    beta[k] = model_ols.params
                except Exception as e:
                    print(
                        f"Warning: OLS initial fit failed for cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    beta[k, 0] = Y_cluster.mean()
                    beta[k, 1:] = 0.0  # Set other coefficients to 0
            else:
                print(
                    f"Warning: Cluster {k} has no observed data points after dropping NaNs. Using random intercept for initialization."
                )
                beta[k, 0] = np.random.normal(0, 0.1)  # for future improvement
                beta[k, 1:] = 0.0
                continue  # Skip to next cluster if no observed data

        return {"beta": beta}

    def log_likelihood(self, y, K, params, units):
        """
        Compute the log-likelihood per individual for each group.

        Parameters:
            y (array): Observed outcomes (N_obs,)
            K (int): Number of groups
            params (dict): Coefficients for each group (K, P)
            units (array): Unit identifiers (N_obs,) to group log-likelihoods

        Returns:
            DataFrame: Log-likelihood per unit and group (N units x K groups)
        """
        beta = params["beta"]  # (K, len_coef)

        ll_obs = np.full((len(y), K), fill_value=np.nan)  # (N_obs, K)

        for k in range(K):
            mu_k = self.X @ beta[k]  # (N_obs,)

            ll_obs[y <= self.lower_bound, k] = norm.logcdf(
                self.lower_bound, loc=mu_k[y <= self.lower_bound], scale=self.sigma
            )
            ll_obs[y >= self.upper_bound, k] = norm.logsf(
                self.upper_bound, loc=mu_k[y >= self.upper_bound], scale=self.sigma
            )
            mid = (y > self.lower_bound) & (y < self.upper_bound)
            ll_obs[mid, k] = norm.logpdf(y[mid], loc=mu_k[mid], scale=self.sigma)

        # Convert to DataFrame and attach unit ID for grouping
        ll_obs = pd.DataFrame(ll_obs)
        grouped_ll = ll_obs.groupby(units, sort=False).sum()

        return grouped_ll

    def neg_loglikelihood(self, beta_k, y, post_k):
        """
        Compute negative log-likelihood for one group, weighted by posteriors.

        Parameters:
            y (array): Observed outcomes (N_obs,)
            beta_k (array): Coefficients for group k (P,)
            post_k (array): Posterior probs for group k (N_obs,)

        Returns:
            float: Negative log-likelihood
        """
        mu_k = self.X @ beta_k
        ll_obs = np.full(len(y), np.nan)

        ll_obs[y <= self.lower_bound] = norm.logcdf(
            self.lower_bound, loc=mu_k[y <= self.lower_bound], scale=self.sigma
        )
        ll_obs[y >= self.upper_bound] = norm.logsf(
            self.upper_bound, loc=mu_k[y >= self.upper_bound], scale=self.sigma
        )
        mid = (y > self.lower_bound) & (y < self.upper_bound)
        ll_obs[mid] = norm.logpdf(y[mid], loc=mu_k[mid], scale=self.sigma)

        total_neg_ll = -np.sum(post_k[~np.isnan(ll_obs)] * ll_obs[~np.isnan(ll_obs)])
        return total_neg_ll

    def maximize(self, y, K, post, params):
        beta = np.zeros_like(params["beta"])
        for k in range(K):
            # For fit, post[:,k] are the weights specific to group k for each individual
            results = minimize(
                self.neg_loglikelihood,
                params["beta"][k],
                args=(y, post[:, k]),  # Pass the correct posterior column
                method="L-BFGS-B",
            )
            beta[k] = results.x
        return {"beta": beta}

    def predict_expected_value(self, params, assigned_groups):
        beta = params["beta"]
        mu = np.empty(len(self.X))
        for k in range(beta.shape[0]):
            idx = assigned_groups == k
            mu[idx] = self.X[idx] @ beta[k]
        return mu
