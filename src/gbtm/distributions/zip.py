import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
import statsmodels.api as sm
from gbtm.distributions.base import DistributionModel


class ZIPModel(DistributionModel):
    def __init__(self):
        self.dist = "zip"
        self.ylabel = "Expected Count"

    def init_params(self, y, clusters, K):
        # initialize params
        beta = np.zeros((K, self.X.shape[1]))
        gamma = np.zeros((K, self.X.shape[1]))

        # loop through clusters for initial GLM fit
        for k in range(K):
            nan_mask = np.isnan(y)
            cluster_mask = clusters == k
            if np.sum(cluster_mask) > 0:
                Y_cluster = y[~nan_mask & cluster_mask]
                X_cluster = self.X.loc[~nan_mask & cluster_mask].to_numpy()

                # initialize beta (Poisson process)
                try:
                    model_pois = sm.GLM(
                        Y_cluster,
                        X_cluster,
                        family=sm.families.Poisson(),
                    ).fit()
                    beta[k] = model_pois.params
                except Exception as e:
                    print(
                        f"Warning: GLM (beta) initial fit failed for ZIP KMeans cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    mean_rate = np.clip(np.mean(Y_cluster), 1e-12, np.inf)
                    beta[k, 0] = np.log(mean_rate)  # Log link for Poisson
                    beta[k, 1:] = 0  # Ensure other coefficients are 0

                # initialize gamma (zero-inflation process)
                Y_gamma = (Y_cluster == 0).astype(int)

                try:
                    model_log = sm.GLM(
                        Y_gamma,
                        X_cluster,
                        family=sm.families.Binomial(),
                    ).fit()
                    gamma[k] = model_log.params
                except Exception as e:
                    print(
                        f"Warning: GLM (gamma) initial fit failed for ZIP KMeans cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    mean_zero_prob = np.clip(np.mean(Y_gamma), 1e-12, 1 - 1e-12)
                    gamma[k, 0] = np.log(
                        mean_zero_prob / (1 - mean_zero_prob)
                    )  # Logit link for Binomial
                    gamma[k, 1:] = 0  # Ensure other coefficients are 0
            else:
                # fallback for empty cluster
                beta[k, 0] = np.random.normal(0, 0.1)  # for future improvement
                gamma[k, 0] = np.random.norma(0, 0.1)  # for future improvement
                print(
                    f"Warning: Cluster {k} has no observed data points after dropping NaNs. Using random intercept for initialization."
                )
        return {"beta": beta, "gamma": gamma}

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
        ll_obs = np.full((len(y), K), fill_value=np.nan)  # (N_obs, K)

        # create indicator for observed zeros
        zero_mask = (y == 0).astype(int)

        for k in range(K):
            # calculate lambda and p for group k for each individual and timepoint
            lam_k = np.exp(self.X @ params["beta"][k])  # (N_obs, )
            p_k = np.clip(
                expit(self.X @ params["gamma"][k]), 1e-12, 1 - 1e-12
            )  # (N_obs, )

            # ZIP log-likelihood calculation based on the two components
            # If data is 0: log(p_t + (1-p_t) * exp(-lambda_t))
            # If data is > 0: log(1-p_t) - lambda_t + y*log(lambda_t) - gammaln(y+1) (Poisson PMF for non-zero)

            # Term for observed zeros (either from zero-inflation or Poisson)
            log_prob_zero_component = np.log(p_k + (1 - p_k) * np.exp(-lam_k))

            # Term for observed non-zeros (must come from Poisson component)
            log_prob_nonzero_component = (
                np.log(1 - p_k) - lam_k + y * np.log(lam_k) - gammaln(y + 1)
            )

            # Combine based on whether the observed data point is zero or not
            ll_obs[:, k] = (
                zero_mask * log_prob_zero_component
                + (1 - zero_mask) * log_prob_nonzero_component
            )

        # Convert to DataFrame and attach unit ID for grouping
        ll_obs = pd.DataFrame(ll_obs)
        grouped_ll = ll_obs.groupby(units, sort=False).sum()

        return grouped_ll

    def maximize(self, y, K, post, params):
        beta = np.zeros_like(params["beta"])  # poisson coefficients
        gamma = np.zeros_like(params["gamma"])  # zero-inflation coefficients
        zero_mask = (y == 0).astype(int)  # Indicator for observed zeros
        nan_mask = np.isnan(y)

        for k in range(K):
            p_k = np.clip(expit(self.X @ params["gamma"][k]), 1e-12, 1 - 1e-12)
            lam_k = np.exp(self.X @ params["beta"][k])

            # calculate Z_approx, posterior probability of observed 0 coming from zero-inflated process
            denominator_Z = p_k + (1 - p_k) * np.exp(-lam_k)
            Z_approx = zero_mask * (p_k / np.clip(denominator_Z, 1e-12, np.inf))

            # maximize gamma (zero-inflation component)
            model_log = sm.GLM(
                Z_approx[~nan_mask],
                self.X[~nan_mask],
                family=sm.families.Binomial(),
                freq_weights=post[~nan_mask, k],
            ).fit()
            gamma[k] = model_log.params

            # maximize beta (poisson part)
            model_pois = sm.GLM(
                y[~nan_mask],
                self.X[~nan_mask],
                family=sm.families.Poisson(),
                freq_weights=(1 - Z_approx[~nan_mask]) * post[~nan_mask, k],
            ).fit()
            beta[k] = model_pois.params

        return {"beta": beta, "gamma": gamma}

    def predict_expected_value(self, params, assigned_groups):
        beta = params["beta"]  # (K, n_covariates)
        gamma = params["gamma"]
        mu = np.empty(len(self.X))
        for k in range(beta.shape[0]):
            idx = assigned_groups == k
            p_zero_k = expit(self.X[idx] @ gamma[k])
            lambda_k = np.exp(self.X[idx] @ beta[k])
            mu[idx] = (1 - p_zero_k) * lambda_k
        return mu
