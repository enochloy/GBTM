import numpy as np
import pandas as pd
from scipy.special import expit
import statsmodels.api as sm
from gbtm.distributions.base import DistributionModel


class BernoulliModel(DistributionModel):
    def __init__(self):
        self.dist = "Bernoulli"
        self.ylabel = "Probability"

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
                    model_log = sm.GLM(
                        Y_cluster,
                        X_cluster,
                        family=sm.families.Binomial(),
                    ).fit()
                    beta[k] = model_log.params
                except Exception as e:
                    # Fallback for small or problematic clusters
                    print(
                        f"Warning: GLM initial fit failed for Bernoulli KMeans cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    mean_prob = np.clip(np.mean(Y_cluster), 1e-12, 1 - 1e-12)
                    beta[k, 0] = np.log(mean_prob / (1 - mean_prob))
            else:
                beta[k, 0] = np.random.normal(0, 0.1)  # for future improvement
                print(
                    f"Warning: Cluster {k} has no observed data points after dropping NaNs. Using random intercept for initialization."
                )
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
        ll_obs = np.full((len(y), K), fill_value=np.nan)  # (N_obs, K)

        for k in range(K):
            p_k = np.clip(
                expit(self.X @ params["beta"][k]), 1e-12, 1 - 1e-12
            )  # (N_obs, )
            ll_obs[:, k] = y * np.log(p_k) + (1 - y) * np.log(1 - p_k)

        # Convert to DataFrame and attach unit ID for grouping
        ll_obs = pd.DataFrame(ll_obs)
        grouped_ll = ll_obs.groupby(units, sort=False).sum()

        return grouped_ll

    def maximize(self, y, K, post, params):
        beta = np.zeros_like(params["beta"])
        nan_mask = np.isnan(y)
        for k in range(K):
            model_log = sm.GLM(
                y[~nan_mask],
                self.X[~nan_mask],
                family=sm.families.Binomial(),
                freq_weights=post[~nan_mask, k],
            ).fit()
            beta[k] = model_log.params

        return {"beta": beta}

    def predict_expected_value(self, params, assigned_groups):
        beta = params["beta"]  # (K, n_covariates)
        mu = np.empty(len(self.X))
        for k in range(beta.shape[0]):
            idx = assigned_groups == k
            mu[idx] = expit(self.X[idx] @ beta[k])
        return mu
