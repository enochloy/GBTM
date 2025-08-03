import numpy as np
import pandas as pd
from scipy.special import expit
from gbtm.distributions.bernoulli import BernoulliModel


class DropoutModel(BernoulliModel):
    def __init__(self, degree, static_cov=None, tv_cov=None):
        self.dist = "dropout"
        self.ylabel = "Dropout probabilities"
        self.degree = degree
        self.static_cov = static_cov if static_cov is not None else []
        self.tv_cov = tv_cov if tv_cov is not None else []

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
            pre_drop_mask = y == 0
            dropout_mask = y == 1
            ll_obs[pre_drop_mask, k] = np.log(1 - p_k[pre_drop_mask])
            ll_obs[dropout_mask, k] = np.log(p_k[dropout_mask])

        # Convert to DataFrame and attach unit ID for grouping
        ll_obs = pd.DataFrame(ll_obs)
        grouped_ll = ll_obs.groupby(units, sort=False).sum()

        return grouped_ll
