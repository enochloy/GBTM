import numpy as np
from scipy.special import expit, gammaln
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import statsmodels.api as sm
from abc import ABC, abstractmethod


class DistributionModel(ABC):
    @abstractmethod
    def init_params(self, data, X, K, degree, seed):
        pass

    @abstractmethod
    def log_likelihood(self, data, X, K, params):
        pass

    @abstractmethod
    def maximize(self, data, X, K, post, params):
        pass

    @abstractmethod
    def num_estimated_params(self, K, degree):
        """Returns the number of estimated parameters for the specific model."""
        pass

    @abstractmethod
    def get_mean_trajectory(self, X, params):
        """
        Calculates the mean trajectory for each group based on the model's parameters.
        Returns a (K, T) array.
        """
        pass


class CensoredNormalModel(DistributionModel):
    def __init__(self, sigma=0.05, lower_bound=-np.inf, upper_bound=np.inf):
        assert sigma != 0, "Sigma must be greater than 0 for CensoredNormalModel."
        self.sigma = sigma
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dist = "cnorm"
        self.ylabel = "Value"

    def init_params(self, data, X, K, degree, seed=42):
        np.random.seed(seed)
        beta = np.zeros((K, degree + 1))

        kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
        cluster_assignments = kmeans.fit_predict(data)

        for k in range(K):
            cluster_data = data[cluster_assignments == k]

            if len(cluster_data) > 0:
                Y_cluster_stacked = cluster_data.flatten()
                X_cluster_stacked = np.tile(X, (len(cluster_data), 1))

                try:
                    model_ols = sm.OLS(Y_cluster_stacked, X_cluster_stacked).fit()
                    beta[k] = model_ols.params
                except Exception as e:
                    print(
                        f"Warning: OLS initial fit failed for CensoredNormal cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    beta[k, 0] = np.mean(cluster_data)
            else:
                beta[k, 0] = np.random.normal(0, 0.1)  # for future improvement
                print(
                    f"Warning: Cluster {k} was empty during CensoredNormal init. Using random intercept."
                )

        return {"beta": beta}

    def log_likelihood(self, data, X, K, params):
        N = data.shape[0]
        log_lik = np.zeros(shape=(N, K))
        beta = params["beta"]

        is_left_censored = data <= self.lower_bound
        is_right_censored = data >= self.upper_bound
        is_observed_interval = (data > self.lower_bound) & (data < self.upper_bound)

        for k in range(K):
            mu_k = np.broadcast_to(X @ beta[k], data.shape)
            ll_obs = np.zeros_like(data, dtype=float)

            ll_obs[is_left_censored] = norm.logcdf(
                self.lower_bound, loc=mu_k[is_left_censored], scale=self.sigma
            )
            ll_obs[is_right_censored] = norm.logsf(
                self.upper_bound, loc=mu_k[is_right_censored], scale=self.sigma
            )
            ll_obs[is_observed_interval] = norm.logpdf(
                data[is_observed_interval],
                loc=mu_k[is_observed_interval],
                scale=self.sigma,
            )

            log_lik[:, k] = np.sum(ll_obs, axis=1)

        return log_lik

    def neg_loglikelihood(self, beta_k, data, X, post_k):
        mu_k = np.broadcast_to(X @ beta_k, data.shape)

        is_left_censored = data <= self.lower_bound
        is_right_censored = data >= self.upper_bound
        is_observed_interval = (data > self.lower_bound) & (data < self.upper_bound)

        ll_obs = np.zeros_like(data, dtype=float)

        ll_obs[is_left_censored] = norm.logcdf(
            self.lower_bound, loc=mu_k[is_left_censored], scale=self.sigma
        )
        ll_obs[is_right_censored] = norm.logsf(
            self.upper_bound, loc=mu_k[is_right_censored], scale=self.sigma
        )
        ll_obs[is_observed_interval] = norm.logpdf(
            data[is_observed_interval], loc=mu_k[is_observed_interval], scale=self.sigma
        )

        # Sum ll_obs for each individual first
        ll_indiv = np.sum(ll_obs, axis=1)
        return -np.sum(post_k * ll_indiv)

    def maximize(self, data, X, K, post, params):
        beta = np.zeros_like(params["beta"])
        for k in range(K):
            # For fit, post[:,k] are the weights specific to group k for each individual
            results = minimize(
                self.neg_loglikelihood,
                params["beta"][k],
                args=(data, X, post[:, k]),  # Pass the correct posterior column
                method="L-BFGS-B",
            )
            beta[k] = results.x
        return {"beta": beta}

    def num_estimated_params(self, K, degree):
        # sigma is fixed here, so it's not an estimated parameter.
        return K * (degree + 1)

    def get_mean_trajectory(self, X, params):
        # For Censored Normal, the mean trajectory is directly X @ beta[k]
        beta = params["beta"]
        K = beta.shape[0]
        mean_trajectories = np.zeros((K, X.shape[0]))  # K groups, T time points
        for k in range(K):
            mean_trajectories[k, :] = X @ beta[k]
        return mean_trajectories


class BernoulliModel(DistributionModel):
    def __init__(self):
        self.dist = "bernoulli"
        self.ylabel = "Probability"

    def init_params(self, data, X, K, degree, seed=42):
        np.random.seed(seed)
        beta = np.zeros((K, degree + 1))
        kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
        cluster_assignments = kmeans.fit_predict(data)

        for k in range(K):
            cluster_data = data[cluster_assignments == k]
            if len(cluster_data) > 0:
                Y_cluster_stacked = cluster_data.flatten()
                X_cluster_stacked = np.tile(X, (len(cluster_data), 1))
                try:
                    model_log = sm.GLM(
                        Y_cluster_stacked,
                        X_cluster_stacked,
                        family=sm.families.Binomial(),
                    ).fit()
                    beta[k] = model_log.params
                except Exception as e:
                    # Fallback for small or problematic clusters
                    print(
                        f"Warning: GLM initial fit failed for Bernoulli KMeans cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    mean_prob = np.clip(np.mean(cluster_data), 1e-12, 1 - 1e-12)
                    beta[k, 0] = np.log(mean_prob / (1 - mean_prob))
            else:
                beta[k, 0] = np.random.normal(0, 0.1)  # for future improvement
                print(
                    f"Warning: KMeans cluster {k} was empty during Bernoulli init. Using random intercept."
                )
        return {"beta": beta}

    def log_likelihood(self, data, X, K, params):
        N = data.shape[0]
        log_lik = np.zeros(shape=(N, K))
        beta = params["beta"]

        for k in range(K):
            p_k = np.clip(expit(X @ beta[k]), 1e-12, 1 - 1e-12)
            ll_obs = data * np.log(p_k) + (1 - data) * np.log(1 - p_k)
            log_lik[:, k] = np.sum(ll_obs, axis=1)

        return log_lik

    def maximize(self, data, X, K, post, params):
        N, T = data.shape
        X_stack = np.tile(X, (N, 1))
        Y_stack = data.flatten()
        beta = np.zeros_like(params["beta"])

        for k in range(K):
            W_stack = np.repeat(post[:, k], T)
            model_log = sm.GLM(
                Y_stack, X_stack, family=sm.families.Binomial(), freq_weights=W_stack
            ).fit()
            beta[k] = model_log.params

        return {"beta": beta}

    def num_estimated_params(self, K, degree):
        return K * (degree + 1)

    def get_mean_trajectory(self, X, params):
        # For Bernoulli, the mean trajectory is the probability p_k = expit(X @ beta[k])
        beta = params["beta"]
        K = beta.shape[0]
        mean_trajectories = np.zeros((K, X.shape[0]))
        for k in range(K):
            mean_trajectories[k, :] = expit(X @ beta[k])
        return mean_trajectories


class ZIPModel(DistributionModel):
    def __init__(self):
        self.dist = "zip"
        self.ylabel = "Expected Count"

    def init_params(self, data, X, K, degree, seed=42):
        np.random.seed(seed)
        beta = np.zeros((K, degree + 1))
        gamma = np.zeros((K, degree + 1))

        kmeans = KMeans(n_clusters=K, random_state=seed, n_init="auto")
        cluster_assignments = kmeans.fit_predict(data)

        for k in range(K):
            cluster_data = data[cluster_assignments == k]
            if len(cluster_data) > 0:
                Y_cluster_stacked = cluster_data.flatten()
                X_cluster_stacked = np.tile(X, (len(cluster_data), 1))

                # initialize beta (Poisson process)
                try:
                    model_pois = sm.GLM(
                        Y_cluster_stacked,
                        X_cluster_stacked,
                        family=sm.families.Poisson(),
                    ).fit()
                    beta[k] = model_pois.params
                except Exception as e:
                    print(
                        f"Warning: GLM (beta) initial fit failed for ZIP KMeans cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    mean_rate = np.clip(np.mean(cluster_data), 1e-12, np.inf)
                    beta[k, 0] = np.log(mean_rate)  # Log link for Poisson
                    beta[k, 1:] = 0  # Ensure other coefficients are 0

                # initialize gamma (zero-inflation process)
                zero_indicator = (cluster_data == 0).astype(int)
                Y_gamma_stacked = zero_indicator.flatten()

                try:
                    model_log = sm.GLM(
                        Y_gamma_stacked,
                        X_cluster_stacked,
                        family=sm.families.Binomial(),
                    ).fit()
                    gamma[k] = model_log.params
                except Exception as e:
                    # Fallback for small or problematic clusters
                    print(
                        f"Warning: GLM (gamma) initial fit failed for ZIP KMeans cluster {k}. Using mean for intercept. Error: {e}"
                    )
                    mean_zero_prob = np.clip(np.mean(zero_indicator), 1e-12, 1 - 1e-12)
                    gamma[k, 0] = np.log(
                        mean_zero_prob / (1 - mean_zero_prob)
                    )  # Logit link for Binomial
                    gamma[k, 1:] = 0  # Ensure other coefficients are 0
            else:
                # fallback for empty cluster
                beta[k, 0] = np.random.normal(0, 0.1)  # for future improvement
                gamma[k, 0] = np.random.norma(0, 0.1)  # for future improvement
                print(
                    f"Warning: KMeans cluster {k} was empty during ZIP init. Using random intercepts for beta and gamma."
                )
        return {"beta": beta, "gamma": gamma}

    def log_likelihood(self, data, X, K, params):
        log_lik = np.zeros(shape=(data.shape[0], K))
        beta, gamma = params["beta"], params["gamma"]
        N, T = data.shape
        zero_mask = (data == 0).astype(int)  # Indicator for observed zeros

        for k in range(K):
            lam_k = np.exp(X @ beta[k])  # Poisson mean parameter (per time point)
            lam_k = np.tile(lam_k, (N, 1))
            p_k = np.clip(
                expit(X @ gamma[k]), 1e-12, 1 - 1e-12
            )  # Probability of being an excess zero (per time point)
            p_k = np.tile(p_k, (N, 1))

            # ZIP log-likelihood calculation based on the two components
            # If data is 0: log(p_t + (1-p_t) * exp(-lambda_t))
            # If data is > 0: log(1-p_t) - lambda_t + data*log(lambda_t) - gammaln(data+1) (Poisson PMF for non-zero)

            # Term for observed zeros (either from zero-inflation or Poisson)
            log_prob_zero_component = np.log(p_k + (1 - p_k) * np.exp(-lam_k))

            # Term for observed non-zeros (must come from Poisson component)
            log_prob_nonzero_component = (
                np.log(1 - p_k) - lam_k + data * np.log(lam_k) - gammaln(data + 1)
            )

            # Combine based on whether the observed data point is zero or not
            ll_obs = (
                zero_mask * log_prob_zero_component
                + (1 - zero_mask) * log_prob_nonzero_component
            )

            log_lik[:, k] = np.sum(
                ll_obs, axis=1
            )  # Sum log-likelihoods over time points for each individual

        return log_lik

    def maximize(self, data, X, K, post, params):
        # initialize parameters
        N, T = data.shape
        beta = np.zeros_like(params["beta"])  # poisson coefficients
        gamma = np.zeros_like(params["gamma"])  # zero-inflation coefficients
        X_stack_common = np.tile(
            X, (N, 1)
        )  # Design matrix stacked for all individuals and time points
        zero_mask = (data == 0).astype(int)  # Indicator for observed zeros

        for k in range(K):
            lam_k = np.exp(X @ params["beta"][k])
            lam_k = np.tile(lam_k, (N, 1))

            p_k = np.clip(expit(X @ params["gamma"][k]), 1e-12, 1 - 1e-12)
            p_k = np.tile(p_k, (N, 1))

            # Z_approx is the posterior probability that an observed zero came from the zero-inflation component
            denominator_Z = p_k + (1 - p_k) * np.exp(-lam_k)
            Z_approx = zero_mask * (p_k / np.clip(denominator_Z, 1e-12, np.inf))

            # --- M-step for gamma (zero-inflation part) ---
            Y_stack_gamma = Z_approx.flatten()
            W_stack_gamma = np.repeat(post[:, k], T)

            model_log = sm.GLM(
                Y_stack_gamma,
                X_stack_common,
                family=sm.families.Binomial(),
                freq_weights=W_stack_gamma,
            ).fit()
            gamma[k] = model_log.params

            # --- M-step for beta (Poisson part) ---
            Y_stack_beta = data.flatten()
            weights_pois = (
                (1 - Z_approx) * post[:, k, np.newaxis]
            ).flatten()  # The weights for this regression are post_k * (1 - Z_approx)
            model_pois = sm.GLM(
                Y_stack_beta,
                X_stack_common,
                family=sm.families.Poisson(),
                freq_weights=weights_pois,
            ).fit()
            beta[k] = model_pois.params

        return {"beta": beta, "gamma": gamma}

    def num_estimated_params(self, K, degree):
        # K * (degree + 1) for beta (Poisson part)
        # K * (degree + 1) for gamma (zero-inflation part)
        return K * (degree + 1) * 2

    def get_mean_trajectory(self, X, params):
        # For ZIP, the mean trajectory is E[Y] = (1 - p_zero) * lambda
        beta = params["beta"]
        gamma = params["gamma"]
        K = beta.shape[0]
        mean_trajectories = np.zeros((K, X.shape[0]))
        for k in range(K):
            lambda_k = np.exp(X @ beta[k])
            p_zero_k = expit(X @ gamma[k])
            mean_trajectories[k, :] = (1 - p_zero_k) * lambda_k
        return mean_trajectories
