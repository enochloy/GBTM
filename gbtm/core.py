import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from .models import DistributionModel


class GBTM:
    def __init__(
        self,
        data,
        K,
        degree,
        model: DistributionModel,
        x_values=None,
        max_iter=100,
        tol=1e-4,
        verbose=True,
        seed=42,
    ):
        self.data = data
        self.N, self.T = data.shape
        self.K = K
        self.degree = degree
        self.model = model
        self.x_values = np.arange(self.T) + 1 if x_values is None else x_values
        self.X = self._design_matrix(self.x_values)
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.seed = seed
        self.pi = np.full(K, 1 / K)
        self.params = model.init_params(data, self.X, K, degree, seed=seed)

        # Initialize attributes for results
        self.assigned_groups = None
        self.appa = None
        self.entropy = None
        self.occ = None
        self.bic = None
        self.aic = None

    def _design_matrix(self, x_values):
        return np.vstack([x_values**d for d in range(self.degree + 1)]).T

    def _total_loglikelihood(self):
        total = np.sum(
            logsumexp(
                self.model.log_likelihood(self.data, self.X, self.K, self.params)
                + np.log(self.pi + 1e-16),
                axis=1,
            )
        )
        return total

    def _e_step(self):
        # calculate the posterior probabilities of belonging to group K, post_k (N * K)
        log_lik = self.model.log_likelihood(self.data, self.X, self.K, self.params)
        log_weighted_lik = log_lik + np.log(self.pi + 1e-16)
        log_post = log_weighted_lik - logsumexp(log_weighted_lik, axis=1, keepdims=True)
        self.post = np.exp(log_post)

    def _m_step(self):
        self.pi = self.post.mean(axis=0)  # update prior probabilities

        # update params
        self.params = self.model.maximize(
            self.data, self.X, self.K, self.post, self.params
        )

    def fit(self):
        # set seed
        np.random.seed(self.seed)

        # initialize previous log_likelihood
        ll_old = -np.inf

        # EM loop
        for i in range(self.max_iter):
            self._e_step()
            self._m_step()
            ll_new = self._total_loglikelihood()  # calculate new log likelihood

            if self.verbose:
                print("Iteration {}: Total log-likelihood: {:.2f}".format(i, ll_new))

            if ll_new - ll_old < self.tol:
                if self.verbose:
                    print(
                        f"Convergence achieved at iteration {i}. Log-likelihood improvement less than tolerance."
                    )
                break
            ll_old = ll_new
        else:
            if self.verbose:
                print(
                    f"Maximum iterations ({self.max_iter}) reached without convergence."
                )

        print(f"Completed EM Algorithm for {self.model.dist} model.")

        # --- Post-convergence calculations ---
        # 1. Assign groups
        self.assigned_groups = np.argmax(self.post, axis=1)

        # 2. Calculate BIC and AIC
        num_params = self.model.num_estimated_params(self.K, self.degree) + (
            self.K - 1
        )  # Add K-1 for pi
        self.bic = -2 * ll_new + num_params * np.log(self.N)
        self.aic = -2 * ll_new + 2 * num_params

        # 3. Calculate APPA_j = (1/n_j) * sum_{i in group_j} P(z_i = j | ...)
        self.appa = np.zeros(self.K)
        for k in range(self.K):
            individuals_in_group_k = np.where(self.assigned_groups == k)[0]
            if len(individuals_in_group_k) > 0:
                self.appa[k] = np.mean(self.post[individuals_in_group_k, k])
            else:
                self.appa[k] = 0

        # 4. Calculate OCC_j = (APPA_j / (1 - APPA_j)) / (pi_j / (1 - pi_j))
        self.occ = np.zeros(self.K)
        for k in range(self.K):
            odds_appa_k = self.appa[k] / (1 - self.appa[k] + 1e-16)
            odds_pi_k = self.pi[k] / (1 - self.pi[k] + 1e-16)
            self.occ[k] = odds_appa_k / np.clip(odds_pi_k, 1e-16, np.inf)

        # 5. Calculate EIC = -sum_{i,j} P(z_i = j | A_iK) log P(z_i = j | A_iK)
        self.eic = -np.sum(self.post * np.log(self.post + 1e-16))

        # Print final metrics
        print(f"  Final Log-Likelihood: {ll_new:.2f}")
        print(f"  BIC: {self.bic:.2f}")
        print(f"  AIC: {self.aic:.2f}")
        print(f"  APPA: {self.appa}")
        print(f"  OCC: {self.occ}")
        print(f"  Entropy Information Criterion: {self.eic:.4f}")

    def plot_trajectories(
        self,
        title=None,
        ylabel=None,
        xlabel="Time",
        show_raw_data=False,
        num_raw_to_show=5,
    ):
        """
        Plots the estimated mean trajectories for each group.

        Args:
            title (str, optional): Title for the plot. Defaults to a generic title.
            show_raw_data (bool, optional): Whether to plot some raw individual trajectories.
            num_raw_to_show (int, optional): Number of random individual trajectories to show if show_raw_data is True.
        """
        plt.figure(figsize=(12, 6))

        if title is None:
            title = f"{self.model.dist.capitalize()} Model: Estimated Trajectories"
        plt.title(title)

        # set ylabel
        if ylabel is None:
            plt.ylabel(self.model.ylabel)

        # set xlabel
        plt.xlabel(xlabel)

        if self.model.dist == "cnorm":
            if hasattr(self.model, "lower_bound") and self.model.lower_bound != -np.inf:
                plt.axhline(
                    y=self.model.lower_bound,
                    color="r",
                    linestyle=":",
                    label=f"Lower Bound {self.model.lower_bound}",
                )
            if hasattr(self.model, "upper_bound") and self.model.upper_bound != np.inf:
                plt.axhline(
                    y=self.model.upper_bound,
                    color="r",
                    linestyle=":",
                    label=f"Upper Bound {self.model.upper_bound}",
                )
        elif self.model.dist == "bernoulli":
            plt.ylim(-0.05, 1.05)  # Keep probabilities within 0-1 range

        colors = plt.cm.viridis(
            np.linspace(0, 1, self.K)
        )  # Use a color map for distinct group colors

        # Get mean trajectories from the model-specific method
        estimated_trajectories = self.model.get_mean_trajectory(self.X, self.params)

        for k in range(self.K):
            plt.plot(
                self.x_values,
                estimated_trajectories[k, :],
                color=colors[k],
                linestyle="-",
                linewidth=2,
                label=f"Group {k + 1} (N={np.sum(self.assigned_groups == k)})",
            )

        if show_raw_data:
            if self.assigned_groups is None:
                print("Warning: assigned_groups not available. Call .fit() first.")
            else:
                # Plot a random subset of individual trajectories colored by their assigned group
                random_indices = np.random.choice(
                    self.N, min(self.N, num_raw_to_show), replace=False
                )
                for idx in random_indices:
                    group_assigned = self.assigned_groups[idx]
                    plt.plot(
                        self.x_values,
                        self.data[idx, :],
                        color=colors[group_assigned],
                        linestyle="--",
                        alpha=0.8,
                        linewidth=1,
                    )
                plt.plot(
                    [],
                    [],
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=0.8,
                    label=f"Random {min(self.N, num_raw_to_show)} Raw Trajectories",
                )

        plt.legend()
        plt.tight_layout()
        plt.show()
