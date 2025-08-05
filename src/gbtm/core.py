import numpy as np
import pandas as pd
from scipy.special import logsumexp
from gbtm.utils import (
    create_fullgrid,
    create_dropout,
    kmeans_data_prep,
    kmeans_clustering,
)
import matplotlib.pyplot as plt


class GBTM:
    def __init__(
        self,
        data: pd.DataFrame,
        K: int,
        degree: int,
        time_col: str,
        unit_col: str,
        outcome_models: dict,
        dropout_models: dict = None,
        static_cov: list = None,
        tv_cov: list = None,
        max_iter: int = 100,
        tol: float = 1e-8,
        verbose: bool = True,
        seed: int = 42,
    ):
        self.data = data.copy()
        self.K = K
        self.degree = degree
        self.time_col = time_col
        self.unit_col = unit_col
        self.outcome_models = outcome_models
        self.dropout_models = dropout_models if dropout_models is not None else {}
        self.static_cov = static_cov if static_cov is not None else []
        self.tv_cov = tv_cov if tv_cov is not None else []
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.seed = seed

        # initialize other parameters
        self.pi = np.full(self.K, 1 / self.K)  # prior probabilities

        # intialize full_grid and units
        self._full_grid = create_fullgrid(self.data, self.unit_col, self.time_col)
        self._units = self._full_grid[self.unit_col]

        # create dropout outcomes in full_grid
        for outcome in self.dropout_models:
            self._full_grid = create_dropout(
                self._full_grid, self.unit_col, self.time_col, outcome
            )

        # create initial clusters for fullgrid
        wide_data_index, scaled_df = kmeans_data_prep(
            self._full_grid,
            self.unit_col,
            self.time_col,
            self.outcome_models,
            self.dropout_models,
        )
        unit_to_cluster_map = kmeans_clustering(scaled_df, wide_data_index, K, seed)
        self._clusters = self._full_grid[self.unit_col].map(unit_to_cluster_map)

        # initialize design matrixes and params for each outcome model
        self.outcome_params = {}
        for outcome in self.outcome_models:
            self.outcome_models[outcome].init_design_matrix(
                self._full_grid,
                self.degree,
                self.time_col,
                self.static_cov,
                self.tv_cov,
            )
            self.outcome_params[outcome] = self.outcome_models[outcome].init_params(
                self._full_grid[outcome], self._clusters, self.K
            )

        # initialize design matrixes and params for each dropout model
        self.dropout_params = {}
        for outcome, model in self.dropout_models.items():
            self.dropout_models[outcome].init_design_matrix(
                self._full_grid,
                model.degree,
                self.time_col,
                model.static_cov,
                model.tv_cov,
            )
            self.dropout_params[outcome] = self.dropout_models[outcome].init_params(
                self._full_grid[f"dropout_{outcome}"], self._clusters, self.K
            )

        # Initialize metrics
        self.assigned_groups = None

    def _group_loglikelihood(self):
        for i, outcome in enumerate(self.outcome_models):
            grouped_ll = self.outcome_models[outcome].log_likelihood(
                self._full_grid[outcome],
                self.K,
                self.outcome_params[outcome],
                self._units,
            )
            if i == 0:
                combined_ll = grouped_ll
            else:
                assert combined_ll.shape == grouped_ll.shape
                combined_ll += grouped_ll

        for outcome in self.dropout_models:
            grouped_ll = self.dropout_models[outcome].log_likelihood(
                self._full_grid[f"dropout_{outcome}"],
                self.K,
                self.dropout_params[outcome],
                self._units,
            )
            assert combined_ll.shape == grouped_ll.shape
            combined_ll += grouped_ll

        return combined_ll

    def _total_loglikelihood(self):
        combined_ll = self._group_loglikelihood()

        total = np.sum(logsumexp(combined_ll + np.log(self.pi + 1e-16), axis=1))

        return total

    def _e_step(self):
        # calculate the posterior probabilities of belonging to group Kpost_k (N * K)
        combined_ll = self._group_loglikelihood()
        log_weighted_lik = combined_ll + np.log(self.pi + 1e-16)
        log_post = log_weighted_lik - logsumexp(log_weighted_lik, axis=1, keepdims=True)
        self.post_wide = np.exp(
            log_post
        )  # leave as pd.dataframe to obtain id to assigned_group map later
        self.post_long = np.exp(
            pd.merge(
                self._units.to_frame(),
                log_post.reset_index(),
                how="left",
                on=self.unit_col,
            )
            .drop(columns=self.unit_col)
            .values
        )

    def _m_step(self):
        self.pi = self.post_wide.values.mean(axis=0)  # update prior probabilities

        # update params
        for outcome in self.outcome_models:
            self.outcome_params[outcome] = self.outcome_models[outcome].maximize(
                self._full_grid[outcome],
                self.K,
                self.post_long,
                self.outcome_params[outcome],
            )
        for outcome in self.dropout_models:
            self.dropout_params[outcome] = self.dropout_models[outcome].maximize(
                self._full_grid[f"dropout_{outcome}"],
                self.K,
                self.post_long,
                self.dropout_params[outcome],
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

        # assign groups based on highest posterior probability
        self.assigned_groups = pd.Series(
            np.argmax(self.post_wide, axis=1), index=self.post_wide.index
        )

        self._assigned_groups_long = np.argmax(self.post_long, axis=1)

        # 2. Calculate BIC and AIC
        num_params = 0
        for outcome in self.outcome_models:
            num_params += self.outcome_models[outcome].num_estimated_params(self.K)
        for outcome in self.dropout_models:
            num_params += self.dropout_models[outcome].num_estimated_params(self.K)

        num_params += self.K - 1  # Add K-1 for pi
        self.bic = -2 * ll_new + num_params * np.log(len(self.assigned_groups.values))
        self.aic = -2 * ll_new + 2 * num_params

        # 3. Calculate APPA_j = (1/n_j) * sum_{i in group_j} P(z_i = j | ...)
        self.appa = np.zeros(self.K)
        for k in range(self.K):
            individuals_in_group_k = np.where(self.assigned_groups.values == k)[0]
            if len(individuals_in_group_k) > 0:
                self.appa[k] = np.mean(self.post_wide.values[individuals_in_group_k, k])
            else:
                self.appa[k] = 0

        # 4. Calculate OCC_j = (APPA_j / (1 - APPA_j)) / (pi_j / (1 - pi_j))
        self.occ = np.zeros(self.K)
        for k in range(self.K):
            odds_appa_k = self.appa[k] / (1 - self.appa[k] + 1e-16)
            odds_pi_k = self.pi[k] / (1 - self.pi[k] + 1e-16)
            self.occ[k] = odds_appa_k / np.clip(odds_pi_k, 1e-16, np.inf)

        # 5. Calculate EIC = -sum_{i,j} P(z_i = j | A_iK) log P(z_i = j | A_iK)
        self.eic = -np.sum(
            (self.post_wide.values * np.log(self.post_wide.values + 1e-16))
        )

        # Print final metrics
        print(f"  Final Log-Likelihood: {ll_new:.2f}")
        print(f"  BIC: {self.bic:.2f}")
        print(f"  AIC: {self.aic:.2f}")
        print(f"  APPA: {self.appa}")
        print(f"  OCC: {self.occ}")
        print(f"  Entropy Information Criterion: {self.eic:.4f}")

    def plot_univariate_trajectories(
        self,
        outcome,
        dropout=False,
        title=None,
        y_label=None,
        x_label=None,
        plot_confidence=True,
        plot_spaghetti=False,
        num_samples=5,
        ax=None,
    ):
        """
        Plot the mean predicted trajectory for each group for a single outcome.

        Parameters:
            outcome (str): The name of the outcome to plot.
            dropout (bool): Whether to plot dropout probabilities.
            title (str): Optional title for the plot.
            y_label (str): Optional y-axis label.
            x_label (str): Optional x-axis label.
            plot_confidence (bool): Whether to plot confidence intervals.
            plot_spaghetti (bool): Whether to plot individual trajectories.
            num_samples (int): Number of individual trajectories to plot if `plot_spaghetti` is True.
            ax (matplotlib.axes.Axes): Optional axes to plot on.
        """
        # ensure that outcome is in the models

        if dropout:
            assert outcome in self.dropout_models, f"'{outcome}' not in dropout models."
            model = self.dropout_models[outcome]
            params = self.dropout_params[outcome]
            actual = self._full_grid[f"dropout_{outcome}"].values
        else:
            assert outcome in self.outcome_models, f"'{outcome}' not in outcome models."
            model = self.outcome_models[outcome]
            params = self.outcome_params[outcome]
            actual = self._full_grid[outcome].values

        # Build DataFrame to group over time and group
        df = self._full_grid[[self.unit_col, self.time_col]].copy()
        df["expected"] = model.predict_expected_value(
            params, self._assigned_groups_long
        )
        df["group"] = self._assigned_groups_long
        df["actual"] = actual

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        mean_df = (
            df.groupby([self.time_col, "group"], sort=False)["expected"]
            .mean()
            .reset_index()
        )
        std_df = (
            df.groupby([self.time_col, "group"], sort=False)["actual"]
            .std()
            .reset_index()
        )

        for k in range(self.K):
            group_mean = mean_df[mean_df["group"] == k]
            ax.plot(
                group_mean[self.time_col],
                group_mean["expected"],
                label=f"Group {k}",
                color=f"C{k}",
            )

            if plot_confidence:
                group_std = std_df[std_df["group"] == k]
                ax.fill_between(
                    group_mean[self.time_col],
                    group_mean["expected"] - 1.96 * group_std["actual"],
                    group_mean["expected"] + 1.96 * group_std["actual"],
                    alpha=0.2,
                )

            if plot_spaghetti:
                unique_units = df[df["group"] == k][self.unit_col].unique()
                sampled_units = np.random.choice(
                    unique_units,
                    size=min(num_samples, len(unique_units)),
                    replace=False,
                )
                for unit in sampled_units:
                    ind_df = df[(df[self.unit_col] == unit) & (df["group"] == k)]
                    ax.plot(
                        ind_df[self.time_col],
                        ind_df["actual"],
                        color=f"C{k}",
                        alpha=0.5,
                        linestyle="--",
                        linewidth=0.8,
                    )

        ax.set_xlabel(x_label if x_label else self.time_col)
        ax.set_ylabel(y_label if y_label else model.ylabel)
        ax.set_title(title if title else f"Mean Trajectories: {outcome}")
        ax.legend()
        plt.tight_layout()
        return ax

    def plot_multivariate_trajectories(
        self,
        outcome_list,
        dropout=False,
        title=None,
        y_label=None,
        x_label=None,
        plot_confidence=True,
        plot_spaghetti=False,
        num_samples=5,
        fig=None,
        axs=None,
    ):
        """
        Plot mean predicted trajectories for each group across multiple outcomes.

        Parameters:
            outcome_list (list): List of outcome names to plot.
            dropout (bool): Whether to use dropout models.
            title (str): Title for the entire figure.
            y_label (str): Y-axis label (overrides individual outcome labels if provided).
            x_label (str): X-axis label.
            plot_confidence (bool): Whether to plot confidence intervals.
            plot_spaghetti (bool): Whether to include individual trajectories.
            num_samples (int): Number of individuals to include in spaghetti plots.
            fig (matplotlib.figure.Figure): Optional figure to plot on.
            axs (np.ndarray or list): Optional array of axes to plot on.
        """
        n_rows = len(outcome_list)
        n_cols = self.K

        if fig is None or axs is None:
            fig, axs = plt.subplots(
                n_rows,
                n_cols,
                figsize=(5 * n_cols, 4 * n_rows),
                sharex="col",
                sharey="row",
            )

        # Ensure axs is always a 2D array, even for single plots
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs.reshape(1, -1)
        elif n_cols == 1:
            axs = axs.reshape(-1, 1)
        else:
            # For the case where axs is already a 2D array from subplots
            axs = np.asarray(axs)

        for i, outcome in enumerate(outcome_list):
            if dropout:
                assert outcome in self.dropout_models, (
                    f"'{outcome}' not in dropout models."
                )
                model = self.dropout_models[outcome]
                params = self.dropout_params[outcome]
                actual = self._full_grid[f"dropout_{outcome}"].values
            else:
                assert outcome in self.outcome_models, (
                    f"'{outcome}' not in outcome models."
                )
                model = self.outcome_models[outcome]
                params = self.outcome_params[outcome]
                actual = self._full_grid[outcome].values

            # Build DataFrame to group over time and group
            df = self._full_grid[[self.unit_col, self.time_col]].copy()
            df["expected"] = model.predict_expected_value(
                params, self._assigned_groups_long
            )
            df["group"] = self._assigned_groups_long
            df["actual"] = actual

            # create mean and std DataFrames
            mean_df = (
                df.groupby([self.time_col, "group"], sort=False)["expected"]
                .mean()
                .reset_index()
            )

            std_df = (
                df.groupby([self.time_col, "group"], sort=False)["actual"]
                .std()
                .reset_index()
            )

            for k in range(self.K):
                ax = axs[i, k]
                group_mean = mean_df[mean_df["group"] == k]

                ax.plot(
                    group_mean[self.time_col],
                    group_mean["expected"],
                    label=f"Group {k}",
                    color=f"C{k}",
                )

                if plot_confidence:
                    group_std = std_df[std_df["group"] == k]
                    ax.fill_between(
                        group_mean[self.time_col],
                        group_mean["expected"] - 1.96 * group_std["actual"],
                        group_mean["expected"] + 1.96 * group_std["actual"],
                        alpha=0.2,
                    )

                if plot_spaghetti:
                    unique_units = df[df["group"] == k][self.unit_col].unique()
                    sampled_units = np.random.choice(
                        unique_units,
                        size=min(num_samples, len(unique_units)),
                        replace=False,
                    )
                    for unit in sampled_units:
                        ind_df = df[(df[self.unit_col] == unit) & (df["group"] == k)]
                        ax.plot(
                            ind_df[self.time_col],
                            ind_df["actual"],
                            color=f"C{k}",
                            alpha=0.5,
                            linestyle="--",
                            linewidth=0.8,
                        )

                if i == n_rows - 1:
                    ax.set_xlabel(x_label or self.time_col)
                if k == 0:
                    ax.set_ylabel(y_label or model.ylabel)
                ax.set_title(f"{outcome} - Group {k}")

        if title:
            fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
