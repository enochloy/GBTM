from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class DistributionModel(ABC):
    def init_design_matrix(self, data, degree, time_col, static_cov=None, tv_cov=None):
        static_cov = static_cov if static_cov is not None else []
        tv_cov = tv_cov if tv_cov is not None else []

        # create empty dataframe with index from original long data
        df = pd.DataFrame(index=data.index)

        # populate time columns from t_0 to t_d
        for d in range(degree + 1):
            df[f"{time_col}{d}"] = data[time_col] ** d

        # populate static_cov and tv_cov
        df[static_cov + tv_cov] = data[static_cov + tv_cov].values

        # init design matrix
        self.X = df

    @abstractmethod
    def init_params(self, y, clusters, K):
        pass

    @abstractmethod
    def log_likelihood(self, y, K, params, units):
        pass

    @abstractmethod
    def maximize(self, y, K, post, params):
        pass

    def num_estimated_params(self, K):
        return K * self.X.shape[1]

    @abstractmethod
    def predict_expected_value(
        self, params: dict, assigned_groups: np.ndarray
    ) -> np.ndarray:
        """
        Return the expected value of y for each row in self.X given assigned group and parameters.
        """
        pass
