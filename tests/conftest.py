import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def variable_dummy_data():
    def _data_generator(N, K, degree, units):
        assert N % units == 0, "N must be divisible by units"
        np.random.seed(42)
        X = pd.DataFrame()
        for d in range(degree + 1):
            X[f"x{d}"] = np.random.normal(0, 1, N)
        y_binary = np.random.binomial(1, 0.3, N)
        y_continuous = np.random.normal(0, 1, N)
        y_zip = np.random.poisson(1.0, N)
        y_zip[np.random.rand(N) < 0.3] = 0  # Inject excess zeros
        clusters = np.repeat(np.random.choice(K, units), N / units)
        units = np.repeat(np.arange(units), N / units)
        return X, y_binary, y_continuous, y_zip, clusters, units
    return _data_generator
