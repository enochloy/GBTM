import pandas as pd
import pytest

from gbtm.distributions.base import DistributionModel


class DummyDistributionModel(DistributionModel):
    def init_params(self, y, clusters, K):
        pass

    def log_likelihood(self, y, K, params, units):
        pass

    def maximize(self, y, K, post, params):
        pass

    def get_mean_trajectory(self, params):
        pass


@pytest.fixture()
def concrete_model():
    """
    Pytest fixture to provide instance of concrete class
    """
    return DummyDistributionModel()


@pytest.fixture()
def data():
    """
    Test data
    """
    data = pd.DataFrame(
        {
            "id": [1, 1, 2],
            "time": [1, 2, 2],
            "male": [1, 1, 0],
            "age": [50, 51, 60],
            "bp": [120, 130, 140],
        }
    )

    return data


def test_init_design_matrix_with_both_cov(concrete_model, data):
    concrete_model.init_design_matrix(
        data, degree=3, time_col="time", static_cov=["male"], tv_cov=["age", "bp"]
    )

    expected_X = pd.DataFrame(
        {
            "time0": [1, 1, 1],
            "time1": [1, 2, 2],
            "time2": [1, 4, 4],
            "time3": [1, 8, 8],
            "male": [1, 1, 0],
            "age": [50, 51, 60],
            "bp": [120, 130, 140],
        }
    )
    expected_cols = ["time0", "time1", "time2", "time3", "male", "age", "bp"]

    assert (concrete_model.X.index == data.index).all()
    assert list(concrete_model.X.columns) == expected_cols
    assert concrete_model.X.equals(expected_X)


def test_init_design_matrix_without_tv(concrete_model, data):
    concrete_model.init_design_matrix(
        data, degree=1, time_col="time", static_cov=["male"]
    )

    expected_X = pd.DataFrame(
        {"time0": [1, 1, 1], "time1": [1, 2, 2], "male": [1, 1, 0]}
    )
    expected_cols = ["time0", "time1", "male"]

    assert (concrete_model.X.index == data.index).all()
    assert list(concrete_model.X.columns) == expected_cols
    assert concrete_model.X.equals(expected_X)


def test_init_design_matrix_without_cov(concrete_model, data):
    concrete_model.init_design_matrix(data, degree=2, time_col="time")

    expected_X = pd.DataFrame(
        {
            "time0": [1, 1, 1],
            "time1": [1, 2, 2],
            "time2": [1, 4, 4],
        }
    )
    expected_cols = ["time0", "time1", "time2"]

    assert (concrete_model.X.index == data.index).all()
    assert list(concrete_model.X.columns) == expected_cols
    assert concrete_model.X.equals(expected_X)


@pytest.mark.parametrize("K", [1, 2, 3, 4, 5, 6, 7, 8])
def test_num_estimated_params(concrete_model, data, K):
    concrete_model.init_design_matrix(
        data, degree=2, time_col="time", static_cov=["male"], tv_cov=["age", "bp"]
    )
    assert concrete_model.num_estimated_params(K) == K * 6


def test_num_estimated_params_without_design_matrix(concrete_model):
    with pytest.raises(AttributeError):
        _ = concrete_model.num_estimated_params(K=3)
