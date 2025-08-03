import pytest
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
from gbtm.distributions.bernoulli import BernoulliModel


@pytest.fixture
def bern_model():
    model = BernoulliModel()
    # Add some mock covariates (N=10, P=2)
    model.X = pd.DataFrame(np.random.randn(10, 2), columns=["x1", "x2"])
    return model


@pytest.fixture
def sample_data():
    np.random.seed(42)
    y = np.random.binomial(1, 0.6, size=10).astype(float)
    y[::3] = np.nan  # introduce some missing
    clusters = np.random.choice(2, size=10)
    return y, clusters


def test_init_params_shape(bern_model, sample_data):
    y, clusters = sample_data
    K = 2
    params = bern_model.init_params(y, clusters, K)
    assert "beta" in params
    assert params["beta"].shape == (K, bern_model.X.shape[1])
    assert np.isfinite(params["beta"]).all()


def test_init_params_with_perfect_separation(bern_model):
    bern_model.X = pd.DataFrame([[1], [1], [1], [1], [1]])  # all same
    y = np.array([1, 1, np.nan, 0, 1])
    clusters = np.array([0, 0, 1, 1, 1])  # cluster 1 is very small
    K = 2

    with pytest.warns(PerfectSeparationWarning):
        bern_model.init_params(y, clusters, K)


def test_log_likelihood_shape(bern_model):
    K = 2
    N = 10
    bern_model.X = pd.DataFrame(np.random.randn(N, 2))
    y = np.random.binomial(1, 0.5, size=N)
    units = np.arange(N)
    beta = np.random.randn(K, 2)

    ll = bern_model.log_likelihood(y, K, {"beta": beta}, units)
    assert ll.shape == (N, K)
    assert not ll.isnull().values.any()


def test_log_likelihood_values(bern_model):
    bern_model.X = pd.DataFrame([[0], [1], [2], [2], [2]])
    y = np.array([0, 1, 1, 1, np.nan])  # test robustness with missing value
    K = 1
    beta = np.array([[0.0]])  # sigmoid(0 * x) = 0.5 for all

    units = np.array([0, 1, 1, 2, 2])
    ll = bern_model.log_likelihood(y, K, {"beta": beta}, units)

    expected_ll = y * np.log(0.5) + (1 - y) * np.log(0.5)
    expected_df = pd.DataFrame({0: expected_ll}).groupby(units).sum()

    pd.testing.assert_frame_equal(ll, expected_df)


def test_log_likelihood_multigroup(bern_model):
    bern_model.X = pd.DataFrame(np.ones((4, 1)))  # force same X
    y = np.array([0, 1, 0, 1])
    units = np.array([0, 1, 2, 3])
    K = 2
    beta = np.array([[0.0], [2.0]])  # sigmoid(0)=0.5, sigmoid(2)=~0.88

    ll = bern_model.log_likelihood(y, K, {"beta": beta}, units)
    assert ll.shape == (4, K)
    assert ll[1][1] > ll[1][0]  # higher beta should match 1s better


def test_maximize_output_shape(bern_model):
    N, K = 10, 2
    bern_model.X = pd.DataFrame(np.random.randn(N, 2))
    y = np.random.binomial(1, 0.7, size=N)
    post = np.random.dirichlet(np.ones(K), size=N)
    beta_init = np.zeros((K, 2))

    new_params = bern_model.maximize(y, K, post, {"beta": beta_init})
    assert "beta" in new_params
    assert new_params["beta"].shape == (K, bern_model.X.shape[1])


def test_maximize_with_zero_weights(bern_model):
    N, K = 10, 2
    bern_model.X = pd.DataFrame(np.random.randn(N, 2))
    y = np.random.binomial(1, 0.5, size=N)
    post = np.zeros((N, K))
    post[:, 0] = 1.0  # all assigned to cluster 0

    beta_init = np.zeros((K, 2))
    new_params = bern_model.maximize(y, K, post, {"beta": beta_init})
    assert np.isfinite(new_params["beta"]).all()
