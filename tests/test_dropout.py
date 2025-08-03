import numpy as np
import pandas as pd
import pytest  # pyright: ignore[reportMissingImports]
from gbtm.distributions.dropout import DropoutModel


@pytest.fixture
def dropout_model():
    model = DropoutModel(degree=1)
    return model


@pytest.fixture
def sample_data():
    y = np.array([0, 0, 1, np.nan])
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    beta = np.array([[0, 0], [0, 0]])
    K = 2
    units = np.array([0, 0, 1, 1])
    return y, X, beta, K, units


def test_init(dropout_model):
    assert dropout_model.degree == 1
    assert isinstance(dropout_model.static_cov, list)
    assert isinstance(dropout_model.tv_cov, list)
    assert len(dropout_model.static_cov) == 0
    assert len(dropout_model.tv_cov) == 0


@pytest.mark.parametrize("N, K, degree, units", [(100, 3, 2, 25), (200, 5, 3, 50)])
def test_loglikelihood_shapes(dropout_model, variable_dummy_data, N, K, degree, units):
    X, y, _, _, clusters, units = variable_dummy_data(N, K, degree, units)
    dropout_model.X = X
    params = dropout_model.init_params(y, clusters, K)

    loglik = dropout_model.log_likelihood(y, K, params, units)
    assert loglik.shape == (len(np.unique(units)), K)
    assert not np.isnan(loglik.values).any()


def test_loglikelihood_values(dropout_model, sample_data):
    y, X, beta, K, units = sample_data
    dropout_model.X = X

    # Calculate the expected result manually
    # For k=0 and k=1, beta is [0, 0], so expit(X @ beta) will be 0.5 for all rows
    # log(1 - 0.5) = log(0.5) = -0.693147

    # For unit 0, y = [0, 0], sum of loglikelihoods should be -0.693147*2
    # for unit 1, y= [1, np.nan], sum of loglikelihoods should be -0.693147
    loglik = dropout_model.log_likelihood(y, K, {"beta": beta}, units)
    expected_loglik = pd.DataFrame(
        {0: [-1.38629436, -1.38629436], 1: [-0.69314718, -0.69314718]}
    ).T

    pd.testing.assert_frame_equal(loglik, expected_loglik, check_exact=False, rtol=1e-5)


@pytest.mark.parametrize("N, K, degree, units", [(100, 3, 2, 25), (200, 5, 3, 50)])
def test_maximize_shapes(dropout_model, variable_dummy_data, N, K, degree, units):
    X, y, _, _, _, units = variable_dummy_data(N, K, degree, units)
    dropout_model.X = X
    post = np.random.dirichlet(np.ones(K), size=N)
    params = dropout_model.maximize(
        y, K, post, {"beta": np.zeros(shape=(K, X.shape[1]))}
    )

    assert "beta" in params
    assert params["beta"].shape == (K, X.shape[1])


@pytest.mark.parametrize("N, K, degree, units", [(100, 3, 2, 25), (200, 5, 3, 50)])
def test_maximize_accept_nan(dropout_model, variable_dummy_data, N, K, degree, units):
    np.random.seed(42)
    X, y, _, _, _, units = variable_dummy_data(N, K, degree, units)
    dropout_model.X = X
    post = np.random.dirichlet(np.ones(K), size=N)
    y = y.astype(float)
    y[np.random.rand(N) < 0.3] = np.nan  # set 30% of y values to be null
    dropout_model.maximize(y, K, post, {"beta": np.zeros(shape=(K, X.shape[1]))})


@pytest.mark.parametrize("N, K, degree, units", [(100, 3, 2, 25), (200, 5, 3, 50)])
def test_likelihood_increases_after_maximize(
    dropout_model, variable_dummy_data, N, K, degree, units
):
    np.random.seed(42)
    X, y, _, _, clusters, units = variable_dummy_data(N, K, degree, units)
    dropout_model.X = X
    post = np.random.dirichlet(np.ones(K), size=len(y))

    # Initial params
    init_params = dropout_model.init_params(y, clusters, K)
    ll_before = dropout_model.log_likelihood(y, K, init_params, units).values.sum()

    # Maximized params
    new_params = dropout_model.maximize(y, K, post, init_params)
    ll_after = dropout_model.log_likelihood(y, K, new_params, units).values.sum()

    assert ll_after >= ll_before
