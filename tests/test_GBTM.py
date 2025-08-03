import pytest
from gbtm.models import GBTM


class TestGBTM:
    def setup_method(self, method):
        print(f"Setting up {method}")
        self.model = GBTM(
            data,
            outcome_col,
            time_col,
            unit_col,
            K,
            degree,
            model,
            static_cov,
            tv_cov,
            max_iter,
            tol=1e-4,
            verbose=True,
            seed=42,
        )

    def teardown_method(self, method):
        print(f"Tearing down {method}")
        del self.model

    def test_init(self):
        assert True
