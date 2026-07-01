import numpy as np

from gbm_sim.simulation import simulate_gbm_path

def test_first_column_is_s0():
    paths = simulate_gbm_path(s0=150, mu=0.1, sigma=0.2,t=1.0,
                                dt=1/252, n_path=10, seed=1)
    assert np.all(paths[:, 0] == 150)

def test_prices_stay_positive():
    paths = simulate_gbm_path(s0=100, mu=0.1, sigma=0.5,t=2.0,
                                dt=1/252, n_path=200, seed=2)
    assert np.all(paths > 0)

def test_reproducible_with_seed():
    p1 = simulate_gbm_path(s0=100, mu=0.1, sigma=0.2,t=1.0,
                             dt=1/252, n_path=5, seed=99)
    p2 = simulate_gbm_path(s0=100, mu=0.1, sigma=0.2, t=1.0,
                             dt=1/252, n_path=5, seed=99)
    np.testing.assert_array_equal(p1, p2)