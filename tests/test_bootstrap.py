import numpy as np
import pandas as pd

from gbm_sim.bootstrap import bootstrap_path
from gbm_sim.simulation import simulate_gbm_path

def make_prices(seed=0):
    paths = simulate_gbm_path(s0=100, mu=0.1, sigma=0.2,t=2.0,
                                dt=1/252, n_path=1, seed=seed)
    return pd.Series(paths[0])

def test_first_col_s0():
    path=bootstrap_path(make_prices(),s0=123,t=10,
                                      n_path=20, seed=1)
    
    assert np.all(path[:, 0] == 123)

def test_reproducible_with_seed():
    p1 = bootstrap_path(make_prices(), s0=100,t=10,
                                   n_path=5, seed=42)
    p2 = bootstrap_path(make_prices(), s0=100,t=10,
                                   n_path=5, seed=42)
    np.testing.assert_array_equal(p1, p2)