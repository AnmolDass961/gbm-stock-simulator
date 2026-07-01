import numpy as np
import pandas as pd

from gbm_sim.diagnostic import analyze_return,qq_plot
from gbm_sim.simulation import simulate_gbm_path

def make_normal_prices():
    paths = simulate_gbm_path(s0=100, mu=0.1, sigma=0.2, t=5,
                                dt=1/252, n_path=1,seed=1)
    return pd.Series(paths[0])

def test_normal_data():
    stats=analyze_return(make_normal_prices())
    assert stats.jb_pvalue > 0.05
    assert abs(stats.kurtosis) < 1.0

def test_qq_plot_data_same_length():
    theo, emp = qq_plot(make_normal_prices())
    assert len(theo) == len(emp)
