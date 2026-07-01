import numpy as np
import pandas as pd

def bootstrap_path(prices:pd.Series,s0,t,n_path,seed)->np.ndarray:
    log_returns=np.log(prices/prices.shift(1)).dropna()
    rng = np.random.default_rng(seed)
    sampled_returns = rng.choice(log_returns, size=(n_path,t), replace=True)
    cum_sum=np.cumsum(sampled_returns,axis=1)
    returns=s0*np.exp(cum_sum)
    return np.hstack([np.full((n_path,1),s0),returns])