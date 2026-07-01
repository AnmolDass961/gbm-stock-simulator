from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class gbm_params:
    mu:float
    sigma:float
    mu_daily:float
    sigma_daily:float

def calibrate_gbm(prices:pd.Series,trading_days=252)->gbm_params:
    log_returns=np.log(prices/prices.shift(1)).dropna()
    mu_daily=log_returns.mean()
    sigma_daily=log_returns.std()

    return gbm_params(mu=mu_daily*trading_days,sigma=sigma_daily*np.sqrt(trading_days),mu_daily=mu_daily,sigma_daily=sigma_daily)
