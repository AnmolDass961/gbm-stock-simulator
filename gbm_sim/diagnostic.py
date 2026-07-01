import numpy as np
import pandas as pd 
from scipy import stats
from dataclasses import dataclass

@dataclass(frozen=True)
class ReturnsDistribution:
    skewness:float
    kurtosis:float
    jb_stat:float
    jb_pvalue:float

def analyze_return(prices:pd.Series)->ReturnsDistribution:
    log_returns=np.log(prices/prices.shift(1)).dropna()
    jb_stat,jb_pvalue=stats.jarque_bera(log_returns)
    return ReturnsDistribution(
        skewness=stats.skew(log_returns),
        kurtosis=stats.kurtosis(log_returns),
        jb_stat=jb_stat,
        jb_pvalue=jb_pvalue
    )

def qq_plot(prices:pd.Series):
    log_returns=np.log(prices/prices.shift(1)).dropna()
    standardized=np.sort((log_returns-log_returns.mean())/(log_returns.std()))
    n=len(standardized)
    probs = (np.arange(1, n + 1) - 0.5) / n   
    theoretical_quantiles = stats.norm.ppf(probs)
    return theoretical_quantiles,standardized


