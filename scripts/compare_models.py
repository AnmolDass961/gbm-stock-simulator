import numpy as np
import pandas as pd

from gbm_sim.calibration import calibrate_gbm
from gbm_sim.data import fetch_data
from gbm_sim.bootstrap import bootstrap_path
from gbm_sim.simulation import simulate_gbm_path

def score_params(final_price:np.ndarray,actual_price)->dict:
    p5,p95=np.percentile(final_price,[5,95])
    return{
        'sim_mean':round(final_price.mean(),2),
        'p5':round(p5,2),
        'p95':round(p95,2),
        'actual_price':round(actual_price,2),
        '90% CI':bool(p5<actual_price<p95),
        'abs_error':round(abs(final_price.mean()-actual_price),2)
    }

def compare_models(full_price:pd.Series,training_days,days,n_path,seed=1)->pd.DataFrame:
    train=full_price.iloc[:training_days]
    s0=train.iloc[-1]
    actual_price=full_price.iloc[training_days-1+days]
    dt=1/252
    years=days*dt

    params=calibrate_gbm(train)
    gbm_path=simulate_gbm_path(s0,params.mu,params.sigma,years,dt,n_path,seed=1)
    gbm_score=score_params(gbm_path[:,-1],actual_price)
    gbm_score['model']='GBM'

    bootstrap=bootstrap_path(train,s0,days,n_path,seed=1)
    bootstrap_score=score_params(bootstrap[:,-1],actual_price)
    bootstrap_score['model']='Bootstrap'

    return pd.DataFrame([gbm_score,bootstrap_score]).set_index('model')

if __name__ == "__main__":
    ticker=['AAPL','TSLA','MSFT']
    for t in ticker:
        prices=fetch_data(t,'2024-01-01','2026-01-01')
        result = compare_models(prices,252,60,1000,seed=1)
        print(t)
        print(result)
