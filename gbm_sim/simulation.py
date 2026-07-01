import numpy as np

def simulate_gbm_path(s0:float,mu:float,sigma:float,t:float,dt:float,n_path:int,seed:int)->np.ndarray:
    rng=np.random.default_rng(seed)
    n_days=int(round(t/dt))
    z=rng.standard_normal((n_path,n_days))
    calc=((mu-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*z)
    cum_sum=np.cumsum(calc,axis=1)
    path=s0*np.exp(cum_sum)
    return np.hstack([np.full((n_path, 1), s0), path])