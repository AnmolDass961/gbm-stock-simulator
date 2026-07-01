import numpy as np
import pandas as pd

from dataclasses import dataclass
from gbm_sim.calibration import calibrate_gbm
from gbm_sim.simulation import simulate_gbm_path

TRADING_DAYS = 252

@dataclass
class WindowResult:
    window_start: str      # calibration window start date
    window_end: str        # calibration window end date (= forecast start)
    s0: float              # price at forecast start
    actual_price: float    # real price at forecast end
    sim_mean: float
    p5: float
    p95: float
    inside_band: bool


def rolling_backtest(
    prices: pd.Series,
    calib_days: int,
    horizon_days: int,
    step_days: int = 21,   # slide window forward this many days each iteration
    n_paths: int = 1000,
    seed: int = 42,
) -> list[WindowResult]:

    results = []
    n = len(prices)
    dt = 1 / TRADING_DAYS

    for i in range(calib_days, n - horizon_days, step_days):
        calib_slice = prices.iloc[i - calib_days: i]
        params = calibrate_gbm(calib_slice)

        s0 = float(prices.iloc[i - 1])
        actual = float(prices.iloc[i - 1 + horizon_days])

        paths = simulate_gbm_path(
            s0, params.mu, params.sigma,
            horizon_days * dt, dt, n_paths,
            seed=seed + i
        )
        final = paths[:, -1]
        p5, p95 = np.percentile(final, [5, 95])

        results.append(WindowResult(
            window_start=str(prices.index[i - calib_days].date()),
            window_end=str(prices.index[i - 1].date()),
            s0=round(s0, 2),
            actual_price=round(actual, 2),
            sim_mean=round(float(final.mean()), 2),
            p5=round(float(p5), 2),
            p95=round(float(p95), 2),
            inside_band=bool(p5 <= actual <= p95),
        ))

    return results


def to_dataframe(results: list[WindowResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def hit_rate(results: list[WindowResult]) -> float:
    """Fraction of windows where actual landed inside 90% band.
    A well-calibrated model should be close to 0.90."""
    return float(np.mean([r.inside_band for r in results]))