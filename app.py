import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time 

from gbm_sim.data import fetch_data
from gbm_sim.calibration import calibrate_gbm
from gbm_sim.simulation import simulate_gbm_path
from gbm_sim.bootstrap import bootstrap_path
from gbm_sim.diagnostic import analyze_return, qq_plot
from gbm_sim.backtest import rolling_backtest, hit_rate,to_dataframe

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="GBM Stock Simulator", layout="wide")
st.title("GBM Stock Price Simulator")
st.caption("Calibrates drift and volatility from real historical data, "
           "simulates future price paths, and tests GBM's assumptions.")

# ── Sidebar controls ─────────────────────────────────────────────────────────
st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Enter Valid Ticker").upper()
start  = st.sidebar.date_input("Calibration start", value=pd.Timestamp("2020-01-01"))
end    = st.sidebar.date_input("Calibration end",   value=time.strftime("%Y-%m-%d", time.localtime()))
n_paths   = st.sidebar.slider("Simulated paths", 100, 5000, 1000, step=100)
horizon   = st.sidebar.slider("Forecast horizon (trading days)", 21, 252, 63)

if st.sidebar.button("Run"):
    st.session_state["ran"] = True

if st.session_state.get("last_ticker") != ticker:
    st.session_state["ran"] = False
    st.session_state["last_ticker"] = ticker

if not st.session_state.get("ran"):
    st.info("Configure settings in the sidebar and click **Run**.")
    st.stop()

with st.spinner(f"Fetching {ticker} data..."):
    try:
        prices = fetch_data(ticker, str(start), str(end))
    except Exception as e:
        st.error(f"Could not fetch data: {e}")
        st.stop()

params = calibrate_gbm(prices)
s0     = float(prices.iloc[-1])
dt     = 1 / 252
years  = horizon * dt

col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Price",  f"${s0:.2f}")
col2.metric("Annual Drift (μ)",  f"{params.mu:.1%}")
col3.metric("Annual Volatility (σ)", f"{params.sigma:.1%}")
col4.metric("Forecast Horizon", f"{horizon} days")

st.divider()

# ── Simulate ─────────────────────────────────────────────────────────────────
gbm_paths  = simulate_gbm_path(s0, params.mu, params.sigma,
                                years, dt, n_paths, seed=42)
boot_paths = bootstrap_path(prices, s0, horizon, n_paths, seed=42)

# ── Tab layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3,tab4 = st.tabs(["Simulated Paths", "Model Comparison", "Diagnostics","Backtest"])

# ── Tab 1: Simulated paths ────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{ticker} — GBM Simulated Paths")

    fig, ax = plt.subplots(figsize=(10, 4))

    for i in range(min(200, n_paths)):
        ax.plot(gbm_paths[i], color="steelblue", alpha=0.03, linewidth=0.8)

    mean_path = gbm_paths.mean(axis=0)
    p5  = np.percentile(gbm_paths, 5,  axis=0)
    p95 = np.percentile(gbm_paths, 95, axis=0)

    ax.plot(mean_path, color="navy",      linewidth=2,   label="Mean path")
    ax.fill_between(range(horizon + 1), p5, p95,
                    color="steelblue", alpha=0.2, label="90% band")
    ax.axhline(s0, color="gray", linestyle="--", linewidth=1, label="S₀")
    ax.set_xlabel("Trading days")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

    # Final price distribution
    st.subheader("Distribution of Final Prices")
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.hist(gbm_paths[:, -1], bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(s0, color="gray",   linestyle="--", label="S₀")
    ax2.axvline(mean_path[-1], color="navy", linewidth=2, label="Mean")
    ax2.set_xlabel("Final Price ($)")
    ax2.set_ylabel("Count")
    ax2.legend()
    st.pyplot(fig2)

# ── Tab 2: Model comparison ───────────────────────────────────────────────────
with tab2:
    st.subheader("GBM vs Bootstrap — Final Price Distribution")
    st.caption("Bootstrap resamples real historical daily returns instead of "
               "assuming normality. Wider tails = more fat-tail-aware.")

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.hist(gbm_paths[:, -1],  bins=60, alpha=0.6, color="steelblue", label="GBM")
    ax3.hist(boot_paths[:, -1], bins=60, alpha=0.6, color="tomato",    label="Bootstrap")
    ax3.set_xlabel("Final Price ($)")
    ax3.set_ylabel("Count")
    ax3.legend()
    st.pyplot(fig3)

    # Summary table
    def summarise(paths, label):
        final = paths[:, -1]
        p5, p95 = np.percentile(final, [5, 95])
        return {"Model": label,
                "Mean": round(final.mean(), 2),
                "P5":   round(p5, 2),
                "P95":  round(p95, 2),
                "Band width": round(p95 - p5, 2)}

    st.dataframe(
        pd.DataFrame([summarise(gbm_paths, "GBM"),
                      summarise(boot_paths, "Bootstrap")]).set_index("Model"),
        use_container_width=True,
    )

# ── Tab 3: Diagnostics ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Are Returns Actually Normal? (GBM's Core Assumption)")

    dist = analyze_return(prices)

    c1, c2, c3 = st.columns(3)
    c1.metric("Skewness",        round(dist.skewness, 3))
    c2.metric("Kurtosis", round(dist.kurtosis, 3),
              help="0 = normal. >0 = fat tails.")
    c3.metric("Jarque-Bera p-value", round(dist.jb_pvalue, 5),
              help="< 0.05 = reject normality.")

    if dist.jb_pvalue < 0.05:
        st.warning("Normality rejected — GBM's distributional assumption "
                   "does not hold for this asset.")
    else:
        st.success("Normality not rejected — GBM is a reasonable fit here.")

    # QQ plot
    st.subheader("QQ Plot — Log Returns vs Normal Distribution")
    st.caption("Points on the diagonal = returns are normal. "
               "S-curve at ends = fat tails.")

    theo, emp = qq_plot(prices)
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.scatter(theo, emp, s=5, color="steelblue", alpha=0.5)
    lim = max(abs(theo.min()), abs(theo.max()))
    ax4.plot([-lim, lim], [-lim, lim], color="tomato", linewidth=1.5, label="y = x")
    ax4.set_xlabel("Theoretical quantiles")
    ax4.set_ylabel("Empirical quantiles")
    ax4.legend()
    st.pyplot(fig4)

with tab4:
    st.subheader("Rolling Backtest — Would This Have Worked Historically?")
    st.caption(
        "At each window, the model trains on the preceding year of data and "
        "predicts the next N days. We check if the actual price landed inside "
        "the model's 90% confidence band. A well-calibrated model should hit ~90%."
    )

    horizon_bt = st.slider("Backtest forecast horizon (days)", 21, 126, 63, key="bt_horizon")
    step_bt    = st.slider("Slide window every N days", 5, 63, 21, key="bt_step")

    if st.button("Run Backtest"):                        
        with st.spinner("Running rolling windows..."):
            from gbm_sim.backtest import rolling_backtest, to_dataframe, hit_rate

            results = rolling_backtest(
                prices,
                calib_days=252,
                horizon_days=horizon_bt,
                step_days=step_bt,
                n_paths=500,
                seed=42,
            )
            df_bt = to_dataframe(results)
            rate  = hit_rate(results)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Windows tested", len(results))
        col_b.metric("Hit rate (90% band)", f"{rate:.1%}")
        col_c.metric("Target", "90.0%")

        if rate >= 0.80:
            st.success(f"Model captured the actual outcome in {rate:.1%} of windows.")
        else:
            st.warning(
                f"Hit rate {rate:.1%} is below 80% — bands are too narrow. "
                "Expected when real returns have fat tails (see Diagnostics tab)."
            )

        st.subheader("Actual Price vs Predicted Band (per window)")
        fig, ax = plt.subplots(figsize=(10, 4))
        window_ends = list(range(len(df_bt)))
        ax.fill_between(window_ends, df_bt["p5"], df_bt["p95"],
                        alpha=0.3, color="steelblue", label="90% band")
        ax.plot(window_ends, df_bt["sim_mean"],
                color="navy", linewidth=1.5, label="Sim mean")
        ax.scatter(window_ends, df_bt["actual_price"],
                   color=df_bt["inside_band"].map({True: "green", False: "red"}),
                   s=20, zorder=5, label="Actual (green=inside, red=outside)")
        ax.set_xlabel("Backtest window index")
        ax.set_ylabel("Price ($)")
        ax.legend(fontsize=8)
        st.pyplot(fig)

        with st.expander("See all window results"):
            st.dataframe(df_bt, use_container_width=True)