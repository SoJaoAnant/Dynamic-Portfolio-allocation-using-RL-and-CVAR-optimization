"""
# evaluate.py — Evaluation Script for Trained PPO Portfolio Agent

This script evaluates a trained PPO agent on unseen test data
using the PortfolioEnv environment.

------------------------------------------------------------

## What this script does

- Loads a trained PPO model
- Loads normalization statistics (VecNormalize)
- Runs the model on test dataset
- Tracks portfolio performance over time
- Computes key financial metrics

Goal:
    Measure how well the trained agent generalizes to new data

------------------------------------------------------------

## Inputs Required

- test.csv                  → unseen market data
- ppo_portfolio_final.zip   → trained PPO model
- vec_normalize.pkl         → normalization stats from training

Important:
    The SAME normalization stats must be used.
    Otherwise results are meaningless.

------------------------------------------------------------

## Setup

- Environment created using PortfolioEnv
- Wrapped with DummyVecEnv + VecNormalize
- Normalization set to:
    training = False
    norm_reward = False

This ensures:
    - observations are normalized
    - rewards remain real (not scaled)

------------------------------------------------------------

## Evaluation Loop

For each timestep:
- Agent predicts action (portfolio weights)
- Environment executes trade
- Portfolio value is updated

Tracks:
    - daily portfolio value
    - daily returns
    - actions (optional)

------------------------------------------------------------

## Metrics Computed

### 1. Total Return
Final portfolio value vs initial capital

### 2. Sharpe Ratio
Risk-adjusted return:
    mean(return) / std(return)

Higher = better

### 3. Maximum Drawdown
Largest drop from peak portfolio value

Lower = safer

### 4. Volatility
Standard deviation of returns

------------------------------------------------------------

## Outputs

- Printed performance metrics:
    → learning curve
    → total return
    → sharpe ratio
    → max drawdown
    → volatility

- Plot:
    → portfolio value over time

- Optional:
    → saves results to CSV

------------------------------------------------------------

## How to Run

python test_agent.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from environment import PortfolioEnv

# ── Config ────────────────────────────────────────────────────────────────────
os.makedirs("results",       exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# Lambda variants you trained — adjust paths if needed
# Add more entries here as you train more lambda variants
# For now just lambda=1.0 using best_model (best val performance)
LAMBDA_VARIANTS = {
    "RL (λ=0.0)": "models/lambda0.0/models/ppo_portfolio_best/best_model",
    "RL (λ=0.5)": "models/lambda0.5/models/ppo_portfolio_best/best_model",
    "RL (λ=1.0)": "models/lambda1.0/models/ppo_portfolio_best/best_model",
    "RL (λ=2.0)": "models/lambda2.0/models/ppo_portfolio_best/best_model",
    "RL (λ=3.0)": "models/lambda3.0/models/ppo_portfolio_best/best_model",
}
# Mapping from model path → its vec_normalize file
# (when you train more lambdas, save each with its own vecnorm)
VECNORM_PATHS = {
    "models/lambda0.0/models/ppo_portfolio_best/best_model": "models/lambda0.0/models/vec_normalize.pkl",
    "models/lambda0.5/models/ppo_portfolio_best/best_model": "models/lambda0.5/models/vec_normalize.pkl",
    "models/lambda1.0/models/ppo_portfolio_best/best_model": "models/lambda1.0/models/vec_normalize.pkl",
    "models/lambda2.0/models/ppo_portfolio_best/best_model": "models/lambda2.0/models/vec_normalize.pkl",
    "models/lambda3.0/models/ppo_portfolio_best/best_model": "models/lambda3.0/models/vec_normalize.pkl",
}

EVAL_PATHS = {
    "0.0": "models/lambda0.0/logs/eval/evaluations.npz",
    "0.5": "models/lambda0.5/logs/eval/evaluations.npz",
    "1.0": "models/lambda1.0/logs/eval/evaluations.npz",
    "2.0": "models/lambda2.0/logs/eval/evaluations.npz",
    "3.0": "models/lambda3.0/logs/eval/evaluations.npz",
}

ENV_KWARGS = dict(
    episode_length   = 500,   # test set has 580 days; 500 fits safely with cvar_lookback buffer
    cvar_alpha       = 0.05,
    cvar_lookback    = 60,
    transaction_cost = 0.001,
)

RISK_FREE_RATE = 0.065   # RBI repo rate ~6.5% annualised

# Plot style
plt.rcParams.update({
    "figure.dpi"      : 150,
    "font.size"       : 11,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
})
COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800",
          "#00BCD4", "#E91E63", "#795548"]

# ── Helper functions ──────────────────────────────────────────────────────────

def compute_metrics(returns: np.ndarray, rf: float = RISK_FREE_RATE) -> dict:
    """
    Compute all portfolio performance metrics from a daily returns array.
    returns: array of daily portfolio returns (e.g. 0.01 = 1% gain)
    """
    returns = np.array(returns)
    n_days  = len(returns)

    # Annualised return (geometric compounding)
    total_return = np.prod(1 + returns) - 1
    ann_return   = (1 + total_return) ** (252 / n_days) - 1

    # Annualised volatility
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe ratio
    daily_rf = rf / 252
    excess   = returns - daily_rf
    sharpe   = (excess.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    # Sortino ratio (only downside volatility)
    downside = returns[returns < daily_rf] - daily_rf
    sortino  = (excess.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 else 0

    # CVaR at 95% confidence
    var_95  = np.percentile(returns, 5)
    tail    = returns[returns <= var_95]
    cvar_95 = -np.mean(tail) * np.sqrt(252) if len(tail) > 0 else 0   # annualised

    # Maximum drawdown
    cum_returns = np.cumprod(1 + returns)
    peak        = np.maximum.accumulate(cum_returns)
    drawdown    = (cum_returns - peak) / peak
    max_dd      = drawdown.min()

    # Calmar ratio (return / max drawdown)
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    # Skewness and kurtosis of return distribution
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)   # excess kurtosis (normal = 0)

    return {
        "Ann. Return (%)": round(ann_return * 100, 2),
        "Ann. Vol (%)":    round(ann_vol * 100, 2),
        "Sharpe Ratio":    round(sharpe, 3),
        "Sortino Ratio":   round(sortino, 3),
        "CVaR 95% (ann)":  round(cvar_95 * 100, 2),
        "Max Drawdown (%)":round(max_dd * 100, 2),
        "Calmar Ratio":    round(calmar, 3),
        "Skewness":        round(skew, 3),
        "Excess Kurtosis": round(kurt, 3),
        "Total Return (%)":round(total_return * 100, 2),
    }


def run_rl_agent(model_path: str, test_df: pd.DataFrame, lambda_val: float) -> dict:
    """Run a trained RL agent on test data and collect daily returns."""
    print(f"  Running agent: {model_path}")

    env = PortfolioEnv(test_df, lambda_cvar=lambda_val, **ENV_KWARGS)
    vec = DummyVecEnv([lambda: Monitor(env)])

    # Load the correct vecnorm for this model
    vecnorm_path = VECNORM_PATHS.get(model_path, "models/vec_normalize.pkl")
    vec = VecNormalize.load(vecnorm_path, vec)
    vec.training    = False
    vec.norm_reward = False

    model = PPO.load(model_path, env=vec)

    obs             = vec.reset()
    returns         = []
    weights_history = []
    done            = False

    while not done:
        action, _    = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec.step(action)
        done         = dones[0]
        info         = infos[0]
        returns.append(info.get("portfolio_return", 0.0))
        # weights are stored in the underlying env via info
        weights_history.append(
            vec.venv.envs[0].env.weights.copy()
        )

    return {
        "returns" : np.array(returns),
        "weights" : np.array(weights_history),
    }


# ── 1. Load test data ─────────────────────────────────────────────────────────
print("Loading test data...")
test_df  = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

test_dates = sorted(pd.to_datetime(test_df["Date"].unique()))
print(f"  Test period: {test_dates[0].date()} → {test_dates[-1].date()}  ({len(test_dates)} days)")


# ── 2. Compute baseline returns ───────────────────────────────────────────────
print("\nComputing baseline returns...")

prices = pd.read_csv("data/Indian_Portfolio_Prices_v2.csv", index_col="Date", parse_dates=True)
prices = prices.loc[test_dates[0]:test_dates[-1]]

# --- Baseline A: Equal weight (1/N) ---
ew_returns = prices.pct_change().dropna()
ew_port    = ew_returns.mean(axis=1).values   # equal weight = mean of all stocks

# --- Baseline B: Buy and hold Nifty50 ---
nifty_macro = pd.read_csv("data/Indian_Portfolio_Macro_v2.csv", parse_dates=["Date"])
nifty_macro = nifty_macro.set_index("Date").loc[test_dates[0]:test_dates[-1]]
nifty_ret   = nifty_macro["Nifty50_ret"].fillna(0).values

# --- Baseline C: CVaR optimal weights (static, from your friend's data) ---
cvar_weights_df = pd.read_csv("data/Optimal_CVaR_Weights_v2.csv", index_col="Unnamed: 0")
cvar_weights    = cvar_weights_df["Long-Only"].values
cvar_weights    = np.clip(cvar_weights, 0, None)          # ensure long-only
cvar_weights    = cvar_weights / cvar_weights.sum()        # renormalise
# Apply static CVaR weights to test returns
tickers_ordered = sorted(test_df["ticker"].unique())
stock_rets_df   = prices[tickers_ordered].pct_change().dropna()
cvar_port       = (stock_rets_df.values @ cvar_weights)   # matrix multiply

# Align lengths (prices.pct_change() drops first row)
n_test = min(len(ew_port), len(nifty_ret), len(cvar_port))
ew_port    = ew_port[:n_test]
nifty_ret  = nifty_ret[:n_test]
cvar_port  = cvar_port[:n_test]
test_dates_aligned = test_dates[1:n_test+1]

print(f"  Baseline returns computed ({n_test} days)")


# ── 3. Run RL agents ──────────────────────────────────────────────────────────
print("\nRunning RL agents on test set...")
rl_results = {}
for name, path in LAMBDA_VARIANTS.items():
    if os.path.exists(path + ".zip"):
        for name, path in LAMBDA_VARIANTS.items():
            if os.path.exists(path + ".zip"):
                lam = float(name.split("=")[1].rstrip(")"))
                rl_results[name] = run_rl_agent(path, test_df, lam)
    else:
        print(f"  WARNING: {path}.zip not found — skipping {name}")
        print(f"  (Train this agent first with lambda={name.split('=')[1][:-1]})")


# ── 4. Build results dictionary ───────────────────────────────────────────────
all_strategies = {
    "Equal Weight"    : ew_port,
    "Nifty50 B&H"     : nifty_ret,
    "CVaR Optimal"    : cvar_port,
}
for name, result in rl_results.items():
    r = result["returns"][:n_test]
    all_strategies[name] = r

# ── 5. Compute metrics for ALL strategies ─────────────────────────────────────
print("\nComputing metrics...")
metrics_rows = {}
for name, rets in all_strategies.items():
    metrics_rows[name] = compute_metrics(rets)
    print(f"  {name:25s} | Sharpe: {metrics_rows[name]['Sharpe Ratio']:>6.3f} | "
          f"CVaR: {metrics_rows[name]['CVaR 95% (ann)']:>6.2f}% | "
          f"Return: {metrics_rows[name]['Ann. Return (%)']:>6.2f}%")

metrics_df = pd.DataFrame(metrics_rows).T
metrics_df.to_csv("results/main_results_table.csv")
print("\nSaved: results/main_results_table.csv")

# -- 6. PLOT 0: Learning Curve
for lamda, path in EVAL_PATHS.items():
    data = np.load(path)
    timesteps = data["timesteps"]
    rewards   = data["results"].mean(axis=1)   

    plt.plot(timesteps, rewards)
    plt.xlabel("Training steps")
    plt.ylabel("Mean eval reward")
    plt.title("Learning curve")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/plots/learning_curve_model_{lamda}")

# ── 6. PLOT 1: Portfolio value curves ─────────────────────────────────────────
print("\nGenerating plots...")
fig, ax = plt.subplots(figsize=(12, 6))

for i, (name, rets) in enumerate(all_strategies.items()):
    cum = np.cumprod(1 + rets[:n_test])
    lw  = 2.5 if "RL" in name else 1.5
    ls  = "-" if "RL" in name else "--"
    ax.plot(test_dates_aligned[:len(cum)], cum,
            label=name, color=COLORS[i % len(COLORS)],
            linewidth=lw, linestyle=ls)

ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio value (normalised to 1.0)")
ax.set_title("Portfolio value — test period (2024–2026)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/01_portfolio_value_curves.png", dpi=150)
plt.close()
print("  Saved: 01_portfolio_value_curves.png")


# ── 7. PLOT 2: Rolling CVaR comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
window = 60

for i, (name, rets) in enumerate(all_strategies.items()):
    rolling_cvar = []
    r = np.array(rets[:n_test])
    for t in range(window, len(r)):
        w      = r[t-window:t]
        var_th = np.percentile(w, 5)
        tail   = w[w <= var_th]
        cv     = -np.mean(tail) if len(tail) > 0 else 0
        rolling_cvar.append(cv * 100)   # as percentage

    lw = 2.5 if "RL" in name else 1.5
    ax.plot(test_dates_aligned[window:window+len(rolling_cvar)],
            rolling_cvar, label=name,
            color=COLORS[i % len(COLORS)], linewidth=lw)

ax.set_xlabel("Date")
ax.set_ylabel("Rolling 60-day CVaR (%)")
ax.set_title("Rolling CVaR (95%) — lower is better tail-risk control")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("results/plots/02_rolling_cvar.png", dpi=150)
plt.close()
print("  Saved: 02_rolling_cvar.png")


# ── 8. PLOT 3: Lambda ablation — risk vs return tradeoff ──────────────────────
lambdas, sharpes, cvars, returns_list = [], [], [], []

for name, rets in all_strategies.items():
    if "RL (λ=" in name:
        lam = float(name.split("=")[1].rstrip(")"))
        m   = compute_metrics(rets)
        lambdas.append(lam)
        sharpes.append(m["Sharpe Ratio"])
        cvars.append(m["CVaR 95% (ann)"])
        returns_list.append(m["Ann. Return (%)"])

if lambdas:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(lambdas, sharpes, "o-", color="#2196F3", linewidth=2, markersize=8)
    for i, (x, y) in enumerate(zip(lambdas, sharpes)):
        axes[0].annotate(f"λ={x}", (x, y), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)
    axes[0].set_xlabel("Lambda (CVaR penalty weight)")
    axes[0].set_ylabel("Sharpe Ratio")
    axes[0].set_title("Lambda vs Sharpe Ratio")

    axes[1].plot(lambdas, cvars, "o-", color="#FF5722", linewidth=2, markersize=8)
    for i, (x, y) in enumerate(zip(lambdas, cvars)):
        axes[1].annotate(f"λ={x}", (x, y), textcoords="offset points",
                         xytext=(5, 5), fontsize=9)
    axes[1].set_xlabel("Lambda (CVaR penalty weight)")
    axes[1].set_ylabel("Annualised CVaR (%)")
    axes[1].set_title("Lambda vs CVaR (lower = safer)")

    plt.suptitle("Risk-return tradeoff: effect of CVaR penalty (λ)", fontsize=13)
    plt.tight_layout()
    plt.savefig("results/plots/03_lambda_ablation.png", dpi=150)
    plt.close()
    print("  Saved: 03_lambda_ablation.png")


# ── 9. PLOT 4: Weight allocation heatmap (best RL agent) ─────────────────────
best_rl_name = max(
    [k for k in rl_results.keys()],
    key=lambda k: compute_metrics(rl_results[k]["returns"])["Sharpe Ratio"]
)
if best_rl_name in rl_results:
    weights_over_time = rl_results[best_rl_name]["weights"]   # shape: (T, 52)
    if weights_over_time.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        # Sample every 20 days to keep plot readable
        step = max(1, len(weights_over_time) // 30)
        wt_sample = weights_over_time[::step]
        dates_sample = test_dates_aligned[:len(weights_over_time):step]

        im = ax.imshow(wt_sample.T, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=0.2)
        ax.set_yticks(range(len(tickers_ordered)))
        ax.set_yticklabels([t.replace(".NS", "") for t in tickers_ordered], fontsize=7)
        ax.set_xticks(range(0, len(dates_sample), 5))
        ax.set_xticklabels([str(d.date()) for d in dates_sample[::5]],
                           rotation=45, ha="right", fontsize=8)
        plt.colorbar(im, ax=ax, label="Portfolio weight")
        ax.set_title(f"Weight allocation over time — {best_rl_name}")
        plt.tight_layout()
        plt.savefig("results/plots/04_weight_heatmap.png", dpi=150)
        plt.close()
        print("  Saved: 04_weight_heatmap.png")


# ── 10. PLOT 5: Heavy-tail analysis ──────────────────────────────────────────
print("  Generating heavy-tail analysis...")
# Use full training returns to show fat tails
all_returns_train = []
for ticker in tickers_ordered[:10]:   # sample 10 stocks for clarity
    tr = train_df[train_df["ticker"] == ticker]["ret_1d"].dropna().values
    all_returns_train.extend(tr)

all_returns_train = np.array(all_returns_train)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# (a) Return distribution with normal overlay
axes[0].hist(all_returns_train, bins=100, density=True,
             alpha=0.7, color="#2196F3", label="Actual returns")
x  = np.linspace(all_returns_train.min(), all_returns_train.max(), 300)
mu, sigma = all_returns_train.mean(), all_returns_train.std()
axes[0].plot(x, stats.norm.pdf(x, mu, sigma),
             "r-", linewidth=2, label="Normal distribution")
axes[0].set_xlabel("Daily return")
axes[0].set_ylabel("Density")
axes[0].set_title("(a) Fat tails vs normal distribution")
axes[0].legend()
axes[0].set_xlim(-0.15, 0.15)

# (b) QQ plot vs normal
quantiles, normal_quantiles = stats.probplot(all_returns_train, dist="norm")[0]
axes[1].scatter(normal_quantiles, quantiles, alpha=0.3, s=3, color="#FF5722")
axes[1].plot([-4, 4], [-4, 4], "k--", linewidth=1.5, label="Normal line")
axes[1].set_xlabel("Theoretical normal quantiles")
axes[1].set_ylabel("Sample quantiles")
axes[1].set_title("(b) QQ plot — deviation = fat tails")
axes[1].legend()

# (c) Kurtosis across stocks (excess kurtosis > 0 = fat tails)
kurt_by_stock = []
stock_labels  = []
for ticker in tickers_ordered:
    tr = train_df[train_df["ticker"] == ticker]["ret_1d"].dropna().values
    if len(tr) > 50:
        kurt_by_stock.append(stats.kurtosis(tr))
        stock_labels.append(ticker.replace(".NS", ""))

kurt_arr = np.array(kurt_by_stock)
axes[2].barh(range(len(kurt_arr[:20])), kurt_arr[:20],
             color=["#FF5722" if k > 3 else "#2196F3" for k in kurt_arr[:20]])
axes[2].set_yticks(range(20))
axes[2].set_yticklabels(stock_labels[:20], fontsize=8)
axes[2].axvline(3, color="red", linestyle="--", linewidth=1.5, label="Heavy tail threshold (k>3)")
axes[2].set_xlabel("Excess kurtosis")
axes[2].set_title("(c) Excess kurtosis per stock\n(red = heavy-tailed)")
axes[2].legend(fontsize=8)

plt.suptitle("Heavy-tail behaviour in Indian stock returns", fontsize=13)
plt.tight_layout()
plt.savefig("results/plots/05_heavy_tail_analysis.png", dpi=150)
plt.close()
print("  Saved: 05_heavy_tail_analysis.png")


# ── 11. Print final results table ─────────────────────────────────────────────
print("\n" + "="*80)
print("MAIN RESULTS TABLE (Table 1 in paper)")
print("="*80)
print(metrics_df[["Ann. Return (%)", "Sharpe Ratio", "CVaR 95% (ann)",
                   "Max Drawdown (%)", "Sortino Ratio", "Calmar Ratio"]].to_string())
print("="*80)

# Save LaTeX table for the paper
latex = metrics_df[["Ann. Return (%)", "Sharpe Ratio", "CVaR 95% (ann)",
                     "Max Drawdown (%)", "Sortino Ratio"]].to_latex(
    caption="Portfolio performance comparison on test set (2024--2026)",
    label="tab:results",
    float_format="%.2f",
)
with open("results/table1_latex.tex", "w", encoding="utf-8") as f:
    f.write(latex)
print("\nSaved: results/table1_latex.tex  (paste directly into your paper!)")
print("\nAll done! Check the results/ folder for everything.")