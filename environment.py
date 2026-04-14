"""
# PortfolioEnv – Risk-Aware Reinforcement Learning Environment

This file defines a custom OpenAI Gym (gymnasium) environment for training
reinforcement learning agents to manage a stock portfolio.

------------------------------------------------------------

## What this environment does

- Simulates daily portfolio allocation across 52 Indian stocks
- Agent decides how much weight to assign to each stock every day
- Uses real market + macroeconomic features as input

The goal:
    Maximize returns while controlling risk

------------------------------------------------------------

## Core Idea

At each timestep (1 trading day):
- Agent outputs portfolio weights (allocation across stocks)
- Environment computes:
    → portfolio return
    → transaction cost
    → risk penalty (CVaR)

Final reward:
    reward = return - λ * CVaR

So the agent is NOT just chasing returns,
it is penalized for risky behavior (heavy losses).

------------------------------------------------------------

## Action Space

- A vector of 52 real values (one per stock)
- Internally passed through softmax → valid portfolio weights

Constraints:
    - All weights ≥ 0
    - Sum of weights = 1 (long-only portfolio)

------------------------------------------------------------

## Reward Function

reward = portfolio_return - lambda_cvar * CVaR

Where:
- portfolio_return → weighted stock returns minus transaction cost
- CVaR (Conditional Value at Risk):
    Measures average loss in worst-case scenarios

This encourages:
    - stable returns
    - avoiding extreme losses

------------------------------------------------------------

## Risk Handling (CVaR)

CVaR focuses on:
    "How bad are the worst X% of outcomes?"

Steps:
    - Look at recent return history
    - Find worst α% of returns
    - Average those losses

Higher CVaR → more penalty

------------------------------------------------------------

## Episode Structure

- One episode = ~252 trading days (1 year)
- Random starting point in dataset
- Portfolio starts equal-weighted

------------------------------------------------------------

## Usage

No need to be explicitally utilized
agent.py uses this environment in itself

"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# ── Feature columns the agent will observe ────────────────────────────────────
# We deliberately exclude: open, high, low, close, volume (raw prices — noisy)
# and is_ffilled (data quality flag, not a market signal).
# We keep: returns, volatilities, momentum, technical indicators, macro.

STOCK_FEATURES = [
    # Returns
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "log_ret",
    # Volatility
    "vol_5d", "vol_21d", "vol_63d",
    # Momentum
    "momentum_21", "momentum_63", "momentum_252",
    # Trend
    "above_sma200", "price_to_52wh", "price_to_52wl",
    # Technical indicators
    "macd", "macd_signal", "macd_hist", "rsi_14",
    "bb_width", "bb_pct",
    # Risk
    "atr_14", "drawdown", "max_dd_63d",
    # Tail shape (key for heavy-tail paper!)
    "skew_21d", "kurt_21d",
    # Volume
    "vol_ratio", "obv",
]

MACRO_FEATURES = [
    "Crude_WTI_ret", "Gold_Futures_ret", "USDINR_ret",
    "India_VIX", "India_VIX_ret", "RBI_rate_pct",
    "NiftyAuto_ret", "NiftyIT_ret", "NiftyMetal_ret",
    "NiftyPharma_ret", "NiftyBank_ret", "Nifty50_ret",
]

N_STOCK_FEATURES = len(STOCK_FEATURES)   # 27
N_MACRO_FEATURES = len(MACRO_FEATURES)   # 12


class PortfolioEnv(gym.Env):
    """
    A daily-rebalancing, long-only portfolio environment for 52 Indian stocks.

    State  : [52 stocks × 26 features] + [12 macro features] + [52 current weights]
             = 1352 + 12 + 52 = 1416-dimensional vector

    Action : 52 portfolio weights in [0, 1] that sum to 1 (softmax applied)

    Reward : portfolio_return - lambda_cvar * CVaR(alpha, lookback_window)

    Episode: one calendar year of trading days (≈252 steps)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        episode_length: int = 252,      # one trading year per episode
        cvar_alpha: float = 0.05,       # CVaR at 95% confidence (worst 5% of days)
        cvar_lookback: int = 60,        # rolling window of past returns for CVaR
        lambda_cvar: float = 1.0,       # how much to penalise tail risk (tune this!)
        transaction_cost: float = 0.001 # 0.1% cost per trade (realistic for India)
    ):
        super().__init__()

        # ── Store hyperparameters ─────────────────────────────────────────────
        self.episode_length    = episode_length
        self.cvar_alpha        = cvar_alpha
        self.cvar_lookback     = cvar_lookback
        self.lambda_cvar       = lambda_cvar
        self.transaction_cost  = transaction_cost

        # ── Prepare data ──────────────────────────────────────────────────────
        self.df      = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.tickers = sorted(self.df["ticker"].unique())
        self.n       = len(self.tickers)                   # 52 stocks

        # Get all unique trading dates in chronological order
        self.dates = sorted(self.df["Date"].unique())

        # Fill NaNs — forward fill then backward fill
        # (handles warmup NaNs at the start of each stock's history)
        self.df = self.df.sort_values(["ticker", "Date"])
        self.df = (
            self.df
            .groupby("ticker", group_keys=False)
            .apply(lambda x: x.ffill().bfill())
            .reset_index(drop=True)
        )

        # Build a fast lookup: date → {ticker → feature_row}
        # This makes _get_state() O(1) instead of O(n) filtering
        print("Building date index for fast lookups...")
        self._build_date_index()

        # ── Define observation space ──────────────────────────────────────────
        # State = 52 stocks × 26 features + 12 macro + 52 current weights
        obs_dim = self.n * N_STOCK_FEATURES + N_MACRO_FEATURES + self.n
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape = (obs_dim,),
            dtype = np.float32
        )

        # ── Define action space ───────────────────────────────────────────────
        # Raw action: 52 real numbers (we softmax them to get valid weights)
        # Action space: symmetric [-1, 1] per SB3 recommendation
        # We still apply softmax inside step() so actual weights are always valid
        self.action_space = spaces.Box(
            low  = -1.0,
            high =  1.0,
            shape = (self.n,),
            dtype = np.float32
        )

        # ── Episode state (set properly in reset()) ───────────────────────────
        self.current_step    = 0
        self.start_idx       = 0
        self.weights         = np.ones(self.n) / self.n   # equal weight at start
        self.portfolio_value = 1.0                         # normalised to 1
        self.return_history  = []                          # for CVaR computation

        print(f"Environment ready!")
        print(f"  Stocks          : {self.n}")
        print(f"  Observation dim : {obs_dim}")
        print(f"  Action dim      : {self.n}")
        print(f"  Episode length  : {self.episode_length} days")
        print(f"  CVaR alpha      : {self.cvar_alpha} (worst {int(self.cvar_alpha*100)}% of days)")
        print(f"  Lambda CVaR     : {self.lambda_cvar}")

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: build a nested dict for O(1) state lookups
    # date_index[date][ticker] = row as numpy array of features
    # ─────────────────────────────────────────────────────────────────────────
    def _build_date_index(self):
        self.date_index = {}
        all_cols = STOCK_FEATURES + ["ret_1d"] + MACRO_FEATURES

        for date in self.dates:
            day_df = self.df[self.df["Date"] == date]
            self.date_index[date] = {}
            # Store macro separately (same for all tickers on this date)
            macro_row = day_df.iloc[0][MACRO_FEATURES].values.astype(np.float32)
            self.date_index[date]["__macro__"] = macro_row
            # Store each ticker's features
            for _, row in day_df.iterrows():
                ticker = row["ticker"]
                self.date_index[date][ticker] = {
                    "features": row[STOCK_FEATURES].values.astype(np.float32),
                    "ret_1d"  : float(row["ret_1d"]) if not pd.isna(row["ret_1d"]) else 0.0
                }

    # ─────────────────────────────────────────────────────────────────────────
    # reset() — called at the start of every new episode
    # Picks a random starting point in the data and resets all state
    # ─────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Pick a random start point, leaving room for a full episode
        # (so we don't start too close to the end of the data)
        max_start = len(self.dates) - self.episode_length - self.cvar_lookback - 1
        self.start_idx    = self.np_random.integers(0, max_start)
        self.current_step = 0

        # Reset portfolio state
        self.weights         = np.ones(self.n) / self.n   # start equal-weight
        self.portfolio_value = 1.0
        self.return_history  = []                          # empty CVaR window

        state = self._get_state()
        info  = {"portfolio_value": self.portfolio_value, "step": self.current_step}

        return state, info

    # ─────────────────────────────────────────────────────────────────────────
    # step() — the core of the environment, called every trading day
    # Takes an action (raw weights), applies softmax, computes return + reward
    # ─────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):

        # ── 3a. Convert raw action → valid portfolio weights ──────────────────
        # Softmax ensures: all weights ≥ 0 and sum to 1 (long-only constraint)
        weights_new = self._softmax(action)

        # ── 3b. Compute transaction cost ──────────────────────────────────────
        # Cost = sum of absolute weight changes × transaction cost rate
        # If agent barely changes weights, cost is near zero
        turnover = np.sum(np.abs(weights_new - self.weights))
        tc_cost  = turnover * self.transaction_cost

        # ── 3c. Get today's stock returns ─────────────────────────────────────
        date      = self.dates[self.start_idx + self.current_step]
        day_data  = self.date_index[date]
        stock_returns = np.array([
            day_data[t]["ret_1d"] if t in day_data else 0.0
            for t in self.tickers
        ], dtype=np.float32)

        # ── 3d. Compute portfolio return ──────────────────────────────────────
        # Weighted average of individual stock returns
        portfolio_return = float(np.dot(weights_new, stock_returns)) - tc_cost

        # Update portfolio value (compounds over the episode)
        self.portfolio_value *= (1 + portfolio_return)

        # Store return in history (for rolling CVaR computation)
        self.return_history.append(portfolio_return)

        # ── 3e. Compute CVaR reward ───────────────────────────────────────────
        cvar  = self._compute_cvar()
        reward = portfolio_return - self.lambda_cvar * cvar

        # ── 3f. Advance state ─────────────────────────────────────────────────
        self.weights       = weights_new
        self.current_step += 1
        done      = self.current_step >= self.episode_length
        truncated = False

        # ── 3g. Build next state ──────────────────────────────────────────────
        next_state = self._get_state()

        info = {
            "portfolio_return" : portfolio_return,
            "portfolio_value"  : self.portfolio_value,
            "cvar"             : cvar,
            "turnover"         : turnover,
            "tc_cost"          : tc_cost,
            "date"             : str(date),
        }

        return next_state, reward, done, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # _get_state() — builds the observation vector the agent receives
    # Shape: (52 × 26 stock features) + (12 macro) + (52 current weights)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        date     = self.dates[self.start_idx + self.current_step]
        day_data = self.date_index[date]

        # Stack all 52 stocks' feature vectors
        stock_feats = np.concatenate([
            day_data[t]["features"] if t in day_data else np.zeros(N_STOCK_FEATURES)
            for t in self.tickers
        ])  # shape: (52 × 26,) = (1352,)

        # Macro features (same for all stocks on this day)
        macro_feats = day_data.get("__macro__", np.zeros(N_MACRO_FEATURES))
        # shape: (12,)

        # Current portfolio weights (so agent knows its own position)
        current_weights = self.weights.astype(np.float32)
        # shape: (52,)

        # Concatenate everything into one flat state vector
        state = np.concatenate([stock_feats, macro_feats, current_weights])
        # shape: (1416,)

        # Replace any remaining NaNs/Infs with 0 (safety net)
        state = np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

        return state.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # _compute_cvar() — Conditional Value at Risk
    #
    # CVaR (also called Expected Shortfall) answers:
    # "On the worst alpha% of days, what was our average loss?"
    #
    # Steps:
    #   1. Take the last `cvar_lookback` portfolio returns
    #   2. Find the alpha-th percentile (the VaR threshold)
    #   3. Average all returns BELOW that threshold
    #   4. Negate it (so CVaR is a positive number = a loss magnitude)
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_cvar(self) -> float:
        # Need at least a few returns to compute a meaningful CVaR
        if len(self.return_history) < 10:
            return 0.0

        # Use only the most recent `cvar_lookback` returns
        window = np.array(self.return_history[-self.cvar_lookback:])

        # VaR threshold: the alpha-th percentile of returns
        # e.g. alpha=0.05 → the 5th percentile → the worst 5% of days
        var_threshold = np.percentile(window, self.cvar_alpha * 100)

        # CVaR: mean of returns that are WORSE than the VaR threshold
        tail_returns = window[window <= var_threshold]

        if len(tail_returns) == 0:
            return 0.0

        # Negate so CVaR is positive (a loss magnitude, not a negative return)
        cvar = -np.mean(tail_returns)

        return float(max(cvar, 0.0))   # clamp to 0 (can't have negative loss)

    # ─────────────────────────────────────────────────────────────────────────
    # _softmax() — converts raw action vector to valid portfolio weights
    # Guarantees: all weights in (0, 1) and sum to exactly 1
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (prevents overflow)
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()