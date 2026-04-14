"""
# agent.py — PPO Training Pipeline for Portfolio Optimization

This script trains a reinforcement learning (RL) agent using PPO
(Proximal Policy Optimization) on the custom PortfolioEnv.

------------------------------------------------------------

## 🧠 What this script does

- Loads training and validation market data
- Creates RL environments (train + validation)
- Applies normalization for stable learning
- Trains a PPO agent to allocate portfolio weights
- Evaluates performance periodically
- Saves the best and final models

Goal:
    Learn a policy that maximizes returns while controlling risk (CVaR)

------------------------------------------------------------

## Core Components

### 1. Environment
Uses PortfolioEnv:
    - 52-stock portfolio allocation
    - Risk-aware reward (return - λ * CVaR)

Wrapped with:
    - Monitor → logs episode stats
    - DummyVecEnv → required by Stable Baselines
    - VecNormalize → normalizes observations & rewards

------------------------------------------------------------

## Agent (PPO)

Uses Stable Baselines3 PPO implementation.

Policy:
    - MLP (fully connected neural network)
    - Architecture: [256, 256, 128]

Key hyperparameters:
    - learning_rate = 3e-4
    - batch_size    = 256
    - n_steps       = 2048
    - gamma         = 0.99
    - clip_range    = 0.2

These control:
    - learning speed
    - stability
    - exploration vs exploitation

------------------------------------------------------------

## Training Setup

- Total timesteps: 500,000 (~2000 episodes)
- Each episode: ~252 trading days (1 year)
- Training uses normalized rewards
- Validation uses real (unnormalized) rewards

------------------------------------------------------------

## Risk Awareness

Controlled by:
    lambda_cvar = [0.0, ...]

Higher value:
    → safer, more conservative strategy

Lower value:
    → more aggressive, higher risk-taking

------------------------------------------------------------

## 💾 Outputs

Saved in:
    models/lambda{lambda_cvar}/

Files:
    - ppo_portfolio_final.zip     → final trained model
    - ppo_portfolio_best/         → best checkpoint during training
    - vec_normalize.pkl           → normalization stats (VERY important)
    - logs/                       → TensorBoard logs
    - training_report.txt         → summary of training

------------------------------------------------------------

## 🚀 How to Run

python train_agent.py

"""

import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment import PortfolioEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────
# These are the main knobs you can tune for the paper's ablation study

ENV_KWARGS = dict(
    episode_length   = 252,    # one trading year per episode
    cvar_alpha       = 0.05,   # CVaR at 95% confidence
    cvar_lookback    = 60,     # rolling 60-day window for CVaR
    lambda_cvar      = 3.0,    # risk penalty strength — KEY hyperparameter!
    transaction_cost = 0.001,  # 0.1% per trade
)

os.makedirs(f"lambda{ENV_KWARGS['lambda_cvar']}/models", exist_ok=True)
os.makedirs(f"lambda{ENV_KWARGS['lambda_cvar']}/logs",   exist_ok=True)


PPO_KWARGS = dict(
    policy          = "MlpPolicy",   # multi-layer perceptron policy
    learning_rate   = 3e-4,          # Adam learning rate
    n_steps         = 2048,          # steps collected per update
    batch_size      = 256,           # minibatch size for gradient updates
    n_epochs        = 10,            # passes over each batch
    gamma           = 0.99,          # discount factor (how much agent values future)
    gae_lambda      = 0.95,          # GAE smoothing (bias-variance tradeoff)
    clip_range      = 0.2,           # PPO clipping (keeps updates stable)
    ent_coef        = 0.01,          # entropy bonus (encourages exploration)
    vf_coef         = 0.5,           # value function loss weight
    max_grad_norm   = 0.5,           # gradient clipping (prevents exploding grads)
    policy_kwargs   = dict(
        net_arch = [256, 256, 128],  # 3-layer network: 256 → 256 → 128 neurons
    ),
    verbose         = 1,
    tensorboard_log = f"lambda{ENV_KWARGS['lambda_cvar']}/logs/",
)

TOTAL_TIMESTEPS = 500_000   # total training steps
                             # ~2000 episodes of 252 days each
                             # increase to 1_000_000 for better results

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("val.csv")
print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")

# ── 2. Create environments ────────────────────────────────────────────────────
print("\nCreating environments...")

# Training environment
# Monitor wraps the env to log episode rewards and lengths automatically
train_env = Monitor(PortfolioEnv(train_df, **ENV_KWARGS))

# Wrap in DummyVecEnv (SB3 requires vectorised envs)
# You can change to SubprocVecEnv for parallel training if you have multiple CPUs
train_vec = DummyVecEnv([lambda: train_env])

# Normalise observations and rewards — very important for stable PPO training!
# obs: normalises state vector to zero mean, unit variance
# reward: scales rewards to a reasonable range
train_vec = VecNormalize(
    train_vec,
    norm_obs    = True,
    norm_reward = True,
    clip_obs    = 10.0,    # clip extreme observations
)

# Validation environment — shorter episode to fit in 260 val days
# (260 days - 60 cvar_lookback - 1 buffer = ~199 max episode length)
val_env_kwargs = {**ENV_KWARGS, "episode_length": 180}
val_env = Monitor(PortfolioEnv(val_df, **val_env_kwargs))
val_vec = DummyVecEnv([lambda: val_env])
val_vec = VecNormalize(
    val_vec,
    norm_obs    = True,
    norm_reward = False,   # don't normalise val rewards — we want true values
    training    = False,   # don't update normalisation stats on val data
)

# ── 3. Sanity check the environment ───────────────────────────────────────────
print("\nRunning environment sanity check...")
check_env(PortfolioEnv(train_df, **ENV_KWARGS), warn=True)
print("  Environment check passed!")

# ── 4. Define callbacks ───────────────────────────────────────────────────────

class TrainingLogCallback(BaseCallback):
    """
    Logs useful metrics to the console every N episodes.
    Helps you monitor if the agent is actually improving during training.
    """
    def __init__(self, log_freq=5000, verbose=0):
        super().__init__(verbose)
        self.log_freq    = log_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect episode rewards from Monitor wrapper
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)

        # Print summary every log_freq steps
        if self.num_timesteps % self.log_freq == 0 and self.episode_rewards:
            recent = self.episode_rewards[-20:]   # last 20 episodes
            print(
                f"  Step {self.num_timesteps:>7,} | "
                f"Episodes: {len(self.episode_rewards):>4} | "
                f"Mean reward (last 20): {np.mean(recent):>8.4f} | "
                f"Std: {np.std(recent):>6.4f}"
            )
        return True   # return False to stop training early


# EvalCallback — evaluates on val set every eval_freq steps
# Saves the best model automatically
eval_callback = EvalCallback(
    val_vec,
    best_model_save_path = f"models/lambda{ENV_KWARGS['lambda_cvar']}/models/ppo_portfolio_best/",
    log_path             = f"models/lambda{ENV_KWARGS['lambda_cvar']}/logs/eval/",
    eval_freq            = 10_000,      # evaluate every 10k steps
    n_eval_episodes      = 10,          # run 10 val episodes for stable estimate
    deterministic        = True,        # use greedy policy for evaluation
    render               = False,
)

log_callback = TrainingLogCallback(log_freq=5000)

callbacks = CallbackList([eval_callback, log_callback])

# ── 5. Create and train the PPO agent ─────────────────────────────────────────
print("\nCreating PPO agent...")
model = PPO(env=train_vec, **PPO_KWARGS)

print(f"\nPolicy network architecture:")
print(model.policy)

print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
print("(Watch the mean reward — it should trend upward over time)\n")

model.learn(
    total_timesteps = TOTAL_TIMESTEPS,
    callback        = callbacks,
    progress_bar    = True,
)

# ── 6. Save the trained model ─────────────────────────────────────────────────
print("\nSaving model...")
model.save(f"models/lambda{ENV_KWARGS['lambda_cvar']}/models/ppo_portfolio_final")
train_vec.save(f"models/lambda{ENV_KWARGS['lambda_cvar']}/models/vec_normalize.pkl")   # save normalisation stats too!
print(f"SAVED : models/lambda{ENV_KWARGS['lambda_cvar']}/models/ppo_portfolio_final.zip")
print(f"Saved: models/lambda{ENV_KWARGS['lambda_cvar']}/models/vec_normalize.pkl")

# ── 7. Quick validation run ───────────────────────────────────────────────────
print("\nRunning quick validation check...")
obs = val_vec.reset()
episode_rewards = []
episode_reward  = 0

for _ in range(ENV_KWARGS["episode_length"]):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = val_vec.step(action)
    episode_reward += reward[0]
    if done[0]:
        episode_rewards.append(episode_reward)
        episode_reward = 0
        obs = val_vec.reset()

print(f"  Val episodes completed  : {len(episode_rewards)}")
if episode_rewards:
    print(f"  Mean episode reward     : {np.mean(episode_rewards):.4f}")
    print(f"  Best episode reward     : {np.max(episode_rewards):.4f}")

# ── 8. Save training report ───────────────────────────────────────────────────
report = [
    "=== Training Report ===\n",
    f"Total timesteps   : {TOTAL_TIMESTEPS:,}",
    f"Episode length    : {ENV_KWARGS['episode_length']} days",
    f"CVaR alpha        : {ENV_KWARGS['cvar_alpha']}",
    f"CVaR lookback     : {ENV_KWARGS['cvar_lookback']} days",
    f"Lambda CVaR       : {ENV_KWARGS['lambda_cvar']}",
    f"Transaction cost  : {ENV_KWARGS['transaction_cost']}",
    f"Network arch      : {PPO_KWARGS['policy_kwargs']['net_arch']}",
    f"Learning rate     : {PPO_KWARGS['learning_rate']}",
    f"Batch size        : {PPO_KWARGS['batch_size']}",
    "",
    "=== Val Results ===",
    f"Mean episode reward : {np.mean(episode_rewards):.4f}" if episode_rewards else "N/A",
    f"Best episode reward : {np.max(episode_rewards):.4f}" if episode_rewards else "N/A",
]
with open(f"models/lambda{ENV_KWARGS['lambda_cvar']}/training_report.txt", "w") as f:
    f.write("\n".join(report))

print("\nTraining complete!")
print("Next step: run evaluate.py to see full results on the test set.")