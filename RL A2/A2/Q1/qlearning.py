import os
import json
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import math
import os
import random
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict
from frozenlake import DiagonalFrozenLake
from cliff import MultiGoalCliffWalkingEnv
from PIL import Image
import agent

# ---- environment ----
env = agent.MultiGoalCliffWalkingEnv(render_mode="rgb_array")

# ---- Q-learning sweep ----
agent.MAX_EPISODES = 5000
agent.GAMMA = 1.0

best_mean = -1e5
best_alpha = 0.1
best_safe = best_risky = 0
best_rewards = []
best_Q = None
best_episode_rewards = None

for i in range(10):
    agent.ALPHA = 0.1 + i * 0.05
    Q_val_q, rewards_q, safe_q, risky_q = agent.q_learning_for_cliff(env)
    episode_rewards_q = agent.evaluate(env, agent.find_policy_epsilon(Q_val_q, 0), 10)
    mean_reward = np.mean(episode_rewards_q)
    if mean_reward > best_mean:
        best_mean = mean_reward
        best_alpha = agent.ALPHA
        best_safe, best_risky = safe_q, risky_q
        best_rewards = rewards_q
        best_Q = Q_val_q
        best_episode_rewards = episode_rewards_q
        
# ---- directories ----
base_dir = os.path.dirname(__file__) or "."
plots_dir = os.path.join(base_dir, "plots")
gifs_dir = os.path.join(base_dir, "gifs")
eval_dir = os.path.join(base_dir, "evaluation")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(gifs_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# plot & gif
agent.plot(best_rewards)
plt.savefig(os.path.join(plots_dir, "qlearning_plot.png"))
plt.close()

agent.create_gif(
    agent.run_and_collect_frames(env, agent.find_policy_epsilon(best_Q, 0)),
    filename=os.path.join(gifs_dir, "qlearning.gif")
)

# save results
results = {
    "QLearning": {
        "mean": float(np.mean(best_episode_rewards)),
        "std": float(np.std(best_episode_rewards)),
        "safe": best_safe,
        "risky": best_risky,
        "best_alpha": best_alpha
    }
}

import json, os

json_path = os.path.join(eval_dir, "cliff_evaluation_results.json")

# Load existing data if file exists
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            existing_data = {}
else:
    existing_data = {}

# Merge new results
existing_data.update(results)

# Write merged data back
with open(json_path, "w") as f:
    json.dump(existing_data, f, indent=4)
