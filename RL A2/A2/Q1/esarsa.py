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

# ---- Expected SARSA ----
agent.ALPHA = 0.45
agent.GAMMA = 0.99
# ---- environment ----
env = agent.MultiGoalCliffWalkingEnv(render_mode="rgb_array")
Q_val, rewards, safe_esarsa, risky_esarsa = agent.expected_SARSA(env)
episode_rewards = agent.evaluate(env, agent.find_policy_epsilon(Q_val, 0), 10)


# ---- directories ----
base_dir = os.path.dirname(__file__) or "."
plots_dir = os.path.join(base_dir, "plots")
gifs_dir = os.path.join(base_dir, "gifs")
eval_dir = os.path.join(base_dir, "evaluation")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(gifs_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)


# plot & gif
agent.plot(rewards)
plt.savefig(os.path.join(plots_dir, "expected_sarsa_plot.png"))
plt.close()

agent.create_gif(
    agent.run_and_collect_frames(env, agent.find_policy_epsilon(Q_val, 0)),
    filename=os.path.join(gifs_dir, "expected_sarsa.gif")
)

# save results
results = {
    "ExpectedSarsa": {
        "mean": float(np.mean(episode_rewards)),
        "std": float(np.std(episode_rewards)),
        "safe": safe_esarsa,
        "risky": risky_esarsa
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


