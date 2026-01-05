import os
import time
import json
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


ENV_NAME = "LunarLander-v2"
SEED = 1
HIDDEN_SIZE = 128
LR_ACTOR = 3e-5     
LR_CRITIC = 1e-5    
GAMMA = 0.99
MAX_EPISODES = 4100 
TARGET_RUNNING_AVG = 250.0
LOG_EVERY_EPISODES = 100
LOGIT_CLIP_MAX = 20

BASE_PATH = "Q4"
CHECKPOINT_PATH = os.path.join(BASE_PATH, "checkpoints")
PLOT_PATH = os.path.join(BASE_PATH, "plots")
SAVE_PATH = os.path.join(CHECKPOINT_PATH, "a2c_model.pt")
SCORES_PATH = os.path.join(BASE_PATH, "a2c_scores.json")
LOSS_PLOT_PATH = os.path.join(PLOT_PATH, "a2c_loss_curves.png")
REWARD_PLOT_PATH = os.path.join(PLOT_PATH, "a2c_reward_curve.png")

# Ensure directories exist
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(PLOT_PATH, exist_ok=True)

Transition = namedtuple("Transition", 
                        ("state", "action", "log_prob", "reward", "next_state", "done"))


def set_seed(env, seed=SEED):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        # Gymnasium > 0.26.0
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    except TypeError:
        # Older gym versions
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        env.reset()


class ActorNet(nn.Module):
    
    def __init__(self, state_dim, action_dim, hidden_dim=HIDDEN_SIZE):
        super(ActorNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        logits = self.layer3(x)
        clipped_logits = torch.clamp(logits, min=-LOGIT_CLIP_MAX, max=LOGIT_CLIP_MAX)
        return F.softmax(clipped_logits, dim=-1)

    def select_action(self, state, device):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=HIDDEN_SIZE):
        super(CriticNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1) # Output a single value

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# key decision: how frequently we update actor and critic parameters.
def train_episode_gae(lambda_gae=0.95):
    env = gym.make(ENV_NAME)
    set_seed(env, SEED)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    actor = ActorNet(obs_dim, n_actions).to(device)
    critic = CriticNet(obs_dim).to(device)

    optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)

    episode_rewards = []
    running_avg_window = deque(maxlen=100)
    total_steps = 0
    start_time = time.time()
    print(f"--- Starting A2C+GAE Training for {MAX_EPISODES} episodes ---")

    actor_losses, critic_losses = [], []

    for ep in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        transitions = []

        # collect full episode
        while not done:
            action, log_prob = actor.select_action(state, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            transitions.append(Transition(state, action, log_prob, reward, next_state, done))
            ep_reward += reward
            total_steps += 1
            state = next_state

        # convert episode to tensors on device
        states = torch.tensor(np.array([t.state for t in transitions]), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array([t.next_state for t in transitions]), dtype=torch.float32, device=device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=device)
        log_probs = torch.stack([t.log_prob for t in transitions]).squeeze()

        # values
        v_s = critic(states).squeeze(-1)                 # shape (T,)
        with torch.no_grad():
            v_s_next = critic(next_states).squeeze(-1)   # shape (T,)

        # GAE advantage computation (backward)
        T = rewards.shape[0]
        advantages = torch.zeros(T, dtype=torch.float32, device=device)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]          # 0 if terminal at t, 1 otherwise
            delta = rewards[t] + GAMMA * v_s_next[t] * mask - v_s[t]
            last_gae = delta + GAMMA * lambda_gae * mask * last_gae
            advantages[t] = last_gae

        # targets = advantages + values (returns estimate)
        returns = (advantages + v_s).detach()
        adv_detached = advantages.detach()

        # normalize advantages
        if adv_detached.numel() > 1:
            adv_detached = (adv_detached - adv_detached.mean()) / (adv_detached.std() + 1e-4)

        # actor update
        actor_loss = -(log_probs * adv_detached).mean()
        optimizer_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.2)
        optimizer_actor.step()
        actor_losses.append(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else float(actor_loss))

        # critic update: regress V(s) to returns
        critic_loss = F.smooth_l1_loss(v_s, returns)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        optimizer_critic.step()
        critic_losses.append(critic_loss.item() if isinstance(critic_loss, torch.Tensor) else float(critic_loss))

        episode_rewards.append(ep_reward)
        running_avg_window.append(ep_reward)
        running_avg = np.mean(running_avg_window)

        if ep % LOG_EVERY_EPISODES == 0:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"Ep {ep}\tAvg: {running_avg:.2f}\tSteps: {total_steps}\tTime: {elapsed}")

            if running_avg >= TARGET_RUNNING_AVG and len(running_avg_window) >= 100:
                print(f"Solved at episode {ep} with avg reward {running_avg:.2f}. Saving model.")
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, SAVE_PATH)
                env.close()
                return  # training finished

    print("Training finished. Did not solve. Saving final model.")
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, SAVE_PATH)
    env.close()

    total_time = time.time() - start_time
    print(f"--- Training Complete ---")
    print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

    print("--- Generating Plots ---")
    
    # Save scores
    with open(SCORES_PATH, 'w') as f:
        json.dump(episode_rewards, f)

    # Plot Loss Curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(actor_losses, label="Actor Loss", alpha=0.8)
    plt.title("A2C Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(critic_losses, label="Critic Loss", color='orange', alpha=0.8)
    plt.title("A2C Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()
    print(f"Loss curves saved to {LOSS_PLOT_PATH}")

    # Plot Reward Curve
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label="Episode Reward", alpha=0.3)
    plt.plot(
        np.convolve(episode_rewards, np.ones(10)/10, mode='valid'),
        label="Moving Average (10)", color='red'
    )
    plt.title("A2C Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(REWARD_PLOT_PATH)
    plt.close()
    print(f"Reward curve saved to {REWARD_PLOT_PATH}")

if __name__ == "__main__":
    train_episode_gae()