

import os
import argparse
import random
import pickle
from collections import deque
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from tqdm import trange, tqdm
import matplotlib.pyplot as plt


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mkdir_if_missing(path):
    os.makedirs(path, exist_ok=True)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64,64), log_std_init=-0.5):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, act_dim)
        # single learnable log_std per action dimension
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(obs)
        mean = self.mean_head(x)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs_np: np.ndarray, device):
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.forward(obs)
            dist = torch.distributions.Normal(mean, std)
            a = dist.sample()
            logp = dist.log_prob(a).sum(axis=-1)
            a = a.cpu().numpy()[0]
        return a, float(logp.cpu().numpy())

    def log_prob_from_batch(self, obs: torch.Tensor, acts: torch.Tensor):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(acts).sum(axis=-1)
        return logp

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(64,64)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # (batch,)



def discount_cumsum(rewards: np.ndarray, gamma: float) -> np.ndarray:
    n = len(rewards)
    returns = np.zeros(n, dtype=np.float32)
    running = 0.0
    for i in reversed(range(n)):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns

def episode_return(rewards: List[float]) -> float:
    return float(np.sum(rewards))


def train_reinforce(env_name: str,
                    baseline_type: str,
                    device,
                    gamma: float = 0.99,
                    lr_policy: float = 1e-3,
                    lr_value: float = 1e-3,
                    batch_episodes: int = 5,
                    max_episodes: int = 5000,
                    target_mean_low: float = 400.0,
                    target_mean_high: float = 500.0,
                    seed: int = 0):
    assert baseline_type in ('no', 'avg', 'reward_to_go', 'value')
    set_seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim).to(device)
    opt_policy = optim.Adam(policy.parameters(), lr=lr_policy)

    value_net = None
    opt_value = None
    if baseline_type == 'value':
        value_net = ValueNetwork(obs_dim).to(device)
        opt_value = optim.Adam(value_net.parameters(), lr=lr_value)

    reward_history = deque(maxlen=100)
    episode = 0

    # storage for printing
    pbar = trange(max_episodes, desc=f"Train {baseline_type}", leave=True)
    for ep in pbar:
        batch_obs = []
        batch_acts = []
        batch_logps = []
        batch_returns = []
        batch_lengths = []
        batch_episode_rewards = []


        for _ in range(batch_episodes):
            obs, _ = env.reset()
            obs_list = []
            acts_list = []
            logps_list = []
            rewards_list = []

            done = False
            while True:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                mean, std = policy(obs_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample().cpu().numpy()[0]
                # clip to action space
                action_env = np.clip(action, env.action_space.low, env.action_space.high)
                logp = dist.log_prob(torch.as_tensor(action, dtype=torch.float32, device=device)).sum().item()

                next_obs, reward, terminated, truncated, info = env.step(action_env)
                done = terminated or truncated

                obs_list.append(obs.copy())
                acts_list.append(action)
                logps_list.append(logp)
                rewards_list.append(float(reward))

                obs = next_obs
                if done:
                    break

            returns = discount_cumsum(np.array(rewards_list, dtype=np.float32), gamma)
            batch_obs.extend(obs_list)
            batch_acts.extend(acts_list)
            batch_logps.extend(logps_list)
            batch_returns.extend(list(returns))
            batch_lengths.append(len(rewards_list))
            batch_episode_rewards.append(sum(rewards_list))
            reward_history.append(sum(rewards_list))
            episode += 1

        # prepare tensors
        obs_b = torch.as_tensor(np.array(batch_obs, dtype=np.float32), device=device)
        acts_b = torch.as_tensor(np.array(batch_acts, dtype=np.float32), device=device)
        returns_b = torch.as_tensor(np.array(batch_returns, dtype=np.float32), device=device)

        # compute advantages depending on baseline_type
        if baseline_type == 'no':
            advantages = returns_b.clone()
        elif baseline_type == 'avg':
            # average reward baseline: scalar baseline = mean of episode returns in this batch
            baseline_scalar = float(np.mean(batch_episode_rewards))
            advantages = returns_b - baseline_scalar
        elif baseline_type == 'reward_to_go':
            advantages = returns_b.clone()
        elif baseline_type == 'value':
            # train value_net to predict returns (MSE) first, then compute advantage = returns - value
            # convert to value predictions
            value_preds = value_net(obs_b).detach()
            advantages = returns_b - value_preds
            # train value network on (obs_b, returns_b)
            for _ in range(1):  # one gradient step per batch
                v_pred = value_net(obs_b)
                loss_v = nn.MSELoss()(v_pred, returns_b)
                opt_value.zero_grad()
                loss_v.backward()
                opt_value.step()
        else:
            raise ValueError("Unknown baseline type")

        # normalize advantages for stability (common practice)
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # policy update (REINFORCE)
        logps = policy.log_prob_from_batch(obs_b, acts_b)
        loss_policy = -(logps * advantages).mean()

        opt_policy.zero_grad()
        loss_policy.backward()
        opt_policy.step()

        # print stats and check stopping
        if len(reward_history) == 100:
            last100 = np.mean(reward_history)
        else:
            last100 = np.mean(list(reward_history))

        pbar.set_postfix({'ep': episode, 'last100': f"{last100:.1f}"})
        # stop when average reward over last 100 episodes is within desired range
        if len(reward_history) == 100 and (target_mean_low <= last100 <= target_mean_high):
            print(f"\nReached target rolling mean {last100:.2f} (episodes: {episode}) for baseline {baseline_type}. Stopping training.")
            break

        if episode >= max_episodes:
            print(f"\nMax episodes reached ({max_episodes}) for baseline {baseline_type}. Last100 mean {last100:.2f}.")
            break

    env.close()
    # return models (policy and optional value_net) and last rolling mean
    return policy, value_net, float(last100)



def collect_trajectories(policy: GaussianPolicy,
                         env_name: str,
                         num_trajectories: int,
                         device,
                         seed:int=0):
    set_seed(seed)
    env = gym.make(env_name)
    trajectories = []
    pbar = tqdm(total=num_trajectories, desc="Collect traj")
    while len(trajectories) < num_trajectories:
        obs, _ = env.reset()
        obs_list, acts_list, rewards_list, logps_list = [], [], [], []
        done = False
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std = policy(obs_tensor)
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample().cpu().numpy()[0]
                logp = dist.log_prob(torch.as_tensor(a, dtype=torch.float32, device=device)).sum(axis=-1).cpu().item()
            action_env = np.clip(a, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            obs_list.append(obs.copy())
            acts_list.append(a)
            logps_list.append(logp)
            rewards_list.append(float(reward))
            obs = next_obs
            if terminated or truncated:
                break
        traj = {
            'obs': np.array(obs_list, dtype=np.float32),
            'acts': np.array(acts_list, dtype=np.float32),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'logps': np.array(logps_list, dtype=np.float32),
        }
        trajectories.append(traj)
        pbar.update(1)
    pbar.close()
    env.close()
    return trajectories



def flatten_grad_list(grad_list: List[torch.Tensor]) -> torch.Tensor:
    parts = []
    for g in grad_list:
        if g is None:
            parts.append(torch.zeros(0, device=grad_list[0].device))
        else:
            parts.append(g.reshape(-1))
    if len(parts) == 0:
        return torch.zeros(0)
    return torch.cat(parts)

def compute_gradient_estimate(policy: GaussianPolicy,
                              trajectories: List[Dict[str, Any]],
                              baseline_type: str,
                              gamma: float,
                              device,
                              value_net: ValueNetwork = None) -> torch.Tensor:
    policy.to(device)
    # Build a single batch from provided trajectories
    batch_obs_list, batch_acts_list, batch_returns_list = [], [], []
    episode_returns = []
    for traj in trajectories:
        obs = traj['obs']
        acts = traj['acts']
        rewards = traj['rewards']
        returns = discount_cumsum(rewards, gamma)
        batch_obs_list.append(obs)
        batch_acts_list.append(acts)
        batch_returns_list.append(returns)
        episode_returns.append(np.sum(rewards))
    obs_b = torch.as_tensor(np.concatenate(batch_obs_list, axis=0), dtype=torch.float32, device=device)
    acts_b = torch.as_tensor(np.concatenate(batch_acts_list, axis=0), dtype=torch.float32, device=device)
    returns_b = torch.as_tensor(np.concatenate(batch_returns_list, axis=0), dtype=torch.float32, device=device)


    if baseline_type == 'no':
        advantages = returns_b.clone()
    elif baseline_type == 'avg':
        baseline_scalar = float(np.mean(episode_returns))
        advantages = returns_b - baseline_scalar
    elif baseline_type == 'reward_to_go':
        advantages = returns_b.clone()
    elif baseline_type == 'value':
        assert value_net is not None, "Value net must be provided for 'value' baseline"
        value_preds = value_net(obs_b).detach()
        advantages = returns_b - value_preds
    else:
        raise ValueError("Unknown baseline_type")

    adv_mean = advantages.mean().item()
    adv_std = advantages.std().item() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    # compute surrogate objective = mean(logp * advantage); gradient of this (negative sign for ascent)
    logps = policy.log_prob_from_batch(obs_b, acts_b)
    surrogate = (logps * advantages).mean()  # we will take gradient of this; policy gradient is gradient of this
    # We want gradient estimate vector of the policy gradient (ascent). We'll return gradient of surrogate (not negative).
    grads = torch.autograd.grad(surrogate, list(policy.parameters()), retain_graph=False, allow_unused=True)
    grad_vec = flatten_grad_list([g if g is not None else torch.zeros_like(p).to(device) for g, p in zip(grads, policy.parameters())])
    # grad_vec is gradient of surrogate (i.e., sample-based estimate)
    return grad_vec.detach().cpu()



import sys
from datetime import datetime


try:
    import torch
except ImportError:
    class DummyTorch:
        @staticmethod
        def cuda(): return type('Cuda', (), {'is_available': lambda: False})
        @staticmethod
        def device(dev): return dev
        @staticmethod
        def is_available(): return False
    torch = DummyTorch()

def set_seed(seed):
    print(f"--- Setting seed to {seed} ---")

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")



def main():
    parser = argparse.ArgumentParser(description="RL Training Script")
    parser.add_argument('--env', type=str, default='InvertedPendulum-v4')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default=None, help="cuda or cpu (auto by default)")
    parser.add_argument('--max_episodes', type=int, default=5000)
    parser.add_argument('--collect_trajectories', type=int, default=500)


    args, unknown_args = parser.parse_known_args()

    if unknown_args:
        print("\n[INFO] Ignoring unknown arguments injected by Notebook/Colab kernel.\n")

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("Device:", device)
    set_seed(args.seed)

    baselines = {
        'no': 'no_baseline',
        'avg': 'average_reward_baseline',
        'reward_to_go': 'reward_to_go_baseline',
        'value': 'value_function_baseline'
    }

    # Directory Setup
    mkdir_if_missing('models')
    mkdir_if_missing('trajectories')
    mkdir_if_missing('plots')

    # Placeholder for other variables
    trained_policies = {}
    trained_values = {}

    print("\n--- Configurations Loaded ---\n")
    print(f"Environment: {args.env}")
    print(f"Max Episodes: {args.max_episodes}")

    # 1) Train each baseline
    for key, name in baselines.items():
        print(f"\n=== Training baseline: {name} ===")
        policy, value_net, last100 = train_reinforce(
            env_name=args.env,
            baseline_type=key if key != 'avg' else 'avg',  # mapping key -> baseline_type
            device=device,
            gamma=args.gamma,
            lr_policy=1e-3,
            lr_value=1e-3,
            batch_episodes=5,
            max_episodes=args.max_episodes,
            seed=args.seed
        )
        # save policy
        save_path = os.path.join('models', f'{name}_policy.pth')
        torch.save(policy.state_dict(), save_path)
        print(f"Saved policy: {save_path}")
        if value_net is not None:
            vpath = os.path.join('models', f'{name}_value.pth')
            torch.save(value_net.state_dict(), vpath)
            print(f"Saved value net: {vpath}")

        trained_policies[key] = policy
        trained_values[key] = value_net


    for key, name in baselines.items():
        print(f"\n=== Collecting trajectories for {name} ===")
        policy = trained_policies[key]
        # ensure policy in eval mode but keep parameters requiring grad for later grad computations
        policy.eval()
        # collect
        trajs = collect_trajectories(policy, args.env, num_trajectories=args.collect_trajectories, device=device, seed=args.seed)
        traj_path = os.path.join('trajectories', f'{name}_trajectories.pkl')
        with open(traj_path, 'wb') as f:
            pickle.dump(trajs, f)
        print(f"Saved {len(trajs)} trajectories to {traj_path}")

    sample_sizes = [20] + list(range(30, 101, 10))  # 20,30,...,100
    repeats = 10

    results = {key: {'sample_sizes': sample_sizes, 'means': [], 'stds': [], 'raw': {}} for key in baselines.keys()}

    for key, name in baselines.items():
        print(f"\n=== Gradient estimation for {name} ===")
        traj_path = os.path.join('trajectories', f'{name}_trajectories.pkl')
        with open(traj_path, 'rb') as f:
            all_trajs = pickle.load(f)

        # if value baseline, load associated value net
        value_net = trained_values[key]
        if value_net is not None:
            value_net.to(device)
            value_net.eval()

        raw_by_size = {}
        for s in sample_sizes:
            grad_norms = []
            raw_by_size[s] = []
            for r in range(repeats):
                sampled = random.sample(all_trajs, s)
                grad_vec = compute_gradient_estimate(trained_policies[key], sampled, baseline_type=key if key!='avg' else 'avg', gamma=args.gamma, device=device, value_net=value_net)
                norm = float(torch.norm(grad_vec, p=2).item())
                grad_norms.append(norm)
                raw_by_size[s].append(grad_vec.numpy())
            mean = float(np.mean(grad_norms))
            std = float(np.std(grad_norms))
            results[key]['means'].append(mean)
            results[key]['stds'].append(std)
            print(f"Baseline {name} | sample {s} | mean norm {mean:.6f} | std {std:.6f}")

        results[key]['raw'] = raw_by_size


    print("\nPlotting results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_list = axes.flatten()
    for idx, (key, name) in enumerate(baselines.items()):
        ax = ax_list[idx]
        sample_sizes = results[key]['sample_sizes']
        means = np.array(results[key]['means'])
        stds = np.array(results[key]['stds'])
        ax.plot(sample_sizes, means, label='mean gradient norm')
        ax.fill_between(sample_sizes, means - stds, means + stds, alpha=0.3, label='±1 std')
        ax.set_title(name.replace('_', ' '))
        ax.set_xlabel('Sample size (num trajectories)')
        ax.set_ylabel('Gradient estimate magnitude (L2 norm)')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    plot_path = os.path.join('plots', 'gradient_estimate_variance.png')
    plt.savefig(plot_path, dpi=200)
    print(f"Saved plot: {plot_path}")

    with open('plots/gradient_estimate_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("Saved results pickles.")

if __name__ == '__main__':
    main()
