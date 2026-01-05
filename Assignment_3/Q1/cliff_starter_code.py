
import torch.nn.functional as F
import copy
import os
import json
import math
import random
import time
from collections import deque, namedtuple
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from cliff import MultiGoalCliffWalkingEnv

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def state_to_tensor(state, env):
    """Converts the environment's discrete state into a feature tensor."""
    grid_size = env.height * env.width
    checkpoint_status = state // grid_size
    position_index = state % grid_size
    
    y = position_index // env.width
    x = position_index % env.width

    checkpoints_binary = [(checkpoint_status >> i) & 1 for i in range(2)]
    
    features = [y / env.height, x / env.width] + checkpoints_binary
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

NUM_SEEDS = 10
EPISODES_PER_SEED = 500
BATCH_SIZE = 64
BUFFER_CAPACITY = 20000
GAMMA = 0.99
LR = 1e-3
TARGET_UPDATE_FREQ = 1000   # steps
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995           # multiplicative per episode
MIN_REPLAY_SIZE = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output folders
MODEL_DIR = "models"
PLOT_DIR = "plots"
EVAL_DIR = "evaluation"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)


def decode_state_index(idx, env):
    width = env.width
    base = idx // 4
    checkpoint_bits = idx % 4
    r = base // width
    c = base % width
    checkA = (checkpoint_bits >> 1) & 1
    checkB = checkpoint_bits & 1
    row_norm = r / float(env.height - 1)
    col_norm = c / float(env.width - 1)
    return np.array([row_norm, col_norm, float(checkA), float(checkB)], dtype=np.float32)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", ["s", "a", "r", "ns", "d"]) 

    def push(self, s, a, r, ns, d):
        self.buffer.append(self.experience(s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.as_tensor(np.stack([b.s for b in batch]), dtype=torch.float32).to(DEVICE)
        actions = torch.as_tensor(np.array([b.a for b in batch]), dtype=torch.int64).unsqueeze(1).to(DEVICE)
        rewards = torch.as_tensor(np.array([b.r for b in batch]), dtype=torch.float32).unsqueeze(1).to(DEVICE)
        next_states = torch.as_tensor(np.stack([b.ns for b in batch]), dtype=torch.float32).to(DEVICE)
        dones = torch.as_tensor(np.array([b.d for b in batch]).astype(np.float32)).unsqueeze(1).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
class LinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearDQN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        return self.fc(x)


class NonLinearDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NonLinearDQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, output_dim)
        )
        # init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.uniform_(final.weight, -1e-3, 1e-3)
            nn.init.constant_(final.bias, 0.0)

    def forward(self, x):
        return self.net(x)

class DQNTrainer:
    def __init__(self, env, network_class, lr=LR, seed=0):
        self.env = env
        self.seed = seed
        set_seed(seed)
        self.input_dim = 4
        self.output_dim = env.action_space.n

        self.policy_net = network_class(self.input_dim, self.output_dim).to(DEVICE)
        self.target_net = network_class(self.input_dim, self.output_dim).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(BUFFER_CAPACITY)

        self.steps_done = 0
        self.epsilon = EPS_START
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, explore=True):
    
        if explore:
            if random.random() > self.epsilon:
                self.policy_net.eval()
                with torch.no_grad():
                    s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    q = self.policy_net(s)
                    action = int(q.argmax(dim=1).item())
                self.policy_net.train()
            else:
                action = self.env.action_space.sample()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                q = self.policy_net(s)
                action = int(q.argmax(dim=1).item())
            self.policy_net.train()
        return action

    def optimize_model(self):
        if len(self.replay) < MIN_REPLAY_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(BATCH_SIZE)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * GAMMA * next_q

        current_q = self.policy_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def train(self, num_episodes=EPISODES_PER_SEED):
        episode_rewards = []
        total_steps = 0
        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset()
            state = decode_state_index(int(obs), self.env)
            ep_reward = 0.0
            done = False
            while not done:
                action = self.select_action(state, explore=True)
                step_ret = self.env.step(action)
                if len(step_ret) == 5:
                    ns_raw, r, terminated, truncated, info = step_ret
                    done = bool(terminated or truncated)
                else:
                    ns_raw, r, done, info = step_ret
                next_state = decode_state_index(int(ns_raw), self.env)

                self.replay.push(state, action, float(r), next_state, float(done))
                state = next_state
                ep_reward += float(r)
                total_steps += 1

                # update
                _loss = self.optimize_model()

            # decay epsilon per episode 
            self.epsilon = max(EPS_END, EPS_START * (EPS_DECAY)**ep/10)
            episode_rewards.append(ep_reward)

            # optionally print
            if ep % 50 == 0:
                recent = np.mean(episode_rewards[-50:])
                print(f"Seed {self.seed} Ep {ep}/{num_episodes}  recent_avg={recent:.2f}  eps={self.epsilon:.3f}  replay={len(self.replay)}")

        return episode_rewards

    def evaluate(self, num_episodes=100):
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            state = decode_state_index(int(obs), self.env)
            done = False
            ep_r = 0.0
            while not done:
                action = self.select_action(state, explore=False)
                step_ret = self.env.step(action)
                if len(step_ret) == 5:
                    ns_raw, r, terminated, truncated, info = step_ret
                    done = bool(terminated or truncated)
                else:
                    ns_raw, r, done, info = step_ret
                state = decode_state_index(int(ns_raw), self.env)
                ep_r += float(r)
            rewards.append(ep_r)
        return float(np.mean(rewards)), float(np.std(rewards))


def run_multi_seed_training(env, network_class, model_name_prefix):
    all_seed_rewards = []
    best_model_path = os.path.join(MODEL_DIR, model_name_prefix)
    best_avg = -1e9
    best_rewards_history = None

    for seed in range(NUM_SEEDS):
        print(f"\nStarting seed {seed} for {model_name_prefix}")
        set_seed(seed)
        trainer = DQNTrainer(env, network_class, lr=LR, seed=seed)
        rewards = trainer.train(num_episodes=EPISODES_PER_SEED)

        avg_last100 = float(np.mean(rewards[-100:]))
        print(f"Seed {seed} finished. avg_last100={avg_last100:.2f}")

        if avg_last100 > best_avg:
            best_avg = avg_last100
            best_rewards_history = rewards
            path = os.path.join(MODEL_DIR, model_name_prefix)
            torch.save(trainer.policy_net.state_dict(), path)
            print(f"Saved new best model to {path} (avg_last100={best_avg:.2f})")

        all_seed_rewards.append(rewards)

    maxlen = max(len(r) for r in all_seed_rewards)
    padded = [r + [r[-1]] * (maxlen - len(r)) for r in all_seed_rewards]
    mean_curve = np.mean(np.array(padded), axis=0)
    return mean_curve, best_model_path, best_rewards_history


def main():
    # set global seed for reproducibility
    GLOBAL_SEED = 509
    set_seed(GLOBAL_SEED)

    train_env = MultiGoalCliffWalkingEnv(train=True)
    eval_env = MultiGoalCliffWalkingEnv(train=False)

    # Train linear agent
    print("Training Linear Agent...")
    linear_mean_curve, linear_model_path, linear_history = run_multi_seed_training(train_env, LinearDQN, "best linear.pt")

    # Train non-linear agent
    print("Training Non-Linear Agent...")
    nonlinear_mean_curve, nonlinear_model_path, nonlinear_history = run_multi_seed_training(train_env, NonLinearDQN, "best nonlinear.pt")

    # Plot and save training reward curves (moving average 50)
    def save_plot(curve, filename, title):
        window = 50
        movavg = np.convolve(curve, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(8,4))
        plt.plot(movavg)
        plt.title(title)
        plt.xlabel('Episode (smoothed)')
        plt.ylabel('Average reward')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    save_plot(linear_mean_curve, os.path.join(PLOT_DIR, "cliff average rewards linear.png"), "Cliff - Linear DQN Average Rewards")
    save_plot(nonlinear_mean_curve, os.path.join(PLOT_DIR, "cliff average rewards nonlinear.png"), "Cliff - Nonlinear DQN Average Rewards")

    print("Evaluation...")
    results = {}

    linear_model = LinearDQN(4, train_env.action_space.n).to(DEVICE)
    linear_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best linear.pt"), map_location=DEVICE))
    linear_trainer = DQNTrainer(eval_env, LinearDQN, lr=LR)
    linear_trainer.policy_net = linear_model
    mean_l, std_l = linear_trainer.evaluate(num_episodes=100)
    results['linear'] = {'mean': mean_l, 'std': std_l}

    nonlinear_model = NonLinearDQN(4, train_env.action_space.n).to(DEVICE)
    nonlinear_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best nonlinear.pt"), map_location=DEVICE))
    nonlinear_trainer = DQNTrainer(eval_env, NonLinearDQN, lr=LR)
    nonlinear_trainer.policy_net = nonlinear_model
    mean_n, std_n = nonlinear_trainer.evaluate(num_episodes=100)
    results['nonlinear'] = {'mean': mean_n, 'std': std_n}

    # Save JSON
    json_path = os.path.join(EVAL_DIR, "cliff evaluation results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print("Done. Models, plots, and evaluation results saved under Q1/")


if __name__ == '__main__':
    main()
