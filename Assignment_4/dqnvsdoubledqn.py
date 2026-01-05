import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time
import os
import json
import matplotlib.pyplot as plt


BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 20000
GAMMA = 0.99 
LR = 1e-4
TARGET_UPDATE_FREQ = 2000 # steps for hard update
NUM_EPISODES = 5000 
MAX_T = 1000 # max timesteps per episode

# Epsilon-greedy parameters
EPS_START = 1.0
EPS_END = 0.01
DECAY_RATE = 0.9955 


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer3 = nn.Linear(128, 128)      
        self.layer4 = nn.Linear(128, action_dim)
        
        self._initialize_weights()

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer3(x))
        return self.layer4(x) 

    def _initialize_weights(self):
        # Kaiming init for ReLU layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        nn.init.uniform_(self.layer4.weight, a=-3e-3, b=3e-3)
        if self.layer4.bias is not None:
            nn.init.constant_(self.layer4.bias, 0)


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device):
        
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQNAgent:

    def __init__(self, state_dim, action_dim, agent_type="DQN"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_type = agent_type
        self.steps_count = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} for {self.agent_type} ---")

        # Q-Networks
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss() 

        # Replay memory
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE, BATCH_SIZE, self.device)
        
        self.epsilon = EPS_START

    def select_action(self, state, use_epsilon=True):
        if use_epsilon and random.random() < self.epsilon:
            return random.choice(np.arange(self.action_dim))
        
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def step(self, state, action, reward, next_state, done):
        
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        self.steps_count += 1
        

        if self.steps_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            

        if len(self.replay_buffer) > BATCH_SIZE:
            self.learn()

    def learn(self):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        with torch.no_grad():
            if self.agent_type == "DQN":
                # Standard DQN: Q_target = r + gamma * max_a' Q_target(s', a')
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                
            elif self.agent_type == "DDQN":
                # Double DQN:
                # 1. Get best action a* from policy_net: a* = argmax_a' Q_policy(s', a')
                best_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                # 2. Evaluate that action a* with target_net: Q_target = r + gamma * Q_target(s', a*)
                next_q_values = self.target_net(next_states).gather(1, best_actions)

            # Compute the final target Q-value
            target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        current_q_values = self.policy_net(states).gather(1, actions)
        

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self, i_episode):
        self.epsilon = max(EPS_END, EPS_START * (DECAY_RATE**(i_episode // 20)))

def train_agent(agent, agent_name):
    print(f"\n--- Training {agent_name} ---")
    env = gym.make("LunarLander-v2")
    scores_window = deque(maxlen=100)
    all_scores = []
    best_score = 0
    start_time = time.time()
    
    for i_episode in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        episode_score = 0
        
        agent.update_epsilon(i_episode)
        
        for t in range(MAX_T):
            action = agent.select_action(state, use_epsilon=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_score += reward
            if done:
                break
                
        scores_window.append(episode_score)
        all_scores.append(episode_score)
        
        if i_episode % 100 == 0:
            avg_score = np.mean(scores_window)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"Episode {i_episode}\tAvg Score (100): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}\tTime: {elapsed}")
        
            if avg_score >= 200.0:
                print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
                if avg_score > best_score:
                    best_score = avg_score
                    torch.save(agent.policy_net.state_dict(), f"models/{agent_name.lower()}.pt")
            
            
    print(f"--- Training Complete. Model saved to models/{agent_name.lower()}.pt ---")
    env.close()
    return all_scores


def plot_rewards(dqn_scores, ddqn_scores):

    print("--- Plotting Training Rewards ---")
    
    # Calculate rolling average
    def rolling_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    window = 100
    dqn_avg = rolling_average(dqn_scores, window)
    ddqn_avg = rolling_average(ddqn_scores, window)

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(dqn_avg)), dqn_avg, label="DQN (100-ep avg)")
    plt.plot(np.arange(len(ddqn_avg)), ddqn_avg, label="Double DQN (100-ep avg)")
    
    # Also plot raw scores for transparency
    plt.plot(np.arange(len(dqn_scores)), dqn_scores, label="DQN (Raw)", alpha=0.2)
    plt.plot(np.arange(len(ddqn_scores)), ddqn_scores, label="Double DQN (Raw)", alpha=0.2)
    
    plt.title("Training Rewards (Rolling Average)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/reward_curves.png")
    plt.close()
    print("Reward plot saved to plots/reward_curves.png")

def evaluate_agent(agent, num_episodes=100):
    env = gym.make("LunarLander-v2")
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        for t in range(MAX_T):
            action = agent.select_action(state, use_epsilon=False) # Greedy policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        rewards.append(total_reward)
        
    env.close()
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward


def evaluate_and_plot_q_values(dqn_agent, ddqn_agent):
    
    print("--- Plotting Q-Values Per Action ---")
    

    def get_q_values_for_episode(agent, seed=42):
        env = gym.make("LunarLander-v2")
        state, _ = env.reset(seed=seed)
        

        q_values = {i: [] for i in range(agent.action_dim)}
        done = False
        t = 0
        
        while not done and t < MAX_T:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                all_qs = agent.policy_net(state_tensor).cpu().numpy().flatten()
                
                # Record Q-values for all actions
                for i in range(agent.action_dim):
                    q_values[i].append(all_qs[i])
                
                # Select action based on these Q-values
                action = np.argmax(all_qs)
                
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            t += 1
            
        env.close()
        return q_values

    # Get Q-values for both agents on the same episode
    dqn_q_values = get_q_values_for_episode(dqn_agent)
    ddqn_q_values = get_q_values_for_episode(ddqn_agent)
    

    fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Q-Value Comparison During Evaluation (Seed=42)", fontsize=16)
    actions_map = ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"]

    for i in range(4):
        ax = axs[i // 2, i % 2]
        
        len_dqn = len(dqn_q_values[i])
        len_ddqn = len(ddqn_q_values[i])
        min_len = min(len_dqn, len_ddqn)
        
        ax.plot(np.arange(min_len), dqn_q_values[i][:min_len], label="DQN")
        ax.plot(np.arange(min_len), ddqn_q_values[i][:min_len], label="Double DQN", linestyle="--")
        
        ax.set_title(f"Action: {actions_map[i]} ({i})")
        ax.set_ylabel("Predicted Q-Value")
        ax.legend()
        ax.grid(True)

    axs[1, 0].set_xlabel("Timestep")
    axs[1, 1].set_xlabel("Timestep")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig("plots/q_values_per_action.png")
    plt.close()
    print("Q-value plot saved to plots/q_values_per_action.png")



if __name__ == "__main__":
    

    os.makedirs("plots", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Get env parameters
    env = gym.make("LunarLander-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()


    dqn_agent = DQNAgent(state_dim, action_dim, agent_type="DQN")
    dqn_scores = train_agent(dqn_agent, "DQN")
    

    ddqn_agent = DQNAgent(state_dim, action_dim, agent_type="DDQN")
    ddqn_scores = train_agent(ddqn_agent, "Double DQN")


    plot_rewards(dqn_scores, ddqn_scores)
    

    print("\n--- Evaluating Agents ---")
    

    dqn_mean, dqn_std = evaluate_agent(dqn_agent)
    print(f"DQN Evaluation (100 episodes):     Mean={dqn_mean:.4f}, Std={dqn_std:.4f}")
    
    ddqn_mean, ddqn_std = evaluate_agent(ddqn_agent)
    print(f"DDQN Evaluation (100 episodes):    Mean={ddqn_mean:.4f}, Std={ddqn_std:.4f}")


    evaluation_results = {
        "dqn": {
            "mean_return": dqn_mean,
            "std_return": dqn_std
        },
        "ddqn": {
            "mean_return": ddqn_mean,
            "std_return": ddqn_std
        }
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print("\nEvaluation results saved to evaluation_results.json")


    # We can use the same agents, no need to reload
    evaluate_and_plot_q_values(dqn_agent, ddqn_agent)
    
    print("\n--- All tasks complete ---")