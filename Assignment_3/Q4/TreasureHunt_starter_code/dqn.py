import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time
import math
from env import TreasureHunt_v2

BUFFER_SIZE = 10000   
BATCH_SIZE = 32
CANDIDATE_MULTIPLIER = 8 # Sample 4x the batch size, then pick the best
CANDIDATE_BATCH_SIZE = BATCH_SIZE * CANDIDATE_MULTIPLIER # 256
GAMMA = 0.97            # Discount factor
LR = 1e-3               # Learning rate 
NUM_EPISODES = 20000     # Max number of training episodes
MAX_T = 500             # Fixed
EPS_START = 1.0         
EPS_END = 0.1          
EPS_DECAY = 0.997       # Epsilon decay rate
TARGET_UPDATE_FREQ = 2000 # How often to update the target network 
TAU = 5e-2              # For soft target network updates 
TARGET_UPDATE_STYLE = 'hard'

class QNetwork(nn.Module):

    def __init__(self, in_channels, num_actions):
        super(QNetwork, self).__init__()

        # Input: B x 4 x 10 x 10
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        # Shape: B x 64 x 10 x 10
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # Shape: B x 64 x 5 x 5
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        # Shape: B x 64 x 3 x 3

        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the conv layers
        x = torch.flatten(x, 1) 

        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.device = device

    def push(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, candidate_batch_size):
        """
        Samples a large 'candidate' batch uniformly.
        The agent will then re-sample from this.
        """
        experiences = random.sample(self.buffer, k=candidate_batch_size)

        states = torch.tensor(
            np.array([e.state for e in experiences if e is not None]),
            dtype=torch.float32).to(self.device)

        actions = torch.tensor(
            np.vstack([e.action for e in experiences if e is not None]),
            dtype=torch.int64).to(self.device)

        rewards = torch.tensor(
            np.vstack([e.reward for e in experiences if e is not None]),
            dtype=torch.float32).to(self.device)

        next_states = torch.tensor(
            np.array([e.next_state for e in experiences if e is not None]),
            dtype=torch.float32).to(self.device)

        dones = torch.tensor(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8),
            dtype=torch.float32).to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_dim, gamma=GAMMA):
        self.state_shape = state_shape
        self.action_dim = action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        self.gamma = gamma

        self.batch_size = BATCH_SIZE
        self.candidate_multiplier = 4 # Sample 4x the batch size
        self.candidate_batch_size = self.batch_size * self.candidate_multiplier

        self.policy_net = QNetwork(state_shape[0], action_dim).to(self.device)
        self.target_net = QNetwork(state_shape[0], action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        
        self.loss_fn = nn.SmoothL1Loss(reduction='none') 
        
        self.memory = ReplayBuffer(BUFFER_SIZE, self.device)

        self.steps_done = 0
        self.epsilon = EPS_START

    def select_action(self, state):
        self.epsilon = max(EPS_END, EPS_START * EPS_DECAY**(self.steps_done//20000))
        self.steps_done += 1

        if random.random() > self.epsilon:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor) 
            return q_values.argmax().item() 
        else:
            return random.choice(np.arange(self.action_dim))

    def push_to_memory(self, state, action, reward, next_state, done):
        
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        
        if len(self.memory) < self.candidate_batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.candidate_batch_size)
        
        with torch.no_grad():
            # Get current Q-values
            current_q_values = self.policy_net(states).gather(1, actions)
            
            next_actions = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + ( self.gamma * next_q_values * (1 - dones) )
            
            # Calculate TD-Error
            td_errors = (target_q_values - current_q_values).squeeze() # Shape (256,)
            
        mean_td_error = td_errors.mean()

        priorities = (td_errors - mean_td_error) ** 2 + 1e-6
        
        # Normalize priorities into a probability distribution
        probabilities = priorities / priorities.sum()
        
        final_batch_indices = torch.multinomial(probabilities, 
                                                num_samples=self.batch_size, 
                                                replacement=True)
        
        # 5. Get the final training batch by indexing the candidate batch
        final_states = states[final_batch_indices]
        final_actions = actions[final_batch_indices]
        final_target_q_values = target_q_values[final_batch_indices]
        
        
        # Get Q-values for the final states/actions
        current_q_values_final = self.policy_net(final_states).gather(1, final_actions)
        
        loss = self.loss_fn(current_q_values_final, final_target_q_values).mean() # Now we take the mean
        
        # 7. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 8. Update target network
        if self.steps_done % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == "__main__":

    env = TreasureHunt_v2()

    state_shape = [4,10,10]
    action_dim = env.env.action_space.n

    agent = DQNAgent(state_shape, action_dim, gamma=GAMMA)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer, gamma=0.999)
    scores = []
    scores_window = deque(maxlen=100) # For avg score
    start_time = time.time()

    print("\n--- Starting Training ---")

    for i_episode in range(1, NUM_EPISODES + 1):
            
            state = env.reset() 
            episode_score = 0
            
            for t in range(MAX_T):
                action = agent.select_action(state)
                
                next_state, reward = env.step(action)
                
                done = False
                if reward > 0.005 and reward < 0.02:
                  done = True
                
                agent.memory.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_score += reward
                
                for i in range(1):
                    agent.learn()
                
                if done:
                    break
            scheduler.step()
            scores_window.append(episode_score)
            scores.append(episode_score)

            if i_episode % 100 == 0:
                    avg_score = np.mean(scores_window)
                    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                    print(f"Episode {i_episode}\tTime: {elapsed_time}\tAvg Score (100): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}")


            if np.mean(scores_window) >= 1.0:
                print(f"\nEnvironment solved in {i_episode} episodes! Avg Score: {np.mean(scores_window):.2f}")
                # torch.save(agent.policy_net.state_dict(), "models/treasure_hunt_dqn.pth")
                break
    torch.save(agent.policy_net.state_dict(), "models/treasure_hunt_dqn.pth")
    env.close()
    state_dict = torch.load("models/treasure_hunt_dqn.pth")
    agent.policy_net.load_state_dict(state_dict)
    policy = np.zeros(env.env.num_states, dtype=int)
    for index in range(env.env.num_states):
        state = env.index_to_spatial(index)
        action = agent.select_action(state)
        policy[index] = action

    print("Test Rewards: ", env.get_policy_rewards(policy))
    env.visualize_policy_execution(policy)
    
    # avg reward : -6
    # best reward : -2