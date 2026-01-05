import numpy as np
import itertools
from collections import deque

# from scipy.stats import mode
import sys
import copy
import time
import random
import matplotlib.pyplot as plt
from tqdm import trange



from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

import math
import random
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

seed = 42
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed)




# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
device="cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PolicyModel(nn.Module):

    def __init__(self, n_observations, n_actions,n_classes):
        super(PolicyModel, self).__init__()


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
 
        return x



class TargetModel(nn.Module):

    def __init__(self, n_observations, n_actions,n_classes):
        super(TargetModel, self).__init__()


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
 
        return x

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import time
import math
import copy
from itertools import product
# current best at LR = 2e-4 and original network
# --- Hyperparameters ---
BUFFER_SIZE = 100_000   # Replay buffer size
BATCH_SIZE = 128         # Mini-batch size
GAMMA = 0.96             
LR = 2e-4               # Learning rate
MIN_LR = 1e-6
TAU = 1e-2              # For soft target network updates
TARGET_UPDATE_STYLE = 'hard'
NUM_EPISODES = 35000     
EPS_START = 1.0         
EPS_END = 0.1          
EPS_DECAY = 0.996       
TARGET_UPDATE_FREQ = 1000  # For hard target updates

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# class QNetwork(nn.Module):
#     def __init__(self, obs_dim, num_actions):
#         super(QNetwork, self).__init__()
        
#         self.network = nn.Sequential(
#             nn.Linear(obs_dim, 256),
#             nn.LayerNorm(256),
#             # nn.Dropout(p=0.2),
#             nn.ReLU(),
#             nn.Linear(256, 1024),
#             nn.LayerNorm(1024),
#             # nn.Dropout(p=0.3),
#             nn.ReLU(),
#             nn.Linear(1024, num_actions)
#         )

#         # apply Kaiming init to hidden layers
#         self.network.apply(init_weights)

#         # small uniform init for final (output) layer
#         nn.init.uniform_(self.network[-1].weight, -1e-3, 1e-3)
#         nn.init.constant_(self.network[-1].bias, 0.0)

#     def forward(self, state):
#         return self.network(state)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super(QNetwork, self).__init__()
        self.num_actions = num_actions
        
        # Shared MLP body
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.LayerNorm(1024),
            nn.ReLU()
        )

        # 1. Value Stream (computes V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1) # Single output for state value
        )
        
        # 2. Advantage Stream (computes A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions) 
        )

        # Apply init (we apply to each new part)
        self.body.apply(init_weights)
        self.value_stream.apply(init_weights)
        self.advantage_stream.apply(init_weights)
        
        # Small uniform init for final advantage layer
        nn.init.uniform_(self.advantage_stream[-1].weight, -1e-3, 1e-3)
        nn.init.constant_(self.advantage_stream[-1].bias, 0.0)

    def forward(self, state):
        # Pass state through the shared body
        shared_embedding = self.body(state)
        
        # Get V(s) and A(s,a)
        value = self.value_stream(shared_embedding)
        advantages = self.advantage_stream(shared_embedding)
        
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
        
        # Simple 1-step experience
        self.experience = namedtuple("Experience", 
                                     field_names=["state", "action_index", "reward", 
                                                  "next_state", "done"])

    def push(self, state, action_index, reward, next_state, done):
        """Adds a 1-step experience to memory."""
        e = self.experience(state, action_index, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, k=batch_size)
        
        states = torch.tensor(
            np.vstack([e.state for e in experiences if e is not None]), 
            dtype=torch.float32).to(self.device)
        action_indices = torch.tensor(
            np.vstack([e.action_index for e in experiences if e is not None]), 
            dtype=torch.int64).to(self.device)
        rewards = torch.tensor(
            np.vstack([e.reward for e in experiences if e is not None]), 
            dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            np.vstack([e.next_state for e in experiences if e is not None]), 
            dtype=torch.float32).to(self.device)
        dones = torch.tensor(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8), 
            dtype=torch.float32).to(self.device)

        return (states, action_indices, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

class StateManager:
    """
    Handles feature engineering outside the environment.
    It takes the raw state from the env and computes the 27-dim state.
    """
    def __init__(self, num_assets, step_limit):
        self.num_assets = num_assets
        self.step_limit = step_limit
        self.price_history = np.zeros((self.num_assets, self.step_limit + 1))
        self.time_step = 0

    def reset(self, raw_initial_state):
        """
        Resets the history and computes the initial engineered state.
        raw_initial_state = [cash, price1..5, shares1..5, time]
        """
        self.time_step = 0
        self.price_history = np.zeros((self.num_assets, self.step_limit + 1))
        
        # Extract initial prices
        initial_prices = raw_initial_state[1 : 1 + self.num_assets]
        self.price_history[:, 0] = initial_prices
        
        # Return the feature-engineered state for t=0
        return self._compute_features(raw_initial_state)

    def process_state(self, raw_state):
        """
        Computes the engineered state for the current timestep.
        raw_state = [cash, price1..5, shares1..5, time]
        """
        self.time_step = int(raw_state[-1]) # Get time from the raw state
        
        # Store current prices in our history
        current_prices = raw_state[1 : 1 + self.num_assets]
        if self.time_step <= self.step_limit:
            self.price_history[:, self.time_step] = current_prices
        
        return self._compute_features(raw_state)

    def _get_price(self, t):
        """Helper to safely get price at time t."""
        if t < 0:
            return self.price_history[:, 0] # Repeat initial price for t<0
        return self.price_history[:, t]

    def _compute_features(self, raw_state):
        """Calculates the 27-dimensional feature-engineered state."""
        
        t = self.time_step
        
        # Extract data from raw state
        cash = raw_state[0]
        current_prices = raw_state[1 : 1 + self.num_assets]
        holdings = raw_state[1 + self.num_assets : 1 + 2 * self.num_assets]

        # 1. Current Price (5 features)
        # (already have current_prices)
        
        # 2. Change in price (t-1) -> (t) (5 features)
        price_t_minus_1 = self._get_price(t - 1)
        change_t_1 = current_prices - price_t_minus_1
        
        # 3. Change in price (t-2) -> (t-1) (5 features)
        price_t_minus_2 = self._get_price(t - 2)
        change_t_2 = price_t_minus_1 - price_t_minus_2
        
        # 4. Diff from mean (5 features)
        episode_prices = self.price_history[:, :t + 1]
        mean_price = np.mean(episode_prices, axis=1)
        diff_from_mean = current_prices - mean_price
        
        # 5. Current Holdings (5 features)
        # (already have holdings)
        
        # 6. Time and Cash (2 features)
        time_feature = np.array([t / self.step_limit]) 
        cash_feature = np.array([cash])
        
        # Concatenate all 27 features
        engineered_state = np.concatenate([
            current_prices,
            change_t_1,
            change_t_2,
            diff_from_mean,
            holdings,
            time_feature,
            cash_feature
        ]).astype(np.float32)
        
        return engineered_state

class DQNAgent:
    def __init__(self, env, engineered_obs_dim):
        self.env = env
        self.obs_dim = engineered_obs_dim 
        self.num_assets = env.num_assets
        self.lot_size = env.lot_size
        
        self.num_actions_per_asset = (2 * self.lot_size) + 1 
        self.total_actions = self.num_actions_per_asset ** self.num_assets 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Agent Initialized (1-Step DQN) ---")
        print(f"Device: {self.device}")
        print(f"Engineered State Dim: {self.obs_dim}")
        
        self.gamma = GAMMA 

        self.policy_net = QNetwork(self.obs_dim, self.total_actions).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.total_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.memory = ReplayBuffer(BUFFER_SIZE, self.device)
        
        self.steps_done = 0
        self.epsilon = EPS_START

    def _map_index_to_vector(self, index):
        action_vector = []
        temp_index = index
        base = self.num_actions_per_asset
        for i in range(self.num_assets):
            action_0_to_4 = temp_index % base
            action_minus_2_to_2 = action_0_to_4 - self.lot_size
            action_vector.append(action_minus_2_to_2)
            temp_index //= base
        return np.array(action_vector)

    def _map_vector_to_index(self, action_vector):
        index = 0
        base = self.num_actions_per_asset
        for i in range(self.num_assets):
            action_0_to_4 = action_vector[i] + self.lot_size
            index += action_0_to_4 * (base ** i)
        return int(index)

    def select_action(self, state, greedy=False):
            # Epsilon decay
            if not greedy:
                self.epsilon = max(EPS_END, EPS_START * EPS_DECAY**(self.steps_done//400.0))
                self.steps_done += 1
            
            # Select action
            if not greedy and random.random() < self.epsilon:
                action_index = random.randrange(self.total_actions)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                action_index = q_values.argmax().item()
            
            action_vector = self._map_index_to_vector(action_index)
            return action_vector, action_index

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample 1-step transitions
        states, action_indices, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        with torch.no_grad():
            next_action_indices = self.policy_net(next_states).argmax(dim=1).unsqueeze(1)
            # Evaluate those actions with *target_net*
            next_q_values = self.target_net(next_states).gather(1, next_action_indices)
            
            # Standard 1-step Bellman equation
            target_q_values = rewards + ( self.gamma * next_q_values * (1 - dones) )

        # --- 2. Calculate Current Q-Values ---
        current_q_values_all = self.policy_net(states)
        current_q_values = current_q_values_all.gather(1, action_indices)

        # --- 3. Compute Loss ---
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # --- 4. Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 24.0)
        self.optimizer.step()

        if TARGET_UPDATE_STYLE == 'soft':
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(TAU * policy_param.data + (1.0 - TAU) * target_param.data)
        elif self.steps_done % TARGET_UPDATE_FREQ == 0:
             self.target_net.load_state_dict(self.policy_net.state_dict())
             
        return loss.item()

def plot_loss(loss_history):
    """Plots the training loss over optimization steps."""
    plt.figure(figsize=(12, 6))
    plt.plot(loss_history)
    plt.title("Training Loss vs. Optimization Steps")
    plt.xlabel("Optimization Steps")
    plt.ylabel("Smooth L1 Loss")
    plt.grid(True)
    plt.show()

def get_portfolio_value(raw_state, num_assets):
    """Helper to calculate portfolio value from a raw state."""
    cash = raw_state[0]
    prices = raw_state[1 : 1 + num_assets]
    holdings = raw_state[1 + num_assets : 1 + 2 * num_assets]
    return cash + np.dot(prices, holdings)

def evaluate_model(agent, env, manager, num_seeds=100):
    """Evaluates the trained agent over 100 seeds."""
    print("\n--- Starting Evaluation ---")
    
    # Try to load the high-score model
    try:
        agent.policy_net.load_state_dict(torch.load("dqn_portfolio_model_highscore.pth"))
        print("Loaded 'dqn_portfolio_model_highscore.pth' for evaluation.")
    except FileNotFoundError:
        print("Evaluation model not found. Using the model from end of training.")
        
    agent.policy_net.eval() # Set model to evaluation mode
    
    num_steps = env.step_limit
    num_assets = env.num_assets
    
    # Store wealth for all seeds and all timesteps
    # (seeds, timesteps + 1) -> +1 for initial wealth
    all_wealths = np.zeros((num_seeds, num_steps + 1))
    
    for i in trange(num_seeds): # Use trange for a progress bar
        raw_state = env.reset() # Assuming env.reset() returns (state, info)
        state = manager.reset(raw_state)
        
        all_wealths[i, 0] = env.initial_cash
        
        for t in range(num_steps):
            # Use greedy action selection (no exploration)
            action_vector, _ = agent.select_action(state, greedy=True)
            
            next_raw_state, reward, done, _ = env.step(action_vector)
            next_state = manager.process_state(next_raw_state)
            
            state = next_state
            raw_state = next_raw_state
            
            # Calculate and store current portfolio value
            current_value = get_portfolio_value(raw_state, num_assets)
            all_wealths[i, t + 1] = current_value
            
            if done:
                # If done early, fill remaining steps with last value
                all_wealths[i, t+2:] = current_value
                break      
    return all_wealths

def plot_evaluation(all_wealths):
    """Plots the mean wealth and std deviation from evaluation."""
    mean_wealth = np.mean(all_wealths, axis=0)
    std_wealth = np.std(all_wealths, axis=0)
    
    timesteps = np.arange(len(mean_wealth))
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_wealth, label="Mean Portfolio Wealth", color="blue", lw=2)
    
    # Create the shaded standard deviation area
    plt.fill_between(timesteps, 
                     mean_wealth - std_wealth, 
                     mean_wealth + std_wealth, 
                     color="blue", alpha=0.2, label="Std. Deviation")
    
    plt.title(f"Portfolio Wealth Over 100 Seeds (Dueling DDQN)")
    plt.xlabel("Timestep")
    plt.ylabel("Portfolio Wealth ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Report final ratio
    mean_final_wealth = mean_wealth[-1]
    std_final_wealth = std_wealth[-1]
    
    if std_final_wealth > 0:
        ratio = mean_final_wealth / std_final_wealth
        print(f"\n--- Evaluation Report ---")
        print(f"Mean Final Wealth: {mean_final_wealth:.2f}")
        print(f"Std. Dev Final Wealth: {std_final_wealth:.2f}")
        print(f"Mean/Std. Dev Ratio: {ratio:.2f}")
    else:
        print("Final standard deviation is zero.")
             
if __name__ == "__main__":
    
    env = DiscretePortfolioOptEnv() 

    manager = StateManager(env.num_assets, env.step_limit)
    _temp_raw_state = env.reset()
    ENGINEERED_STATE_DIM = manager.reset(_temp_raw_state).shape[0] 
    
    agent = DQNAgent(env, engineered_obs_dim=ENGINEERED_STATE_DIM)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer, gamma=0.999)
    scores = []
    scores_window = deque(maxlen=100)
    loss_history = []
    start_time = time.time()
    
    print("\n--- Starting Training ---")
    
    for i_episode in range(1, NUM_EPISODES + 1):
        raw_state = env.reset()
        state = manager.reset(raw_state) 
        
        episode_score = 0
        for t in range(env.step_limit ): 
            action_vector, action_index = agent.select_action(state)
            if t >= env.step_limit:
                next_raw_state, reward, done = raw_state, 0.0, True
            else:
                next_raw_state, reward, done, _ = env.step(action_vector)
            
            next_state = manager.process_state(next_raw_state)
            agent.memory.push(state, action_index, reward, next_state, done)
            state = next_state
            raw_state = next_raw_state
            
            if not (t >= env.step_limit):
                episode_score += reward
            
            loss = agent.learn()
            if loss is not None:
                loss_history.append(loss)
            
            
            if done:
                break
        
        scheduler.step()
        for param_group in agent.optimizer.param_groups:
            if param_group['lr'] < MIN_LR:
                param_group['lr'] = MIN_LR
        scores_window.append(episode_score)
        scores.append(episode_score)
        
        if i_episode % 100 == 0:
            avg_score = np.mean(scores_window)
            # elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"Episode {i_episode}\tAvg Score (100): {avg_score:.2f}\tEpsilon: {agent.epsilon:.3f}")
            if avg_score >= 10.0:
                print(f"\n--- Environment Good in {i_episode} episodes! ---")
                torch.save(agent.policy_net.state_dict(), "dqn_portfolio_model.pth")
            if avg_score >= 18.0:
                print(f"\n--- Environment Solved in {i_episode} episodes! ---")
                torch.save(agent.policy_net.state_dict(), "dqn_portfolio_model_highscore.pth")
                break
        
            
    # env.close() 
    print("--- Training Complete ---")
    
    plot_loss(loss_history)
    
    # Run evaluation
    evaluation_wealths = evaluate_model(agent, env, manager, num_seeds=100)
    
    # Plot evaluation
    plot_evaluation(evaluation_wealths)

