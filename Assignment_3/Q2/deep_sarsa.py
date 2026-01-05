import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import time
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
# current best by gamma 0.97, hard, LR 1e-3, decay rate 0.996, train epochs, AdamW, episodes 24k, batch size 128, hidden layers: 64, 128, 32 
BATCH_SIZE = 128
GAMMA = 0.97
LR = 1e-4 # lower is better
TRAIN_EPOCHS = 10
TARGET_UPDATE_FREQ = 1000 # steps for hard update
NUM_EPISODES = 32000
MAX_T = 1000 # max timesteps
TAU = 0.1 # soft update
TARGET_UPDATE_STYLE = 'hard'

# Epsilon-greedy parameters
EPS_START = 1.0
EPS_END = 0.02
DECAY_RATE = 0.996

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        # self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 128)        
        self.layer4 = nn.Linear(128, action_dim)
        
        # Call the custom initialization method
        self._initialize_weights()

    def forward(self, state):
        x = F.relu(self.layer1(state))
        # x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x) 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.layer4.weight, a=-3e-3, b=3e-3)
        if self.layer4.bias is not None:
            nn.init.constant_(self.layer4.bias, 0)

 
class DeepSARSAAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.steps_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Using device: {self.device} ---")

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR)
        self.loss_fn = nn.SmoothL1Loss()

        self.batch_memory = []
        self.experience_tuple = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "next_action", "done"])

        self.epsilon = EPS_START

    def select_action(self, state, episode, use_epsilon=True):
        if use_epsilon:
            # if self.epsilon >= 0.4:
            self.epsilon = max(EPS_END, EPS_START*(DECAY_RATE**(episode//50)))
            # else:
            #     self.epsilon = max(EPS_END, 0.4 - (episode - 15000)/10000)
            if random.random() > self.epsilon:
                use_epsilon = False # Go to exploitation
            else:
                return random.choice(np.arange(self.action_dim))

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.target_net(state_tensor) # notice that target network being used
        return q_values.argmax().item()

    def select_action_boltzmann(self, state, episode, use_epsilon=True):

        if use_epsilon:
            self.epsilon = max(EPS_END*10, EPS_START*10 - episode / 100.0, 5 - episode/500.0)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor) # notice that policy network being used

        scaled_q_values = q_values / self.epsilon

        # Apply softmax to get action probabilities
        action_probs = F.softmax(scaled_q_values, dim=1)
        action = action_probs.multinomial(num_samples=1).squeeze().item()

        return action



    def train_on_batch(self):

        if len(self.batch_memory) < BATCH_SIZE:
            self.batch_memory.clear() 
            return

        states = torch.tensor(np.vstack([e.state for e in self.batch_memory]),
                              dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.vstack([e.action for e in self.batch_memory]),
                               dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.vstack([e.reward for e in self.batch_memory]),
                               dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.vstack([e.next_state for e in self.batch_memory]),
                                   dtype=torch.float32).to(self.device)
        next_actions = torch.tensor(np.vstack([e.next_action for e in self.batch_memory]),
                                    dtype=torch.int64).to(self.device)
        dones = torch.tensor(np.vstack([e.done for e in self.batch_memory]).astype(np.uint8),
                             dtype=torch.float32).to(self.device)

        dataset_size = len(self.batch_memory)

        # CAN IMPROVE THIS BY CHOOSING NON UNIFORMLY
        for _ in range(TRAIN_EPOCHS):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            for start_idx in range(0, dataset_size, BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, dataset_size)
                batch_indices = indices[start_idx:end_idx]

                b_states = states[batch_indices]
                b_actions = actions[batch_indices]
                b_rewards = rewards[batch_indices]
                b_next_states = next_states[batch_indices]
                b_next_actions = next_actions[batch_indices]
                b_dones = dones[batch_indices]

                current_q_values = self.policy_net(b_states).gather(1, b_actions)
                with torch.no_grad():
                    next_action_indices = self.policy_net(b_next_states).argmax(dim=1).unsqueeze(1)
                    next_q_values = self.target_net(b_next_states).gather(1, next_action_indices)
                    # next_q_values = self.target_net(b_next_states).gather(1, b_next_actions)
                    target_q_values = b_rewards + (GAMMA * next_q_values * (1 - b_dones))

                
                loss = self.loss_fn(current_q_values, target_q_values)
                self.optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
                self.optimizer.step()

        # Discard all data for on policy
        self.batch_memory.clear()


    def run(self):
        env = gym.make("LunarLander-v2")
        scores_window = deque(maxlen=100)
        scores = []
        success_count = 0
        start_time = time.time()
        goal_count = 0
        prev_goal_count = goal_count
        for i_episode in range(1, NUM_EPISODES + 1):
            state, _ = env.reset()
            episode_score = 0

            action = self.select_action(state,0, use_epsilon=True)

            for t in range(MAX_T):
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.steps_count += 1
                if reward >= 100 :
                  goal_count += 1
                done = terminated or truncated
                episode_score += reward

                next_action = self.select_action(next_state, i_episode, use_epsilon=True)
                # next_action = self.select_action_boltzmann(next_state, i_episode, use_epsilon=True)

                if TARGET_UPDATE_STYLE == 'hard':
                    if self.steps_count % TARGET_UPDATE_FREQ == 0:
                        self.target_net.load_state_dict(self.policy_net.state_dict())
                elif TARGET_UPDATE_STYLE == 'soft':
                    target_params = self.target_net.state_dict()
                    policy_params = self.policy_net.state_dict()
                    for key in policy_params:
                        target_params[key] = policy_params[key]*TAU + target_params[key]*(1-TAU)
                    self.target_net.load_state_dict(target_params)

                e = self.experience_tuple(state, action, reward, next_state, next_action, done)
                self.batch_memory.append(e)

                state = next_state
                action = next_action

                if done:
                    break

            self.train_on_batch()

            scores_window.append(episode_score)
            scores.append(episode_score)

            if i_episode % 100 == 0:
                avg_score = np.mean(scores_window)
                # elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f"Episode {i_episode}\tAvg Score (100): {avg_score:.2f}\tGoal_visits: {goal_count-prev_goal_count}\tEpsilon: {self.epsilon:.3f}")
                prev_goal_count = goal_count

            # if (i_episode % 100 == 0) and np.mean(scores_window) >= 150.0:
            #     print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
            #     torch.save(self.policy_net.state_dict(), "lunar_lander_sarsa_onpolicy.pth")
            if np.mean(scores_window) >= 200.0:
                print(f"\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}")
                torch.save(self.policy_net.state_dict(), "models/lunar_lander_sarsa_onpolicy.pth")
                success_count += 1
                
            if success_count >= 20:
                break

            
            
                

        env.close()
        print("--- Training Complete ---")
        

def save_gif(agent,seed):
    env = gym.make("LunarLander-v2", render_mode='rgb_array')
    total_reward = 0.0
    state, _ = env.reset(seed = seed)
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.select_action(state, 0.0)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    env.close()
    imageio.mimsave('gifs/lunarlander1.gif', frames, fps= 30)
    print(f"Total reward obtained: {total_reward}")


def evaluate(agent):
    best_reward = -np.inf
    best_frames = []
    rewards = np.zeros(100)
    env = gym.make("LunarLander-v2", render_mode = 'rgb_array')
    with torch.no_grad():
        for sed in range(100):
            total_reward = 0.0
            state, _ = env.reset()
            done = False
            frames = []
            for i in range(1000):
                frame = env.render()
                frames.append(frame)
                action = agent.select_action(state, 0, use_epsilon = False)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    final_frame = env.render()
                    frames.append(final_frame) 
                    break
            if total_reward > best_reward:
                best_reward = total_reward
                best_frames = frames
            rewards[sed] = total_reward
    imageio.mimsave('gifs/lunarlander1.gif', best_frames, fps= 30)
    print(f"   Mean reward: {np.mean(rewards):.4f}")
    print(f"   Standard Deviation (STD): {np.std(rewards):.4f}")
    print(f"   Best reward achieved: {best_reward:.4f}")

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode = 'rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    agent = DeepSARSAAgent(state_dim, action_dim)
    agent.run()
    evaluate(agent)
    