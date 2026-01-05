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

# self.reset(seed)
# r, c = self.agent_pos
# self._get_obs(): return self._state_to_index(r, c, self.checkA, self.checkB)
# self.step(action): return self._get_obs(), reward, terminated, False, {"goal": goal}
# use render_mode="rgb_array" for generating gif.

# -------------------------------------------------------------------------------
# LEFT: Hyperparameter tuning, subsequent images and gif
# convergence for sarsa: GLIE, for Qlearning: all states visited, robbins monro  ; Expected sarsa: 
MAX_STEPS = 1000
ALPHA = 0.55 # optimal for SARSA: 0.55 ; optimal for ESARSA: 0.45; Q learning: 0.55
SEEDS = [4, 5, 7, 9, 11, 12, 16, 24, 27, 28]
MAX_EPISODES = 500 # optimal for sarsa and esarsa: 500
GAMMA = 0.99 # 0.99 for sarsa and esarsa
# Good exploration is needed so either exponential decay or 1/sqrt(episode)
EPSILON_DECAY = 0.99

# -------------------------------------------------------------------------------

def find_policy_epsilon(Q_value, epsilon):
    
    base = np.full(Q_value.shape, epsilon/Q_value.shape[1])
    optimal_policy = np.argmax(Q_value, axis=1)
    # for i in range(Q_value.shape[0]):
    #     base[i, optimal_policy[i]] += 1 - epsilon
    # Better:
    row_indices = np.arange(Q_value.shape[0])
    base[row_indices, optimal_policy] += 1 - epsilon
    
    return base

def find_policy_softmax(Q_value, temp):
    # Prevent division by zero when temp -> 0
    temp = max(temp, 1e-7)  
    
    # Subtract row-wise max for numerical stability
    shifted = Q_value / temp
    shifted = shifted - np.max(shifted, axis=1, keepdims=True)
    
    exp = np.exp(shifted)
    exp = np.clip(exp, 1e-12, None)  
    return exp / np.sum(exp, axis=1, keepdims=True)



def choose_action(policy, state):
    return np.random.choice(np.arange(policy.shape[1]), p=policy[state])

def terminal_states(env):
    term = []
    for i in range(1, env.width-1):
        for j in [True, False]:
            for k in [True, False]:
                term.append(env._state_to_index(env.height - 1, i, j, k))
    term.append(env._state_to_index(env.risky_goal[0], env.risky_goal[1], True, True))
    
    return term
    
    
    
    
    

def SARSA(env):
    '''
    Implement the SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    best_Q_value = np.full((env.observation_space.n, env.action_space.n), -1e9)
    episode_rewards = [0 for _ in range(MAX_EPISODES)]
    safe_visits = 0
    risky_visits = 0
    for seed in SEEDS:
        # Optimistic Initialization: provides a natural incentive for the agent to try unexplored actions because it believes they might lead to a high reward.
        Q_value = np.full((env.observation_space.n, env.action_space.n), 1.0)
        # terminal states to 0
        Q_value[terminal_states(env)] = 0.0
        
        for episode in range(MAX_EPISODES):
            # current_policy = find_policy_epsilon(Q_value, EPSILON_DECAY**episode)
            current_policy = find_policy_epsilon(Q_value, 1/math.sqrt((episode + 1)))
            # current_policy = find_policy_softmax(Q_value,1)
            state, _ = env.reset(seed=seed)
            action = choose_action(current_policy, state)
            for _ in range(MAX_STEPS):
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    Q_value[state, action] += ALPHA * (reward - Q_value[state, action])
                    if info["goal"] == "safe":
                        safe_visits += 1.00000
                    elif info["goal"] == "risky":
                        risky_visits += 1.00000
                    break
                else:
                    next_action = choose_action(current_policy, next_state)
                    Q_value[state, action] += ALPHA * (reward + GAMMA * Q_value[next_state, next_action] - Q_value[state, action])
                    episode_rewards[episode] += reward/len(SEEDS)
                state = next_state
                action = next_action
                
        
        start_state = env._state_to_index(env.height -1, 0, False, False)
        # Best Q value by comparing Value function of start state
        if Q_value[start_state,:].max() > best_Q_value[start_state,:].max():
            best_Q_value = Q_value.copy() # else we only create a reference
        
        
    safe_visits /= len(SEEDS*MAX_EPISODES) 
    risky_visits /= len(SEEDS*MAX_EPISODES)

    return best_Q_value, episode_rewards, safe_visits, risky_visits

def q_learning_for_cliff(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    best_Q_value = np.full((env.observation_space.n, env.action_space.n), -1e9)
    episode_rewards = [0 for _ in range(MAX_EPISODES)]
    safe_visits = 0
    risky_visits = 0
    for seed in SEEDS:
        # Optimistic Initialization: provides a natural incentive for the agent to try unexplored actions because it believes they might lead to a high reward.
        Q_value = np.full((env.observation_space.n, env.action_space.n), 1.0)
        # terminal states to 0
        Q_value[terminal_states(env)] = 0.0
        
        for episode in range(MAX_EPISODES):
            # current_policy = find_policy_epsilon(Q_value, EPSILON_DECAY**episode)
            current_policy = find_policy_epsilon(Q_value, 1/math.sqrt(math.sqrt((episode + 1))))
            # current_policy = find_policy_softmax(Q_value,1/(episode + 1))
            state, _ = env.reset(seed=seed)
            action = choose_action(current_policy, state)
            for _ in range(MAX_STEPS):
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    Q_value[state, action] += ALPHA * (reward - Q_value[state, action])
                    if info["goal"] == "safe":
                        safe_visits += 1.00000
                    elif info["goal"] == "risky":
                        risky_visits += 1.00000
                    break
                else:
                    next_action = choose_action(current_policy, next_state)
                    Q_value[state, action] += ALPHA * (reward + GAMMA * np.max(Q_value[next_state,:]) - Q_value[state, action])
                    episode_rewards[episode] += reward/len(SEEDS)
                state = next_state
                action = next_action
                
        
        start_state = env._state_to_index(env.height -1, 0, False, False)
        # Best Q value by comparing Value function of start state
        if Q_value[start_state,:].max() > best_Q_value[start_state,:].max():
            best_Q_value = Q_value.copy() # else we only create a reference
        
        
    safe_visits /= len(SEEDS*MAX_EPISODES) 
    risky_visits /= len(SEEDS*MAX_EPISODES)

    return best_Q_value, episode_rewards, safe_visits, risky_visits

def expected_SARSA(env):
    '''
    Implement the Expected SARSA algorithm to find the optimal policy for the given environment.
    return: best Q table -> np.array of shape (num_states, num_actions)
    return average training_rewards across 10 seeds-> []
    return: average safe_visits across 10 seeds -> float
    return: average risky_visits across 10 seeds -> float
    '''
    best_Q_value = np.full((env.observation_space.n, env.action_space.n), -1e9)
    episode_rewards = [0 for _ in range(MAX_EPISODES)]
    safe_visits = 0
    risky_visits = 0
    for seed in SEEDS:
        # Optimistic Initialization: provides a natural incentive for the agent to try unexplored actions because it believes they might lead to a high reward.
        Q_value = np.full((env.observation_space.n, env.action_space.n), 20.0)
        # terminal states to 0
        Q_value[terminal_states(env)] = 0.0
        
        for episode in range(MAX_EPISODES):
            # current_policy = find_policy_epsilon(Q_value, EPSILON_DECAY**episode)
            current_policy = find_policy_epsilon(Q_value, 1/math.sqrt((episode + 1)))
            # current_policy = find_policy_softmax(Q_value,1/(episode + 1))
            state, _ = env.reset(seed=seed)
            action = choose_action(current_policy, state)
            for _ in range(MAX_STEPS):
                next_state, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    Q_value[state, action] += ALPHA * (reward - Q_value[state, action])
                    if info["goal"] == "safe":
                        safe_visits += 1.00000
                    elif info["goal"] == "risky":
                        risky_visits += 1.00000
                    break
                else:
                    next_action = choose_action(current_policy, next_state)
                    Q_value[state, action] += ALPHA * (reward + GAMMA * np.dot(current_policy[next_state], Q_value[next_state,:]) - Q_value[state, action])
                    episode_rewards[episode] += reward/len(SEEDS)
                state = next_state
                action = next_action
                
        
        start_state = env._state_to_index(env.height -1, 0, False, False)
        # Best Q value by comparing Value function of start state
        if Q_value[start_state,:].max() > best_Q_value[start_state,:].max():
            best_Q_value = Q_value.copy() # else we only create a reference
        
        
    safe_visits /= len(SEEDS*MAX_EPISODES) 
    risky_visits /= len(SEEDS*MAX_EPISODES)

    return best_Q_value, episode_rewards, safe_visits, risky_visits


def run_and_collect_frames(env, policy, max_steps=500):
    """
    Runs the policy in the environment and collects the rendered frames.

    Args:
        env: The CliffWalking environment instance.
        policy: A function that takes the current state and returns an action.
        max_steps: Maximum steps for the episode.

    Returns:
        A list of PIL Image objects (frames).
    """
    frames = []
    
    # Reset the environment to get the initial state and frame
    observation, info = env.reset(seed = 7) 
    
    # Collect the first frame
    frame_array = env.render()
    if frame_array is not None:
        frames.append(Image.fromarray(frame_array))
    
    terminated = False
    truncated = False
    step = 0

    while not terminated and not truncated and step < max_steps:
        
        state_index = observation
        action = choose_action(policy, state_index)

        # 2. Take a step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 3. Render and collect the frame
        frame_array = env.render()
        if frame_array is not None:
            frames.append(Image.fromarray(frame_array))
        
        step += 1
        
    env.close() # Clean up resources
    return frames
def create_gif(frames, filename="policy_run.gif", duration=250):
    """
    Saves a list of PIL Image objects as an animated GIF.

    Args:
        frames: List of PIL Image objects.
        filename: Name for the output GIF file.
        duration: The duration (in milliseconds) for each frame. 
                  Since your render_fps is 4, 1/4 second = 250ms.
    """
    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duration,
            loop=0  # 0 means loop infinitely
        )
        print(f"GIF successfully saved as {filename}")
    else:
        print("No frames were collected to create the GIF.")


def evaluate(env, policy, n_episodes=100):
    episode_reward = [0 for _ in range(n_episodes)]
    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            action = choose_action(policy, state)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward[episode] += reward
            done = terminated or truncated
            steps += 1
            
    return episode_reward

def generate_gif():
    pass

def plot(arr, title="Rewards"):
    # Indices are simply the range from 0 to len(arr) - 1
    indices = np.arange(len(arr))
    plt.figure(figsize=(10, 4))
    plt.plot(indices, arr, marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid(True)
    # plt.show()

env = MultiGoalCliffWalkingEnv(render_mode="rgb_array")

# print("SARSA testing")
# Q_val, rewards, safe, risky = SARSA(env)
# print(safe, risky) # 0.5196, 0.051
# plot(rewards)
# episode_rewards = evaluate(env, find_policy_epsilon(Q_val, 0), 10)
# print(episode_rewards)
# print(np.array(episode_rewards).mean(), np.array(episode_rewards).std())
# create_gif(run_and_collect_frames(env, find_policy_epsilon(Q_val, 0)), filename="sarsa.gif")
# [170, 170, 170, 170, 168, 170, 170, 170, 170, 170]
# 169.8 0.6000000000000001

# print("Q learning testing")

# best_mean = -1e5
# best_alpha = 0.1
# best_safe, best_risky = 0, 0
# best_rewards = []
# best_Q = None
# best_episode_rewards = None
# for i in range(10):
#     ALPHA = 0.1 + i*0.05
#     Q_val, rewards, safe, risky = q_learning_for_cliff(env)
#     episode_rewards = evaluate(env, find_policy_epsilon(Q_val, 0), 10)
#     print(ALPHA, np.array(episode_rewards).mean())
#     if np.array(episode_rewards).mean() > best_mean:
#         best_mean = np.array(episode_rewards).mean()
#         best_alpha = ALPHA
#         best_safe, best_risky = safe, risky
#         best_rewards = rewards
#         best_Q = Q_val
#         best_episode_rewards = episode_rewards
        
# print(best_safe, best_risky) # 0.77112, 0.01694
# print(best_episode_rewards)  
# print(np.array(episode_rewards).mean(), np.array(episode_rewards).std())
# print(best_Q)
# print(best_mean, best_alpha)
# plot(best_rewards)
# create_gif(run_and_collect_frames(env, find_policy_epsilon(Q_val, 0)), filename="qlearning.gif")



# print("Expected SARSA testing")
# Q_val, rewards, safe, risky = expected_SARSA(env)
# print(safe, risky) # 0.5702, 0.023
# plot(rewards)
# episode_rewards = evaluate(env, find_policy_epsilon(Q_val, 0), 10)
# print(episode_rewards)
# print(np.array(episode_rewards).mean(), np.array(episode_rewards).std())
# create_gif(run_and_collect_frames(env, find_policy_epsilon(Q_val, 0)), filename="expected_sarsa.gif")
# [166, 170, 172, 172, 172, 172, 172, 168, 168, 172]
# 170.4 2.1540659228538015



# -------------------------------------------------------------------------------------------------------------------------

max_episodes = 300000 # optimal for MC = 200000 
seed = 509
gamma = 1 #optimal for both = 1
alpha = 0.2 # optimal for Qlearning
epsln = 0.5 # for MC(0,3) only, for (0,5) we get 0.66

def Find_policy_epsilon(Q_value, epsilon):
    num_states = Q_value.shape[0] 
    weights = np.array([2/6, 1/6, 3/6]) 
    weighted_epsilon_base = epsilon * weights
    policy_probs = np.tile(weighted_epsilon_base, (num_states, 1))
    optimal_policy = np.argmax(Q_value, axis=1)
    row_indices = np.arange(num_states)
    policy_probs[row_indices, optimal_policy] += 1.0 - epsilon
   
    return policy_probs

def monte_carlo(env):
    '''
    Implement the Monte Carlo algorithm to find the optimal policy for the given environment.
    Return Q table.
    return: Q table -> np.array of shape (num_states, num_actions)
    return: episode_rewards -> []
    return: _ 
    return: _ 
    '''
    episode_rewards = [0 for _ in range(max_episodes)]
    Q_value = np.full((env.observation_space.n, env.action_space.n), 1.0)
    visits = np.zeros(Q_value.shape)
    # terminal states to 0
    terminal_states = [(env.nrow-1)*env.ncol + i for i in range(env.ncol)]
    Q_value[terminal_states] = 0.0   
    goal_visit = 0
    term_dict = {}
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        history = []
        if epsln > 0.6:
            current_policy = find_policy_epsilon(Q_value, epsln) # 0.66 optimal for (0,5)
        else:
            current_policy = Find_policy_epsilon(Q_value, epsln)
        # current_policy = find_policy_epsilon(Q_value, 1/math.sqrt(math.sqrt(episode + 1)))
        # current_policy = find_policy_softmax(Q_value,1/(episode + 1))
        
        while True:
            action = choose_action(current_policy, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if next_state == env.state_to_index(env.goal_state):
                goal_visit += 1
            history.append((state, action, reward))
            state = next_state
            if terminated or truncated:
                if next_state not in term_dict:
                    term_dict[next_state] = 0
                term_dict[next_state] += 1
                break

        G = 0
        for state, action, reward in reversed(history):
            G = gamma * G + reward
            visits[state, action] += 1
            Q_value[state, action] += (G - Q_value[state, action]) / visits[state, action]

        episode_rewards[episode] = sum(r for _, _, r in history)


    
            

    return Q_value, episode_rewards
    
def q_learning_for_frozenlake(env):
    '''
    Implement the Q-learning algorithm to find the optimal policy for the given environment.
    return: Q table -> np.array of shape (num_states, num_actions)
    return episode_rewards_for_one_seed -> []
    '''
    episode_rewards = [0 for _ in range(max_episodes)]
    Q_value = np.full((env.observation_space.n, env.action_space.n), 1.0)
    visits = np.zeros(Q_value.shape)
    # terminal states to 0
    terminal_states = [(env.nrow-1)*env.ncol + i for i in range(env.ncol)]
    Q_value[terminal_states] = 0.0   
    goal_visit = 0
    term_dict = {}
    
    for episode in range(max_episodes):
        state, _ = env.reset(seed=seed)
        current_policy = find_policy_epsilon(Q_value, 1/math.sqrt(episode + 1))
        # current_policy = find_policy_softmax(Q_value,1/(episode + 1))
        while True:
            action = choose_action(current_policy, state)
            next_state, reward, terminated, truncated, info = env.step(action)
            if next_state == env.state_to_index(env.goal_state):
                goal_visit += 1
            Q_value[state, action] += alpha * (reward + gamma * np.max(Q_value[next_state]) - Q_value[state, action])
            episode_rewards[episode] += reward
            state = next_state
            if terminated or truncated:
                if next_state not in term_dict:
                    term_dict[next_state] = 0
                term_dict[next_state] += 1
                break


    
            

    return Q_value, episode_rewards

# env = DiagonalFrozenLake(render_mode="rgb_array", start_state=(0,3))

# Q_val, rewards= monte_carlo(env)
# # print(goal_visit)
# # print(term_dict)
# plot(rewards)
# episode_rewards = evaluate(env, find_policy_epsilon(Q_val, 0), 10)
# print(episode_rewards)
# print(np.array(episode_rewards).mean(), np.array(episode_rewards).std())
# create_gif(run_and_collect_frames(env, find_policy_epsilon(Q_val, 0)), filename="frozenlake_mc_(0,3).gif")

# for i in range(4):
#     epsln = 0.9 + i*0.3
#     Q_val, rewards, goal_visit, term_dict = monte_carlo(env)
#     print("Epsilon, goal visits", epsln, goal_visit)
#     episode_rewards = evaluate(env, find_policy_epsilon(Q_val, 0), 10)
#     print(episode_rewards)
#     print(np.array(episode_rewards).mean(), np.array(episode_rewards).std())
    