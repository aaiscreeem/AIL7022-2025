import gymnasium
import gym_gridworlds
import numpy as np
import random
import imageio
import os
import matplotlib.pyplot as plt 
from collections import defaultdict
from gymnasium.envs.registration import register
from behaviour_policies import create_behaviour
import math
import os, json
import matplotlib.pyplot as plt

GRID_ROWS = 4
GRID_COLS = 5
NOISE = 0.01

# Register the custom environment
register(
    id="Gym-Gridworlds/Full-4x5-v0",
    entry_point="gym_gridworlds.gridworld:Gridworld",
    max_episode_steps=500,
    kwargs={"grid": "4x5_full"},
)

def set_global_seed(seed: int):
    """Set seed for reproducibility across modules."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def state_to_cord(state):
    """Convert state number to (row, col) coordinates."""
    return divmod(state, GRID_COLS)

def cord_to_state(row, col):
    """Convert (row, col) coordinates to state number."""
    return row * GRID_COLS + col

def find_policy_epsilon(Q_value, epsilon):
    
    base = np.full(Q_value.shape, epsilon/Q_value.shape[1])
    optimal_policy = (Q_value == Q_value.max(axis=1, keepdims=True)).argmax(axis=1)
    # optimal_policy = np.argmax(Q_value, axis=1)
    row_indices = np.arange(Q_value.shape[0])
    base[row_indices, optimal_policy] += 1 - epsilon
    
    return base

def monte_carlo_off_policy_control(env, num_episodes = 500, seed = 509, gamma = 0.85, epsilon = 0.05):
    
    Q = np.zeros((env.observation_space.n, env.action_space.n)) 
    C = np.zeros((env.observation_space.n, env.action_space.n))
    behav_policy = create_behaviour(NOISE)
    episode_rewards = np.zeros(num_episodes)

    def behavior_policy(state):
        """
        Returns:
            action: Action to take (integer)
        """
        return np.random.choice(np.arange(behav_policy.shape[1]), p=behav_policy[state])
        

    def get_behavior_prob(state, action):
        """TODO: Get the probability of taking action in state under behavior policy."""
        return behav_policy[state][action]

    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed) 
        history = []  
        while True:
            action = behavior_policy(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_rewards[episode] += reward
            history.append((state, action, reward))
            state = next_state
            if terminated or truncated:
                break
        epsilon_prime = epsilon * max(0.1,1 - (episode / num_episodes))
        pi = find_policy_epsilon(Q, epsilon_prime)
        G = 0.0
        W = 1.0
        for state, action, reward in reversed(history):
            G = gamma * G + reward
            C[state, action] += W
            Q[state, action] += W *(G - Q[state, action]) / C[state, action]
            W *= pi[state,action] / get_behavior_prob(state, action)
            if W < 1e-9 or W > 1e9:
                break
    
    final_policy = (Q == Q.max(axis=1, keepdims=True)).argmax(axis=1)
    # final_policy = np.argmax(Q, axis=1)                    
    return Q, final_policy, episode_rewards
    def get_target_prob(state, action):
        """
        TODO: Get the probability of taking action in state under target policy
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            probability
        """
        return 1 if Q[state][action] == np.max(Q[state]) else 0

    def target_policy(state):
        """
        TODO: Implement the target policy 
        
        Args:
            state: Current state
            
        Returns:
            action: Action to take (integer)
        """
        return np.argmax(Q[state])

def evaluate_policy(env, policy, n_episodes=100, max_steps=500):
    """
    Evaluate a given policy by running it for multiple episodes.
    
    Args:
        env: The environment
        policy: Policy to evaluate (array of actions for each state)
        n_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        
    Returns:
        tuple: (mean_reward, min_reward, max_reward, std_reward, success_rate)
    """
    rewards = []
    success_rate = 0

    for episode in range(n_episodes):
        state, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0
        steps = 0
        
        while not done and steps < max_steps:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
            
        rewards.append(episode_reward)
        
        # Consider episode successful if reward > 0.5
        if episode_reward > 0.5:
            success_rate += 1

    return (np.mean(rewards), np.min(rewards), np.max(rewards), 
            np.std(rewards), success_rate / n_episodes)

def generate_policy_gif(env, policy, noise, max_steps=500, fps=4):
    """
    Run a single episode using a deterministic policy (array of actions per state)
    and save RGB frames as a GIF in the gifs folder.
    """

    # Ensure the gifs directory exists
    gifs_dir = os.path.join(os.path.dirname(__file__), "gifs")
    os.makedirs(gifs_dir, exist_ok=True)

    filename = os.path.join(gifs_dir, f"monte_carlo_gif_({noise}).gif")

    frames = []
    print(f"\nGenerating GIF... saving to {filename}")

    # Create a fresh renderable env instance
    env_render = gymnasium.make(env.spec.id, render_mode='rgb_array', random_action_prob=0.1)
    state, _ = env_render.reset()
    done = False
    steps = 0

    # Capture initial frame
    frames.append(env_render.render())

    while not done and steps < max_steps:
        action = int(policy[state])
        state, reward, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        steps += 1
        frames.append(env_render.render())

    env_render.close()

    imageio.mimsave(filename, frames, fps=fps)
    print(f"GIF saved successfully to {os.path.abspath(filename)}")


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

if __name__ == '__main__':
    

    # ensure folders exist next to this file
    base_dir = os.path.dirname(__file__) if "__file__" in globals() else "."
    plots_dir = os.path.join(base_dir, "plots")
    gifs_dir = os.path.join(base_dir, "gifs")
    eval_dir = os.path.join(base_dir, "evaluation")
    for d in (plots_dir, gifs_dir, eval_dir):
        os.makedirs(d, exist_ok=True)

    for noise in [0.0, 0.1, 0.01]:
        # Training parameters
        NOISE = noise
        num_seeds = 10
        num_episodes = 20000  # TODO:CHANGE TO SUITABLE VALUE 
        best_policy = None
        best_q_values = None
        best_reward = -1e6
        best_std = 0

        print(f"--- Starting training across {num_seeds} seeds (NOISE={NOISE}) ---")

        mean_rewards = np.zeros(num_episodes)
        for seed in range(num_seeds):
            env = gymnasium.make('Gym-Gridworlds/Full-4x5-v0', random_action_prob=0.1)
            env.action_space.seed(seed) 
            env.observation_space.seed(seed) 
            env.np_random,_ = gymnasium.utils.seeding.np_random(seed)
            print(f"\n--- Training Seed {seed + 1}/{num_seeds} ---")
            set_global_seed(seed)

            # Train the policy
            q_values, policy, episode_rewards = monte_carlo_off_policy_control(env, num_episodes, seed)
            mean_rewards += episode_rewards

            # Evaluate the trained policy
            mean_reward, min_reward, max_reward, std_reward, success_rate = evaluate_policy(env, policy)

            print(f"\nResults: Mean={mean_reward:.3f}, Min={min_reward:.3f}, "
                  f"Max={max_reward:.3f}, Std={std_reward:.3f}, Success Rate={success_rate:.3f}")

            if mean_reward > best_reward:
                best_reward = mean_reward
                best_policy = policy
                best_q_values = q_values
                best_std = std_reward
                

        mean_rewards /= num_seeds

        # produce and save plot (filename includes NOISE)
        plot(mean_rewards, title="Rewards vs Episode")
        plot_fname = os.path.join(plots_dir, f"monte_carlo_reward_curve_({NOISE}).png")
        try:
            plt.savefig(plot_fname)
        except Exception:
            # some plot() implementations create a new figure or close it; be robust
            plt.figure(figsize=(8,4))
            plt.plot(mean_rewards)
            plt.title("Rewards vs Episode")
            plt.tight_layout()
            plt.savefig(plot_fname)
        plt.close()
        print("Saved plot to", plot_fname)

        action_map = {0: 'Left', 1: 'Down', 2: 'Right', 3: 'Up', 4: 'Stay'}

        print("\nBest Optimal Policy (Action to take in each state):")
        if best_policy is not None:
            policy_grid = np.array([action_map[i] for i in best_policy]).reshape(GRID_ROWS, GRID_COLS)
            print(policy_grid)

            print("\nQ-values for State 0 (top-left) from the best policy:")
            if 0 in best_q_values:
                for action, value in enumerate(best_q_values[0]):
                    print(f"  Action: {action_map[action]}, Q-value: {value:.3f}")

            # existing gif generation (keeps original behaviour)
            generate_policy_gif(env, best_policy, noise=NOISE)
        else:
            print("No successful policy was trained.")

        env.close()

        # write mean & std to JSON in evaluation folder under requested key
        mean_val = float(np.mean(mean_rewards))
        std_val = float(np.std(mean_rewards))
        json_path = os.path.join(eval_dir, "importance_sampling_evaluation_results.json")
        key = f"MC_ImportanceSampling({NOISE})"
        entry = {"mean": best_reward, "std": best_std}

        # load existing, update, save
        existing = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as jf:
                    existing = json.load(jf)
            except Exception:
                existing = {}
        existing[key] = entry
        with open(json_path, "w") as jf:
            json.dump(existing, jf, indent=4)

        print("Saved evaluation JSON to", json_path)

