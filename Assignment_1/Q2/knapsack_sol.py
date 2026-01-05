import numpy as np
from or_gym.envs.classic_or.knapsack import OnlineKnapsackEnv
import matplotlib.pyplot as plt
from scipy.stats import mode
import numpy as np
import imageio
import io

# Methods of OnlineKnapsackEnv:
# _RESET(self)
# _STEP(self, action)
# _update_state(self) returns self.state, reward, done, _
        # self.N = 200
        # self.max_weight = 200
        # self.current_weight = 0
        # self._max_reward = 10000
        # self.mask = True
        # self.seed = 0
        # self.item_numbers = np.arange(self.N)
        # self.item_weights = np.random.randint(1, 100, size=self.N)
        # self.item_values = np.random.randint(0, 100, size=self.N)
        # self.over_packed_penalty = 0
        # self.randomize_params_on_reset = False
        # self._collected_items.clear()
        # self.action_space = spaces.Discrete(2)

        # obs_space = spaces.Box(0, self.max_weight, shape=(4,), dtype=np.int32)
        # if self.mask:
        #     self.observation_space = spaces.Dict({
        #         'state': obs_space,
        #         'avail_actions': spaces.Box(0, 1, shape=(2,), dtype=np.uint8),
        #         'action_mask': spaces.Box(0, 1, shape=(2,), dtype=np.uint8)
        #     })
        # else:
        #     self.observation_space = obs_space
        # self.item_limits_init = np.random.randint(1e9, 1e9, size=self.N, dtype=np.int32) # No limit
# self.step_counter = 0
# self.step_limit = 50        
# self.state = self.reset()
# self._max_reward = 600
# State: (weight, item, timestep)
# uniform probability of going into the next state
    # Episode Termination:
    #     Full knapsack, selection that puts the knapsack over the limit, or
    #     the number of items to be drawn has been reached.
class ValueIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_iterations = 50
        self.value_function = np.zeros((env.N, env.max_weight + 1, env.step_limit + 1))
        self.policy = np.zeros((env.N, env.max_weight + 1, env.step_limit + 1))
        pass


    def get_reward_and_done(self, current_weight, item_idx, action, time):
        # returns expected reward i.e. summation probability*(reward + gamma * value_function(next_state))
        done = False
        reward = 0
        if action == 0:
            if current_weight == self.env.max_weight:
                done = True
            else:
                reward = self.gamma * np.mean(self.value_function[:][current_weight][time + 1])
            
        else:
            # Check that item will fit
            if self.env.item_weights[item_idx] + current_weight <= self.env.max_weight:
                current_weight += self.env.item_weights[item_idx]
                reward = self.env.item_values[item_idx]
                if current_weight == self.env.max_weight:
                    done = True
                else:
                    reward += self.gamma * np.mean(self.value_function[:][current_weight][time + 1])
                
            else:
                # End if over weight
                reward = 0
                done = True
        return reward, done
        

    def value_iteration(self):
        for i in range(self.max_iterations):
            delta = 0
            for t in range(self.env.step_limit, -1, -1):
                for weight in range(self.env.max_weight + 1):
                    for item in range(self.env.N):
                        old_val = self.value_function[item][weight][t]
                        #Terminal State
                        if t == self.env.step_limit:
                            self.value_function[item][weight][t] = 0
                        else:
                            max_val = -1e9
                            best_action = -1
                            for action in range(2):
                                qval, done = self.get_reward_and_done(weight, item, action,t)
                                if qval > max_val:
                                    max_val = qval
                                    best_action = action
                            self.value_function[item][weight][t] = max_val
                            self.policy[item][weight][t] = best_action
                        delta = max(delta, abs(old_val - self.value_function[item][weight][t]))
            print(f"Iteration: {i}, delta: {delta}")
            if delta < self.epsilon:
                break 

        return self.policy, self.value_function
        
                                      

    def get_action(self, state):
        # assume time dependent state is (item, weight, time)
        return self.policy[state[0]][state[1]][state[2]]
    

class PolicyIterationOnlineKnapsack:
    def __init__(self, env, gamma=0.95, epsilon=1e-4, eval_iterations=10, max_iterations=10):
        
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon # pointless
        self.eval_iterations = eval_iterations
        self.max_iterations = max_iterations
        # State: (item_idx, current_weight, time_step)
        self.value_function = np.zeros((self.env.N, self.env.max_weight + 1, self.env.step_limit + 1))
        
        # Initialize a policy to always reject 
        self.policy = np.zeros((self.env.N, self.env.max_weight + 1, self.env.step_limit + 1))

    def _calculate_q_value(self, item_idx, weight, t, action):
        if t >= self.env.step_limit -1:
            if action == 1 and weight + self.env.item_weights[item_idx] <= self.env.max_weight:
                return self.env.item_values[item_idx]
            else:
                return 0

        if action == 0:
            reward = 0
            next_weight = weight
            if next_weight == self.env.max_weight:
                return reward
            return reward + self.gamma * np.mean(self.value_function[:][next_weight][t + 1])

        elif action == 1:
            item_w = self.env.item_weights[item_idx]
            if weight + item_w > self.env.max_weight:
                return 0
            
            reward = self.env.item_values[item_idx]
            next_weight = weight + item_w
            if next_weight == self.env.max_weight:
                return reward
            return reward + self.gamma * np.mean(self.value_function[:][next_weight][t + 1])
        
        return 0

    def policy_evaluation(self):
        
        for i in range(self.eval_iterations):
            for t in range(self.env.step_limit - 1, -1, -1):
                for item_idx in range(self.env.N):
                    for weight in range(self.env.max_weight + 1):
                        action = self.policy[item_idx, weight, t]
                        self.value_function[item_idx, weight, t] = self._calculate_q_value(item_idx, weight, t, action)
        
        

    def policy_improvement(self):
        
        policy_stable = True
        for t in range(self.env.step_limit - 1, -1, -1):
            for item_idx in range(self.env.N):
                for weight in range(self.env.max_weight + 1):
                    old_action = self.policy[item_idx, weight, t]
                    
                    q_reject = self._calculate_q_value(item_idx, weight, t, 0)
                    q_accept = self._calculate_q_value(item_idx, weight, t, 1)
                    
                    best_action = 1 if q_accept > q_reject else 0
                    
                    self.policy[item_idx, weight, t] = best_action
                    
                    if old_action != best_action:
                        policy_stable = False
                        
        return policy_stable

    def run_policy_iteration(self):
        i = 0
        for i in range(self.max_iterations):
            self.policy_evaluation()
            is_stable = self.policy_improvement()
            
            print(f"Iteration {i+1} completed.")
            
            if is_stable:
                print("Iterations for convergence", i+1)
                break
        if i == self.max_iterations: 
             print("No convergence.")

        return self.policy, self.value_function


def evaluate(policy, seed):

    env = OnlineKnapsackEnv()
    state = env._RESET()
    env.seed = seed
    
    total_reward = 0
    done = False
    
    timesteps = [0]
    knapsack_values = [0]
    print(f"Max Weight: {env.max_weight} | Episode Length: {env.step_limit} steps")
    print("-" * 50)

    while not done:

        item_idx = env.current_item
        current_weight = env.current_weight
        time_step = env.step_counter

        # Choose an action from the pre-computed policy table
        action = policy[item_idx, current_weight, time_step]
        
        # Take a step in the environment using the chosen action
        next_state, reward, done, info = env._STEP(action)
        
        # Update the total reward
        total_reward += reward

        # Log data for plotting
        timesteps.append(env.step_counter)
        knapsack_values.append(total_reward)

        # Print a detailed log of the current step
        item_w = env.item_weights[item_idx]
        item_v = env.item_values[item_idx]
        print(f"Step {time_step:02d}: Item(W:{item_w:2d}, V:{item_v:2d}) -> "
              f"Action: {'Accept' if action == 1 else 'Reject'} -> "
              f"Reward: {reward:2d} | Knapsack Value: {total_reward:3d}")
    
    print("-" * 50)
    print(f"Episode finished.")
    print(f"Final Knapsack Value: {total_reward}")
    print(f"Final Knapsack Weight: {env.current_weight}/{env.max_weight}")
    print("-" * 50)

    # 4. Generate the plot
    plt.figure(figsize=(12, 7))
    plt.plot(timesteps, knapsack_values, marker='o', linestyle='-', color='b', label='Knapsack Value')
    plt.step(timesteps, knapsack_values, where='post', linestyle='--', color='grey')
    
    plt.title(f'Knapsack Value During Simulation (Seed: {seed})', fontsize=16)
    plt.xlabel('Timestep / Item Presented', fontsize=12)
    plt.ylabel('Cumulative Knapsack Value', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.xticks(np.arange(0, env.step_limit + 1, step=max(1, env.step_limit // 10)))
    plt.tight_layout()
    plt.show()
    
    return total_reward

    
def generate_value_function_gif(value_function, env, filename="output.gif", fps=3):
    
    # --- 1. Sort items by weight for consistent visualization ---
    item_weights = env.item_weights
    item_values = env.item_values
    sorted_indices = np.argsort(item_weights)
    
    sorted_weights = item_weights[sorted_indices]
    sorted_values = item_values[sorted_indices]
    # Reorder the value function according to the sorted item weights
    sorted_vf = value_function[sorted_indices, :, :]

    frames = []
    # --- 2. Generate a plot for each timestep ---
    for t in range(env.step_limit):
        fig, ax = plt.subplots(figsize=(14, 8), dpi=100)
        
        # We transpose the slice so weight is on the Y-axis
        vf_slice_at_t = sorted_vf[:, :, t].T
        
        im = ax.imshow(
            vf_slice_at_t, 
            aspect='auto', 
            cmap='viridis', 
            origin='lower',
            extent=[-0.5, env.N - 0.5, -0.5, env.max_weight + 0.5]
        )

        # --- 3. Configure plot aesthetics (labels, titles, ticks) ---
        ax.set_title(f"Value Function at Timestep t={t}/{env.step_limit-1}", fontsize=16)
        ax.set_ylabel("Current Knapsack Weight", fontsize=12)
        ax.set_xlabel("Items (Sorted by Weight)", fontsize=12)

        # Configure x-axis ticks to show item weight and value
        num_items = env.N
        labels = [f"W:{w}\nV:{v}" for w, v in zip(sorted_weights, sorted_values)]
        
        # Smartly adjust number of labels to prevent clutter
        if num_items > 20:
            tick_count = 15
            tick_indices = np.linspace(0, num_items - 1, tick_count, dtype=int)
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([labels[i] for i in tick_indices], rotation=45, ha="right", fontsize=9)
        else:
            ax.set_xticks(np.arange(num_items))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        # Add a color bar to show the value scale
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Expected Future Reward (Value)", fontsize=12)
        
        fig.tight_layout()

        # --- 4. Save the plot to a memory buffer ---
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close(fig) # Close the figure to free memory
        
        print(f"  - Frame {t+1}/{env.step_limit} generated.")

    # --- 5. Stitch all frames into a GIF ---
    imageio.mimsave(filename, frames, fps=fps)
    print(f"\nSuccessfully saved GIF to '{filename}'")


if __name__=="__main__":
    env=OnlineKnapsackEnv()
    
    pi = PolicyIterationOnlineKnapsack(env)
    policy, value_function = pi.run_policy_iteration() 
    generate_value_function_gif(value_function, env, filename="output_pi.gif")
    for i in range(5):
        evaluate(policy, seed=i)
    
    # vi = ValueIterationOnlineKnapsack(env)
    # policy, value_function = vi.value_iteration()
    # generate_value_function_gif(value_function, env, filename="output_vi_10.gif")
    # for i in range(5):
    #     evaluate(policy, seed=i)
    
    
    

