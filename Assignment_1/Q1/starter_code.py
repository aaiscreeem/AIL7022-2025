from env import FootballSkillsEnv
import numpy as np
def policy_iteration(envr , gamma = 0.95, tolerance = 1e-6, MAX_ITERATIONS = 1000):
    '''
    Implements the Policy Iteration algorithm to find the optimal policy for the 
    Football Skills Environment.
    
    Args:
        envr (class, optional): Environment class to instantiate. Defaults to FootballSkillsEnv.
    
    Returns:
        tuple: (optimal_policy, value_function, num_iterations)
            - optimal_policy (dict): Maps state indices to optimal actions
            - value_function (numpy.ndarray): Value of each state under optimal policy  
            - num_iterations (int): Number of iterations until convergence
    
    Algorithm:
    1. Initialize arbitrary policy and value function
    2. Policy Evaluation: Iteratively update value function until convergence
    3. Policy Improvement: Update policy greedily based on current values  
    4. Repeat steps 2-3 until policy converges
    
    Key Environment Methods to Use:
    - env.state_to_index(state_tuple): Converts (x, y, has_shot) tuple to integer index
    - env.index_to_state(index): Converts integer index back to (x, y, has_shot) tuple
    - env.get_transitions_at_time(state, action, time_step=None): gives list of tuples (probability, next state)
    - env._is_terminal(state): Check if state is terminal (has_shot=True)
    - env._get_reward(ball_pos, action, player_pos): Get reward for transition
    - env.reset(seed=None): Reset environment to initial state, returns (observation, info)
    - env.step(action): Execute action, returns (obs, reward, done, truncated, info)
    - env.get_obs(): Get current player position and has_shot flag
    - env.get_gif(policy, seed=20, filename="output.gif"): Generate GIF visualization 
      of policy execution from given seed
    
    Key Env Variables Notes:
    - env.observation_space.n: Total number of states (use env.grid_size^2 * 2)
    - env.action_space.n: Total number of actions (7 actions: 4 movement + 3 shooting)
    - env.grid_size: Total number of rows in the grid
    
    Remarks:
    The primary terminating condition is when the agent performs any of the three shooting actions.
    Once a shot is taken, the episode ends, and the final reward is calculated based on the ball’s landing position.
    Another way to end the episode is to hit an obstacle
    get_transitions_at_time gives next state as the ball position
    We want deterministic policy
    
    '''
    
    num_states = envr.grid_size**2 * 2
    num_actions = envr.action_space.n
    value_function = np.zeros(num_states) # improve initialisation
    policy = {} 
    for state in range(num_states):
        policy[state] = np.random.randint(num_actions) # need to improve this
    num_iter = 0
    calls = 0
    temp = 0
    i = 0
    for num_iter in range(MAX_ITERATIONS):    
        # Policy Evaluation with asynchronous updates for faster convergence
        for i in range(MAX_ITERATIONS):
            delta = 0
            for state in range(num_states):
                state_tuple = envr.index_to_state(state)
                if envr._is_terminal(state_tuple):
                    continue
                old_val = value_function[state]
                new_val = 0
                transitions = envr.get_transitions_at_time(state_tuple, policy[state])
                calls += 1
                for prob, next_state in transitions:
                    # need to verify this step whether get_reward works this way
                    reward = envr._get_reward(next_state[:2], policy[state], state_tuple[:2])
                    if envr._is_terminal(next_state):
                        new_val += prob * reward
                    else:
                        new_val += prob * ( reward + gamma * value_function[envr.state_to_index(next_state)])
                value_function[state] = new_val
                delta = max(delta, abs(old_val - new_val))
            if delta < tolerance:
                break      
                    
        # Policy Improvement
        policy_stable = True
        old_policy = policy.copy()
        for state in range(num_states):
            state_tuple = envr.index_to_state(state)
            if envr._is_terminal(state_tuple):
                    continue
            best_action = None
            best_qval = -1e9
            for action in range(num_actions):
                qval = 0
                transitions = envr.get_transitions_at_time(state_tuple, action)
                calls += 1
                for prob, next_state in transitions:
                    reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                    if envr._is_terminal(next_state):
                        qval += prob * reward
                    else:
                        qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)])
                if qval > best_qval:
                    best_action = action
                    best_qval = qval
            policy[state] = best_action 
            if policy[state] != old_policy[state]:
                policy_stable = False
        

        
        if policy_stable:
            break
         
            
    return policy, value_function, num_iter, calls
        

def value_iteration(envr, gamma = 0.95, tolerance = 1e-6, MAX_ITERATIONS = 1000):
    
    num_states = envr.grid_size**2 * 2
    num_actions = envr.action_space.n
    value_function = np.zeros(num_states) # improve initialisation
    policy = np.zeros(num_states)
    calls = 0
    # Asynchronous updates for faster convergence
    for num_iter in range(MAX_ITERATIONS):
        delta = 0
        for state in range(num_states):
            state_tuple = envr.index_to_state(state)
            if envr._is_terminal(state_tuple):
                continue
            old_val = value_function[state]
            new_val = -1e9
            for action in range(num_actions):
                qval = 0
                transitions = envr.get_transitions_at_time(state_tuple, action)
                calls += 1
                for prob, next_state in transitions:
                    reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                    if envr._is_terminal(next_state):
                        qval += prob * reward
                    else:
                        qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)])
                if qval > new_val:
                    new_val = qval
            value_function[state] = new_val
            delta = max(delta, abs(old_val - new_val))
        if delta < tolerance:
            break 
        
    # Final Policy
    for state in range(num_states):
        state_tuple = envr.index_to_state(state)
        if envr._is_terminal(state_tuple):
                continue
        best_action = None
        best_qval = -1e9
        for action in range(num_actions):
            qval = 0
            transitions = envr.get_transitions_at_time(state_tuple, action)
            calls += 1
            for prob, next_state in transitions:
                reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                if envr._is_terminal(next_state):
                    qval += prob * reward
                else:
                    qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)])
            if qval > best_qval:
                best_action = action
                best_qval = qval
        policy[state] = best_action 
    return policy, value_function, num_iter+1, calls

def check_same_policy(policy1, policy2):
    check = True
    for index in range(800):
        if ((index % 2) == 0):
            if (policy1[index] != policy2[index]):
                check = False
    return check

def eval_policy( policy):
    env = FootballSkillsEnv(render_mode = 'gif')
    rewards = np.zeros(20)
    for i in range (20):
        _, reward = env.get_gif(policy, seed = i, filename="output.gif")
        rewards[i] = reward
    return np.mean(rewards), np.std(rewards)



def non_stationary(envr, gamma = 0.95, tolerance = 1e-6, MAX_ITERATIONS = 1000, horizon = 40):
    
    num_states = envr.grid_size**2 * 2
    num_actions = envr.action_space.n
    value_function = np.zeros((num_states,horizon+1)) # improve initialisation
    policy = np.zeros((horizon+1, num_states))
    calls = 0
    num_iter = 0
    # Asynchronous updates for faster convergence
    for num_iter in range(MAX_ITERATIONS):
        delta = 0
        # go reverse in time because t+1 needed for t
        for t in range(horizon-1, -1,-1):
            for state in range(num_states):
                state_tuple = envr.index_to_state(state)
                if envr._is_terminal(state_tuple):
                    continue
                old_val = value_function[state][t]
                new_val = -1e9
                for action in range(num_actions):
                    qval = 0
                    transitions = envr.get_transitions_at_time(state_tuple, action,t)
                    calls += 1
                    for prob, next_state in transitions:
                        reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                        if envr._is_terminal(next_state):
                            qval += prob * reward
                        else:
                            qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)][t+1])
                    if qval > new_val:
                        new_val = qval
                value_function[state][t] = new_val
                delta = max(delta, abs(old_val - new_val))
        if delta < tolerance:
            break 
        
    # Final Policy
    for state in range(num_states):
        state_tuple = envr.index_to_state(state)
        if envr._is_terminal(state_tuple):
                continue
        for t in range(horizon):
            best_action = None
            best_qval = -1e9
            for action in range(num_actions):
                qval = 0
                transitions = envr.get_transitions_at_time(state_tuple, action, t)
                calls += 1
                for prob, next_state in transitions:
                    reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                    if envr._is_terminal(next_state):
                        qval += prob * reward
                    else:
                        qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)][t+1])
                if qval > best_qval:
                    best_action = action
                    best_qval = qval
            policy[t][state] = best_action 
    return policy, value_function, num_iter+1, calls

# env_degraded = FootballSkillsEnv(render_mode='gif', degrade_pitch=True)
# policy, value_function, num_iter, calls = non_stationary(env_degraded)
# env_degraded.get_gif(policy, seed=20, filename="output_non.gif")
# print(f"Number of iterations: {num_iter+1}")
# print(f"Number of calls to get_transitions_at_time: {calls}") # 336000
# policy3, value_function, num_iter, calls = value_iteration(env_degraded)
# time_policy = np.zeros((41, len(policy3)))
# for i in range(40):
#     for j in range(len(policy3)):
#         time_policy[i][j] = policy3[j]
# env_degraded.get_gif(time_policy, seed=20, filename="output_vi.gif")
# print(f"Number of iterations: {num_iter+1}")
# print(f"Number of calls to get_transitions_at_time: {calls}")

def eval_policy_non( policy):
    env = FootballSkillsEnv(render_mode = 'gif', degrade_pitch=True)
    rewards = np.zeros(20)
    for i in range (20):
        _, reward = env.get_gif(policy, seed = i, filename="output.gif")
        rewards[i] = reward
    return np.mean(rewards), np.std(rewards)

# print(eval_policy_non(policy)) # 37.7, 39.6
# print(eval_policy_non(time_policy)) # mean 23.65, std 44.71
    
import heapq

def modified_value_iteration(envr, gamma = 0.95, tolerance = 1e-6, MAX_ITERATIONS = 1000):
    
    num_states = envr.grid_size**2 * 2
    num_actions = envr.action_space.n
    value_function = np.zeros(num_states) 
    policy = np.zeros(num_states)
    calls = 0

    heap = []
    initial_priority = 1e9
    for state in range(num_states):
        state_tuple = envr.index_to_state(state)
        if envr._is_terminal(state_tuple):
            continue
        heapq.heappush(heap, (-initial_priority, state))

    for num_iter in range(MAX_ITERATIONS):
        # process until top priority < tolerance (or heap empty)
        while heap:
            top_neg_prio, top_state = heap[0]  # peek
            top_prio = -top_neg_prio
            if top_prio < tolerance:
                break

            heapq.heappop(heap)
            state = top_state
            state_tuple = envr.index_to_state(state)
            if envr._is_terminal(state_tuple):
                continue

            old_val = value_function[state]
            new_val = -1e9
            for action in range(num_actions):
                qval = 0
                transitions = envr.get_transitions_at_time(state_tuple, action)
                calls += 1
                for prob, next_state in transitions:
                    reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                    if envr._is_terminal(next_state):
                        qval += prob * reward
                    else:
                        qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)])
                if qval > new_val:
                    new_val = qval

            value_function[state] = new_val
            new_priority = abs(new_val - old_val)

            heapq.heappush(heap, (-new_priority, state))

        # termination condition for outer loop: if queue empty or top priority small
        if not heap:
            break
        if -heap[0][0] < tolerance:
            break

    # Final Policy
    for state in range(num_states):
        state_tuple = envr.index_to_state(state)
        if envr._is_terminal(state_tuple):
                continue
        best_action = None
        best_qval = -1e9
        for action in range(num_actions):
            qval = 0
            transitions = envr.get_transitions_at_time(state_tuple, action)
            calls += 1
            for prob, next_state in transitions:
                reward = envr._get_reward(next_state[:2], action, state_tuple[:2])
                if envr._is_terminal(next_state):
                    qval += prob * reward
                else:
                    qval += prob * (reward + gamma * value_function[envr.state_to_index(next_state)])
            if qval > best_qval:
                best_action = action
                best_qval = qval
        policy[state] = best_action 
    return policy, value_function, num_iter+1, calls


# env_normal = FootballSkillsEnv(render_mode = 'gif')
# policy1, value_function, num_iter, calls = policy_iteration(env_normal, gamma = 0.3)
# env_normal.get_gif(policy1, seed=20, filename="output1.gif")
# policy2, value_function, num_iter, calls = value_iteration(env_normal)
# env_normal.get_gif(policy2, seed=20, filename="output.gif")

# print(check_same_policy(policy1, policy2)) # True

# print(eval_policy(policy1)) 
# print(eval_policy(policy2)) 

# matrix = np.zeros((20,20))
# for index in range(800):
#     i,j,k = env_normal.index_to_state(index)
#     if(k == 0):
#         matrix[i][j] = policy2[index]
# print(matrix)
# print(f"Number of iterations: {num_iter+1}")
# print(f"Number of calls to get_transitions_at_time: {calls}")
              
            
# policy iteration: 
#   gamma = 0.95: iterations = 20, calls = 290400, mean reward = 47, std = 21.8
#   gamma = 0.5: iterations = 25, calls = 110000, mean reward = 32, std = 57.24
#   gamma = 0.3: iterations = 23, calls = 91200, mean reward = 32, std = 57.24
# value iteration:
#   gamma = 0.95: iterations = 31, calls = 86800, mean reward = 47, std = 21.8
#   gamma = 0.5: iterations = 27, calls = 75600, mean reward = 32, std = 57.24
#   gamma = 0.3: iterations = 17, calls = 47600, mean reward = 32, std = 57.24
# policy same at gamma = 0.5, 0.95 but different at gamma = 0.3
# modified value iteration: calls = 30219 and same policy


                 
    
