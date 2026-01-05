import numpy as np
import itertools
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import mode
import sys
import copy
import time
import random
import math


random.seed(42)       # Set seed for Python's built-in random module
np.random.seed(42) 
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv

class Solver:
    def __init__(self, envr, gamma = 0.999, max_cash = 100, max_price = 100, price_std = 3, max_jump = 10, max_iterations = 1000):
        self.envr = envr
        self.gamma = gamma
        self.max_cash = max_cash
        self.max_price = max_price
        self.std = price_std
        self.max_jump = max_jump
        self.max_iterations = max_iterations

        self.max_holdings = int(envr.holding_limit[0])
        self.lot_size = int(getattr(envr, 'lot_size', 2))

        self.actions = [-2, -1, 0, 1, 2]

        self.num_steps = int(getattr(envr, 'step_limit'))

        self.value_function = np.zeros((self.num_steps + 1,
                                        self.max_holdings + 1,
                                        self.max_cash + 1,
                                        self.max_price + 1), dtype=float)

        # actions at times 0..T-1
        self.policy = np.zeros((self.num_steps,
                                self.max_holdings + 1,
                                self.max_cash + 1,
                                self.max_price + 1), dtype=int)

        self.price_pmf = self._build_price_pmfs()

        # initialize terminal layer and heuristic earlier
        self._initialize_values()

    def _initialize_values(self):
        T = self.num_steps
        # Terminal exact wealth
        for h in range(self.max_holdings + 1):
            for c in range(self.max_cash + 1):
                for p in range(self.max_price + 1):
                    self.value_function[T, h, c, p] = float(c + h * p)
        # Heuristic for earlier times
        for t in range(T+1):
            for h in range(self.max_holdings + 1):
                for c in range(self.max_cash + 1):
                    for p in range(self.max_price + 1):
                        self.value_function[t, h, c, p] = float(c + h * 5)

    def _build_price_pmfs(self):
        pmfs = dict()
        for p in range(self.max_price + 1):
            low = max(0, p - self.max_jump)
            high = min(self.max_price, p + self.max_jump)
            candidates = np.arange(low, high + 1)
            if len(candidates) == 0:
                pmfs[p] = [(p, 1.0)]
                continue
            diffs = (candidates - p) / float(self.std)
            weights = np.exp(-0.5 * (diffs ** 2))
            sumw = weights.sum()
            if sumw == 0:
                probs = np.repeat(1.0 / len(candidates), len(candidates))
            else:
                probs = weights / sumw
            pmfs[p] = [(int(candidates[i]), float(probs[i])) for i in range(len(candidates))]
        return pmfs

    def _clip_cash(self, c: int) -> int:
        return max(0, min(self.max_cash, int(c)))

    def _clip_holdings(self, h: int) -> int:
        return max(0, min(self.max_holdings, int(h)))

    def _legal_actions(self, holdings, cash, price):
        legal = []
        buy_cost_per_unit = float(getattr(self.envr, 'buy_cost')[0])
        sell_cost_per_unit = float(getattr(self.envr, 'sell_cost')[0])

        for lots in self.actions:
            units = int(lots * self.lot_size)
            if units < 0:
                qty_sell = -units
                if qty_sell > holdings:
                    continue
                new_cash = cash + qty_sell * price - sell_cost_per_unit * qty_sell
                if new_cash < 0:
                    continue
                if holdings - qty_sell < 0:
                    continue
                legal.append(lots)
            elif units > 0:
                qty_buy = units
                total_cost = qty_buy * price + buy_cost_per_unit * qty_buy
                if total_cost > cash:
                    continue
                if holdings + qty_buy > self.max_holdings:
                    continue
                legal.append(lots)
            else:
                legal.append(lots)
        return legal

    def _apply_action(self, holdings, cash, price, lots):
        buy_cost_per_unit = int(getattr(self.envr, 'buy_cost')[0])
        sell_cost_per_unit = int(getattr(self.envr, 'sell_cost')[0])

        units = int(lots * self.lot_size)
        if units > 0:
            transaction_cost = buy_cost_per_unit * units
            new_holdings = holdings + units
            new_cash = cash - units * price - transaction_cost
        elif units < 0:
            qty_sell = -units
            transaction_cost = sell_cost_per_unit * qty_sell
            new_holdings = holdings - qty_sell
            new_cash = cash + qty_sell * price - transaction_cost
        else:
            transaction_cost = 0
            new_holdings = holdings
            new_cash = cash
        return int(new_holdings), int(new_cash), int(transaction_cost)

    def _expected_backup(self, t, holdings, cash, price):
        if t >= self.num_steps:
            return self.value_function[t, holdings, cash, price], 0
        best_q = -math.inf
        best_action = 0
        legal = self._legal_actions(holdings, cash, price)
        if len(legal) == 0:
            return self.value_function[t, holdings, cash, price], 0
        pmf = self.price_pmf[price]
        for lots in legal:
            new_h, new_c, trans_cost = self._apply_action(holdings, cash, price, lots)
            new_c_idx = self._clip_cash(new_c)
            new_h_idx = self._clip_holdings(new_h)
            immediate = -trans_cost
            cont = 0.0
            for pprime, prob in pmf:
                cont += prob * self.value_function[t+1, new_h_idx, new_c_idx, pprime]
            q = immediate + self.gamma * cont
            if q > best_q:
                best_q = q
                best_action = lots
        return best_q, best_action

    def value_iteration(self, tolerance = 1e-3, verbose = False):
        T = self.num_steps
        iterations = 0
        calls = 0
        for it in range(self.max_iterations):
            iterations += 1
            # start a trial by resetting env
            reset_ret = self.envr.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            cash = int(round(obs[0])); price = int(round(obs[1])); holdings = int(round(obs[2]))
            cash = self._clip_cash(cash)
            price = max(0, min(self.max_price, price))
            holdings = self._clip_holdings(holdings)

            t = 0
            max_update = 0.0
            done = False

            while not done and t < T:
                old_v = self.value_function[t, holdings, cash, price]
                new_v, best_action = self._expected_backup(t, holdings, cash, price)
                self.value_function[t, holdings, cash, price] = new_v
                calls += 1
                max_update = max(max_update, abs(new_v - old_v))

                # take greedy action by interacting with env
                next_obs, _env_reward, done, _info = self.envr.step(np.array([best_action], dtype=np.int32))
                next_cash = int(round(next_obs[0])); next_price = int(round(next_obs[1])); next_holdings = int(round(next_obs[2]))
                next_cash = self._clip_cash(next_cash)
                next_price = max(0, min(self.max_price, next_price))
                next_holdings = self._clip_holdings(next_holdings)

                # if the env says terminal at this transition, incorporate terminal payoff
                if done:

                    terminal_value = float(next_cash + next_holdings * next_price)
                    # set terminal layer value at t+1
                    self.value_function[t+1, next_holdings, next_cash, next_price] = terminal_value
                    break

                t += 1
                holdings, cash, price = next_holdings, next_cash, next_price


            if max_update < tolerance:
                break

        # Extract greedy policy from final V
        for t in range(T):
            for h in range(self.max_holdings + 1):
                for c in range(self.max_cash + 1):
                    for p in range(self.max_price + 1):
                        _, best_action = self._expected_backup_time(t, h, c, p)
                        self.policy[t, h, c, p] = int(best_action)

        return self.policy, self.value_function, iterations, max_update

    def _policy_evaluation(self, policy):
        
        V_pi = np.zeros_like(self.value_function)
        T = self.num_steps

        # At the final time step T, the value is the terminal wealth
        for h in range(self.max_holdings + 1):
            for c in range(self.max_cash + 1):
                for p in range(self.max_price + 1):
                    V_pi[T, h, c, p] = c + h * p

        for t in reversed(range(T)):
            for h in range(self.max_holdings + 1):
                for c in range(self.max_cash + 1):
                    for p in range(self.max_price + 1):

                        lots = policy[t, h, c, p]
                        
                        new_h, new_c, cost = self._apply_action(h, c, p, lots)
                        new_h_idx = self._clip_holdings(new_h)
                        new_c_idx = self._clip_cash(new_c)

                        immediate_reward = -cost
                        
                        # Calculate the expected future value by averaging over stochastic price changes
                        expected_future_v = 0.0
                        pmf = self.price_pmf[p]
                        for p_prime, prob in pmf:

                            expected_future_v += prob * V_pi[t + 1, new_h_idx, new_c_idx, p_prime]
                            
                        V_pi[t, h, c, p] = immediate_reward + self.gamma * expected_future_v
        return V_pi

    def policy_iteration(self, max_pi_iterations=100, verbose=False):
        
        print("Starting policy iteration training...")
        start_time = time.time() 

        T = self.num_steps
        
        v_start_history = []
        max_value_diff_history = [] 

        # Initialize a random (or all-zero) policy and a placeholder old value function
        self.policy = np.zeros((T, self.max_holdings + 1, self.max_cash + 1, self.max_price + 1), dtype=int)
        old_value_function = np.zeros_like(self.value_function) 
        
        for it in range(max_pi_iterations):
            print(f"Policy Iteration {it + 1}/{max_pi_iterations}...")
            
            # 1. Policy Evaluation
            print("  - Evaluating policy...")
            self.value_function = self._policy_evaluation(self.policy)

            max_diff = np.max(np.abs(self.value_function - old_value_function))
            max_value_diff_history.append(max_diff)
            old_value_function = self.value_function.copy() # Update for the next iteration

            # Track progress by recording the value of the initial state
            reset_ret = self.envr.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            cash = self._clip_cash(int(round(obs[0])))
            price = max(0, min(self.max_price, int(round(obs[1]))))
            holdings = self._clip_holdings(int(round(obs[2])))
            v_start_history.append(self.value_function[0, holdings, cash, price])

            # 2. Policy Improvement
            print("  - Improving policy...")
            new_policy = np.zeros_like(self.policy)
            for t in range(T):
                for h in range(self.max_holdings + 1):
                    for c in range(self.max_cash + 1):
                        for p in range(self.max_price + 1):
                            _, best_action = self._expected_backup(t, h, c, p)
                            new_policy[t, h, c, p] = best_action
            
            if np.array_equal(self.policy, new_policy):
                print("\nPolicy has converged.")
                self.policy = new_policy
                break
            
            self.policy = new_policy
        
        if it == max_pi_iterations - 1:
            print(f"\nMax policy iterations ({max_pi_iterations}) reached.")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTotal Execution Time: {execution_time:.2f} seconds")

        plt.figure(figsize=(10, 6))
        plt.plot(max_value_diff_history, marker='d', color='purple')
        plt.title('Maximum Value Difference vs. Policy Iteration')
        plt.xlabel('Policy Iteration Step')
        plt.ylabel('Max |V_new(s) - V_old(s)|')
        plt.xticks(range(len(max_value_diff_history)), [str(i+1) for i in range(len(max_value_diff_history))])
        plt.grid(True)
        plt.yscale('log') 
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(v_start_history, marker='o')
        plt.title('Expected Wealth from Start State vs. Policy Iteration')
        plt.xlabel('Policy Iteration Step')
        plt.ylabel('Expected Final Wealth ($)')
        plt.xticks(range(len(v_start_history)), [str(i+1) for i in range(len(v_start_history))])
        plt.grid(True)
        plt.show()

        return self.policy, self.value_function, it + 1, 0 


if __name__=="__main__":
    start_time=time.time()



    ###Part 1 and Part 2
    ####Please train the value and policy iteration training algo for the given three sequences of prices
    ####Config1
    env = DiscretePortfolioOptEnv(prices=[1, 3, 5, 5 , 4, 3, 2, 3, 5, 8])
   
    solver = Solver(env)
    policy, value_function, num_iter, delta = solver.policy_iteration()
    # print(policy[:4,:4,4])
    print(num_iter, delta)

    ####Config2
    env = DiscretePortfolioOptEnv(prices=[2, 2, 2, 4 ,2, 2, 4, 2, 2, 2])
    solver = Solver(env)
    policy, value_function, num_iter, delta = solver.policy_iteration()
    # print(policy[:4,:4,4])
    print(num_iter, delta)

    ####Config3
    env = DiscretePortfolioOptEnv(prices=[4, 1, 4, 1 ,4, 4, 4, 1, 1, 4])
    solver = Solver(env)
    policy, value_function, num_iter, delta = solver.policy_iteration()
    # print(policy[:4,:4,4])
    print(num_iter, delta)


    ###Part 3. (Portfolio Optimizaton)
    # env = DiscretePortfolioOptEnv(variance=1)
    # solver = Solver(env)
    # policy, value_function, num_iter, delta = solver.policy_iteration()
    # print(num_iter, delta)
