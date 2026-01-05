import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import imageio
import numpy as np
import random
import os
import math
import matplotlib.pyplot as plt
import pandas as pd 
from dataclasses import dataclass
from collections import defaultdict

def set_global_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

SEED = 509
set_global_seed(SEED)

class DiagonalFrozenLake(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, map_size=16, slip_prob=0.0, start_state = (0, 5)):
        super().__init__()
        self.render_mode = render_mode
        self.map_size = map_size
        self.slip_prob = slip_prob
        self.start_state = start_state
        
        self.desc = self._generate_map(map_size)
        self.nrow, self.ncol = self.desc.shape

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.nrow * self.ncol)

        self.cell_size = 60
        self.window = None
        self.clock = None

        self.goal_state = (self.nrow - 1, self.ncol - 1)
        self.agent_pos = self.start_state

    def _generate_map(self, n):
        grid = np.full((n, n), "F", dtype=str)
        grid[self.start_state] = "S"
        grid[-1, -1] = "G"
        return grid

    def state_to_index(self, state):
        return state[0] * self.ncol + state[1]

    def index_to_state(self, idx):
        return (idx // self.ncol, idx % self.ncol)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.start_state
        return self.state_to_index(self.agent_pos), {}

    def step(self, action):
        r, c = self.agent_pos

        if np.random.rand() < self.slip_prob:
            action = self.action_space.sample()

        # Movement (only if not in last row)
        if r < self.nrow - 1:
            if action == 0:   # down
                r += 1
            elif action == 1: # down-left
                r += 1
                c = max(c - 1, 0)
            elif action == 2: # down-right
                r += 1
                c = min(c + 1, self.ncol - 1)

        new_pos = (r, c)
        cell = self.desc[r, c]

        reward, terminated, truncated = 0.0, False, False

        if cell == "H":
            terminated = True
        elif cell == "G":
            reward = 1
            terminated = True
        elif r == self.nrow - 1:
            # last row but not goal
            terminated = True
            reward = -1.0

        # Extra penalty if action is down-right and the agent actually moved
        if action == 2 and self.agent_pos[0] < r:  
            reward -= 0.1 / math.sqrt(self.map_size)

        self.agent_pos = new_pos
        return self.state_to_index(new_pos), reward, terminated, truncated, {}

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.ncol * self.cell_size, self.nrow * self.cell_size)
            )
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.ncol * self.cell_size, self.nrow * self.cell_size))
        canvas.fill((255, 255, 255))

        # Draw grid
        for r in range(self.nrow):
            for c in range(self.ncol):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                if self.desc[r, c] == "S":
                    color = (0, 255, 0)
                elif self.desc[r, c] == "F":
                    color = (180, 220, 255)
                elif self.desc[r, c] == "H":
                    color = (0, 0, 0)
                elif self.desc[r, c] == "G":
                    color = (255, 215, 0)
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, 1)

        # Draw agent
        ar, ac = self.agent_pos
        pygame.draw.circle(canvas, (255, 0, 0),
                           (ac * self.cell_size + self.cell_size // 2,
                            ar * self.cell_size + self.cell_size // 2),
                           self.cell_size // 3)

        self.window.blit(canvas, (0, 0))
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
