import os 
import random
import gymnasium as gym
import numpy as np
from PIL import Image, ImageEnhance
from gymnasium import spaces



class FootballSkillsEnv(gym.Env):

    metadata = {"render_modes": ["human", "gif", "ascii"]}

    def __init__(self, render_mode=None, grid_size = 20, degrade_pitch=False, degrade_mode="sigmoid"):
        super().__init__()
        self.grid_size = grid_size
        self.default_save_path="./output_seeds"
        os.makedirs(self.default_save_path, exist_ok = True)

        self.start_pos = (0, 10)

        self.goal_positions = [(self.grid_size-1, self.grid_size//2-1), (self.grid_size-1, self.grid_size//2), (self.grid_size-1, self.grid_size//2+1)]
        cones_positions = [
            (17, 8),                     
            (17, 10),                    
            (15, 9),                    
        ]

        cutouts_positions = [
            (18, 7), (18, 8), (18, 9),
            (16, 8),                     
            (15, 10)                     
        ]


        self.obstacles = {"cones": cones_positions, "cutouts": cutouts_positions}

        self.background=Image.open('assets/bg_large.jpg')
        self.background = self.background.rotate(90, expand=True)
        self.background = ImageEnhance.Brightness(self.background).enhance(0.7)

        self.player_png=Image.open('assets/ronaldo.png')
        self.cone_png=Image.open('assets/cone.png')
        self.cutout_png=Image.open('assets/cutouts.png')
        self.goal_png=Image.open('assets/sui.png')
        self.ball_png=Image.open('assets/ball.png')

        self.action_mapping = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
            4: "short_shot_straight", # High probability, short distance
            5: "long_shot_straight",  # Low probability, long distance
            6: "long_shot_curl"     # Low probability, long distance, curls
        }
        self.movement_actions = {0, 1, 2, 3}
        self.shooting_actions = {4, 5, 6}
        self.action_space = spaces.Discrete(len(self.action_mapping))

        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_size),
            spaces.Discrete(self.grid_size),
            spaces.Discrete(2) # 0 for False, 1 for True
        ))

        self.degrade_pitch = degrade_pitch
        self.degrade_mode = degrade_mode # "sigmoid" or "linear"
        self.time_step = 0
        self.temps = []

        self.player_position = None
        self.ball_position = None
        self.has_shot = False
        self.done = False

        if not self.degrade_pitch:
            self.T = self._generate_transition_matrix()
        else:
            self.T = {}
            
        self.render_mode = render_mode

        self.reset()

    def _get_obs(self):
        return (self.player_position[0], self.player_position[1], int(self.has_shot))

    def validate_cones_and_cutouts(self):
        for i in self.obstacles:
            for j in self.obstacles[i]:
                if j in self.goal_positions:
                    raise Exception(f"{i}, {j} overlaps with goal position, check cone and cutout positions.")

    def state_to_index(self, state):
        """Converts a (x, y, has_shot) state tuple to a single integer index."""
        x, y, has_shot = state
        return (x * self.grid_size + y) * 2 + int(has_shot)

    def index_to_state(self, index):
        """Converts a single integer index back to a (x, y, has_shot) state tuple."""
        has_shot = index % 2
        base = index // 2
        x = base // self.grid_size
        y = base % self.grid_size
        return (x, y, has_shot)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.time_step = 0

        self.player_position = self.start_pos
        self.ball_position = self.player_position
        self.has_shot = False
        self.done = False
        if self.render_mode == 'gif':
            self.frames = []
            self.prepare_for_gif()
        return self._get_obs(), {}

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        self.time_step += 1

        # Penalize and end episode for invalid actions after shooting
        if self.has_shot:
            reward = -25 
            self.done = True
            return self._get_obs(), reward, self.done, False, {"error": "Cannot act after shooting"}

        player_pos_at_shot_time = self.player_position
        current_obs = self._get_obs()

        if self.degrade_pitch:
            transitions = self._transition_for(current_obs, action)
        else:
            transitions = self.T.get((current_obs, action))

        if not transitions:
            return self._get_obs(), -1, self.done, False, {}

        probs, next_states = zip(*transitions)
        next_state_tuple = next_states[np.random.choice(len(next_states), p=probs)]
        next_pos = (next_state_tuple[0], next_state_tuple[1])

        if action in self.movement_actions:
            self.player_position = next_pos
            self.ball_position = next_pos
        elif action in self.shooting_actions:
            self.ball_position = next_pos
            self.has_shot = True

        reward = self._get_reward(self.ball_position, action, player_pos_at_shot_time)
        self.done = self._is_terminal(next_state_tuple)

        return self._get_obs(), reward, self.done, False, {}

    def _get_reward(self, ball_pos, action, player_pos_at_shot):
        if action in self.shooting_actions:
            if ball_pos in self.goal_positions:
                # Reward is higher for goals scored from farther away (smaller x)
                distance_from_start = player_pos_at_shot[0]
                distance_bonus = (self.grid_size - 1 - distance_from_start) * 10
                base_reward = 50
                return base_reward + distance_bonus
            else:
                return -30 # Penalty for a missed shot
        
        if ball_pos in self.obstacles["cones"]: return -10
        elif ball_pos in self.obstacles["cutouts"]: return -20
        return -1 # Standard movement cost

    def _is_terminal(self, state):
        # Game ends if a shot has been taken. The outcome is determined by the reward.
        if state[-1]==1:
            return True
        return False

    def get_slip_probability(self):
        """Calculate current slip probability based on time and degradation mode"""
        if not self.degrade_pitch:
            return 0.0
            
        if self.degrade_mode == "sigmoid":
            max_slip = 0.4       
            steepness = 0.05     
            midpoint = 20        
            slip_prob = max_slip * (1.0 / (1.0 + np.exp(-steepness * (self.time_step - midpoint))))
        else:
            rate = 0.01         
            max_slip = 0.4
            slip_prob = min(max_slip, rate * self.time_step)
            
        return slip_prob

    def prepare_for_gif(self):
        if self.render_mode == 'gif':

            self.height_unit = self.background.height // self.grid_size
            self.width_unit = self.background.width // self.grid_size
            self.background = self.background.resize((700, 700), Image.Resampling.LANCZOS)
            self.cutout_png = ImageEnhance.Color(self.cutout_png).enhance(1.2)
            img_array = np.array(self.cutout_png)
            img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.5, 0, 255)
            img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.5, 0, 255)
            self.cutout_png = Image.fromarray(img_array.astype(np.uint8))
            max_width = int(self.width_unit * 0.8)
            max_height = int(self.height_unit * 0.8)
            player_width = int(self.width_unit * 1.5)
            player_height = int(self.height_unit * 1.5)

            self.cone_png.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.cutout_png.thumbnail((int(max_width * 1), int(max_height * 1)), Image.Resampling.LANCZOS)
            self.goal_png.thumbnail((player_width, player_height), Image.Resampling.LANCZOS)
            self.ball_png.thumbnail((int(max_width * 0.4), int(max_height * 0.4)), Image.Resampling.LANCZOS)
            self.player_png.thumbnail((player_width, player_height), Image.Resampling.LANCZOS)

            self.static_background = self.background.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(self.static_background, 'RGBA')
            
            if self.goal_positions:
                min_x = min(pos[0] for pos in self.goal_positions)
                max_x = max(pos[0] for pos in self.goal_positions)
                min_y = min(pos[1] for pos in self.goal_positions)
                max_y = max(pos[1] for pos in self.goal_positions)
                
                rect_left = min_y * self.width_unit
                rect_top = min_x * self.height_unit
                rect_right = (max_y + 1) * self.width_unit
                rect_bottom = (max_x + 1) * self.height_unit
                
                draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], 
                            fill=(0, 255, 0, 128))  # Green with alpha=128 (50% transparent)

            for obstacle_type, positions in self.obstacles.items():
                for (x, y) in positions:
                    if obstacle_type == "cones":
                        img_to_paste = self.cone_png
                    else: 
                        img_to_paste = self.cutout_png
                    
                    img_w, img_h = img_to_paste.size
                    offset_x = (self.width_unit - img_w) // 2
                    offset_y = (self.height_unit - img_h) // 2
                    
                    grid_center_x = y * self.width_unit + self.width_unit // 2
                    grid_center_y = x * self.height_unit + self.height_unit // 2
                    
                    paste_x = grid_center_x - img_w // 2
                    paste_y = grid_center_y - img_h // 2
                    
                    paste_pos = (paste_x, paste_y)
                    self.static_background.paste(img_to_paste, paste_pos, img_to_paste)

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode=='ascii':
            grid = np.full((self.grid_size, self.grid_size), '·', dtype=str)

            for (x, y) in self.obstacles["cones"]: grid[x, y] = "C"
            for (x, y) in self.obstacles["cutouts"]: grid[x, y] = "X"
            for (x, y) in self.goal_positions: grid[x, y] = "G"

            px, py = self.player_position
            bx, by = self.ball_position

            grid[px, py] = "P"

            if self.has_shot:
                if grid[bx, by] != 'G': 
                    grid[bx, by] = "o" 
                if (px, py) == (bx, by):
                    grid[px, py] = "P/o" 

            print(f"\nPlayer: {self.player_position}, Ball: {self.ball_position}, Has Shot: {self.has_shot}")
            for row in grid:
                print(" ".join(row))

            if self.done:
                if self.ball_position in self.goal_positions:
                    print("🎉 GOAL! Ball reached the goal!")
                elif self.ball_position in self.obstacles["cones"]:
                    print("💥 Ball hit a cone!")
                elif self.ball_position in self.obstacles["cutouts"]:
                    print("❌ Ball hit a cutout!")
                else:
                    print("💨 Missed! The ball went out of bounds.")
            elif self.has_shot:
                print("⚽ Ball is in flight!")
            else:
                print("🏃 Player moving with ball...")
        elif self.render_mode == 'gif':
            frame = self.static_background.copy()

            px, py = self.player_position
            player_w, player_h = self.player_png.size
            
            grid_center_x = py * self.width_unit + self.width_unit // 2
            grid_center_y = px * self.height_unit + self.height_unit // 2
            
            player_paste_x = grid_center_x - player_w // 2
            player_paste_y = grid_center_y - player_h // 2
            if not self.has_shot:
                player_paste_pos = (player_paste_x, player_paste_y)
                frame.paste(self.player_png, player_paste_pos, self.player_png)
            if self.has_shot:
                if self.ball_position in self.goal_positions:
                    goal_x, goal_y = self.ball_position  
                    goal_w, goal_h = self.ball_png.size
                    
                    goal_grid_center_x = goal_y * self.width_unit + self.width_unit // 2
                    goal_grid_center_y = goal_x * self.height_unit + self.height_unit // 2
                    
                    goal_paste_x = goal_grid_center_x - goal_w // 2
                    goal_paste_y = goal_grid_center_y - goal_h // 2

                    celebration_paste_x = grid_center_x - player_w // 2
                    celebration_paste_y = grid_center_y - player_h // 2
                    
                    goal_paste_pos = (goal_paste_x, goal_paste_y)
                    celebration_paste_pos=(celebration_paste_x, celebration_paste_y)
                    frame.paste(self.goal_png, celebration_paste_pos, self.goal_png)
                    frame.paste(self.ball_png, goal_paste_pos, self.ball_png)
                else:
                    bx, by = self.ball_position
                    ball_w, ball_h = self.ball_png.size
                    ball_offset_x = (self.width_unit - ball_w) // 2
                    ball_offset_y = (self.height_unit - ball_h) // 2
                    ball_paste_pos = (by * self.width_unit + ball_offset_x, bx * self.height_unit + ball_offset_y)
                    frame.paste(self.ball_png, ball_paste_pos, self.ball_png)
                    
            self.frames.append(frame)


    def generate_gif(self, filename = "output.gif"):
        if self.render_mode == 'gif' :
            if self.frames:
                self.frames[0].save(f"{self.default_save_path}/{filename}", save_all=True, append_images=self.frames[1:], duration=500, loop=0)
                print(f"Episode GIF saved to {self.default_save_path}/{filename}")
            else:
                print("No frames to save for GIF. Ensure you called render() during the episode.")

    def get_gif(self, policy, seed = 20, filename = "output.gif"):
        obs, _ = self.reset(seed = seed)
        done = False
        steps = 0
        max_steps = 100
        total_reward = 0
        
        print(f"Starting rollout from position: {obs}")
        
        while not done and steps < max_steps:
            self.render()
            
            if self.degrade_pitch:
                time_step = min(steps, len(policy) - 1)
                s_index = self.state_to_index(obs)
                action = policy[time_step][s_index]
            else:
                s_index = self.state_to_index(obs)
                action = policy[s_index]
            
            if action == -1:
                print("No valid action found!")
                break
                
            obs, reward, done, _, _ = self.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                print()
                print("Reached terminal state.")
                self.render()
                break
        
        self.generate_gif(filename = filename)
        return steps, total_reward

    def _generate_transition_matrix(self):
        T = {}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for is_shot_val in [0, 1]:
                    state_tuple = (x, y, is_shot_val)
                    for action in range(self.action_space.n):
                        if is_shot_val == 1 and action in self.movement_actions:
                            continue

                        transitions = self._transition_for(state_tuple, action)
                        if transitions:
                            T[(state_tuple, action)] = transitions
        return T

    def _get_nearby_obstacles(self, pos):
        x, y, _ = pos
        all_obstacles = self.obstacles["cones"] + self.obstacles["cutouts"]
        obstacles = {'front': [], 'left': [], 'right': []}
        fx = max(0, x - 2)

        straight_path_rows = range(fx, x)
        left_path_cols = range(y - 1, y + 1)
        right_path_cols = range(y, y + 2)

        for (ox, oy) in all_obstacles:
            if ox in straight_path_rows:
                if oy == y: obstacles['front'].append((ox, oy))
                if oy in left_path_cols: obstacles['left'].append((ox, oy))
                if oy in right_path_cols: obstacles['right'].append((ox,oy))
        return obstacles

    def _is_obstacle_in_path(self, start_pos, end_pos):
        all_obstacles = self.obstacles["cones"] + self.obstacles["cutouts"]
        x1, y1 = start_pos
        x2, y2, _ = end_pos

        for i in range(1, int(max(abs(x2 - x1), abs(y2 - y1))) + 1):
            path_x = round(x1 + i * (x2 - x1) / max(1, abs(x2 - x1), abs(y2 - y1)))
            path_y = round(y1 + i * (y2 - y1) / max(1, abs(y2 - y1), abs(y1 - y2)))
            
            current_path_pos = (int(path_x), int(path_y))
            
            if current_path_pos in all_obstacles:
                return current_path_pos 
                
        return None

    def get_transitions_at_time(self, state, action, time_step = None):
        """
        Get transitions for a given state-action pair at a specific time step.
        This handles both degraded and non-degraded pitch conditions.
        """
        
        if not self.degrade_pitch:
            return self.T.get((state, action), [])

        elif self.degrade_pitch and time_step is None:
            _, _, _ , _ ,_=self.step(0)
            transitions = self._transition_for(state, action)
            return transitions
        else:
            original_time = self.time_step
            self.time_step = time_step
            transitions = self._transition_for(state, action)
            self.time_step = original_time
            return transitions

    def _transition_for(self, pos, action):
        x, y, _ = pos
        outcomes = []
        
        def clip(p):
            return max(0, min(self.grid_size - 1, p))

        # Get current slip probability
        slip_prob = self.get_slip_probability()
        success_prob = 1.0 - slip_prob
        side_slip = slip_prob / 3.0

        # MOVEMENT ACTIONS (0-3) - affected by pitch degradation
        if action in self.movement_actions:
            if action == 0:   # up
                return [(success_prob, (clip(x - 1), y, 0)),
                        (2*side_slip, (x, clip(y - 1), 0)),
                        (side_slip, (x, clip(y + 1), 0))]
            elif action == 1: # down
                return [(success_prob, (clip(x + 1), y, 0)),
                        (side_slip, (x, clip(y - 1), 0)),
                        (2*side_slip, (x, clip(y + 1), 0))]
            elif action == 2: # left
                return [(success_prob, (x, clip(y - 1), 0)),
                        (side_slip, (clip(x - 1), y, 0)),
                        (2*side_slip, (clip(x + 1), y, 0))]
            elif action == 3: # right
                return [(success_prob, (x, clip(y + 1), 0)),
                        (2*side_slip, (clip(x - 1), y, 0)),
                        (side_slip, (clip(x + 1), y, 0))]

        # SHOOTING ACTIONS (4-6) - ALSO affected by pitch degradation now
        elif action in self.shooting_actions:
            base_outcomes = []
            
            # ACTION 4: Short Shot - degradation affects accuracy
            if action == 4:
                fx = clip(x + 2)
                if self.degrade_pitch:
                    # Degraded shot accuracy
                    accuracy = 0.85 - slip_prob * 0.4  # accuracy decreases with slip
                    miss_prob = (1 - accuracy) / 3
                    base_outcomes = [
                        (accuracy, (fx, y, 1)), 
                        (miss_prob, (fx - 1, y, 1)), 
                        (miss_prob, (fx, clip(y - 1), 1)), 
                        (miss_prob, (fx, clip(y + 1), 1))
                    ]
                else:
                    # Original accuracy
                    base_outcomes = [(0.85, (fx, y, 1)), (0.05, (fx - 1, y, 1)), 
                                   (0.05, (fx, clip(y - 1), 1)), (0.05, (fx, clip(y + 1), 1))]

            # ACTION 5: Long Shot - more affected by degradation
            elif action == 5:
                fx = clip(x + 4)
                if self.degrade_pitch:
                    # More severely affected by pitch conditions
                    accuracy = 0.60 - slip_prob * 0.5
                    miss_prob = (1 - accuracy) / 3
                    base_outcomes = [
                        (accuracy, (fx, y, 1)), 
                        (miss_prob, (fx - 1, y, 1)), 
                        (miss_prob, (fx, clip(y - 1), 1)), 
                        (miss_prob, (fx, clip(y + 1), 1))
                    ]
                else:
                    base_outcomes = [(0.60, (fx, y, 1)), (0.20, (fx - 1, y, 1)), 
                                   (0.10, (fx, clip(y - 1), 1)), (0.10, (fx, clip(y + 1), 1))]

            # ACTION 6: Curl Shot - degradation affects curl control
            else: # action == 6
                fx = clip(x + 4)
                if self.degrade_pitch:
                    # Pitch conditions make curl unpredictable
                    accuracy = 0.55 - slip_prob * 0.45
                    wild_curl = slip_prob * 0.3  # more extreme misses due to bad pitch
                    normal_miss = (1 - accuracy - wild_curl) / 2
                    base_outcomes = [
                        (accuracy, (fx, clip(y + 2), 1)), 
                        (normal_miss, (fx - 1, clip(y + 1), 1)), 
                        (normal_miss, (fx, clip(y + 1), 1)), 
                        (wild_curl, (fx, clip(y + 4), 1))  # wild miss due to bad pitch
                    ]
                else:
                    base_outcomes = [(0.55, (fx, clip(y + 2), 1)), (0.25, (fx - 1, clip(y + 1), 1)), 
                                   (0.10, (fx, clip(y + 1), 1)), (0.10, (fx, clip(y + 4), 1))]

            # Check for obstacles in path and adjust outcomes
            final_outcomes = []
            for prob, target_pos in base_outcomes:
                blocking_obstacle = self._is_obstacle_in_path((x, y), target_pos)
                
                if blocking_obstacle:
                    final_outcomes.append((prob, (blocking_obstacle[0], blocking_obstacle[1], 1)))
                else:
                    final_outcomes.append((prob, target_pos))
            
            # Consolidate duplicate states
            consolidated = {}
            for prob, state in final_outcomes:
                consolidated[state] = consolidated.get(state, 0) + prob
            
            return [(prob, state) for state, prob in consolidated.items()]

        return []