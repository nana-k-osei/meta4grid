import gymnasium as gym
import numpy as np
import os

# Setup a macro log file path (ensure logs directory exists)
LOG_FILE = "./red_ball_playground/macro_usage_log.txt"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Track object that gave the highest reward and its macro usage
class RewardTracker(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.tracked_type = None
        self.tracked_color = None
        self.macro_use_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward > 0:
            image = obs["image"]
            x, y = self.env.unwrapped.agent_pos
            obj_id, color_id, _ = image[x, y]
            self.tracked_type = obj_id
            self.tracked_color = color_id

        # Check if tracked object appears in view
        used_macro = False
        if self.tracked_type is not None and self.tracked_color is not None:
            for x in range(obs["image"].shape[0]):
                for y in range(obs["image"].shape[1]):
                    obj_id, color_id, _ = obs["image"][x, y]
                    if obj_id == self.tracked_type and color_id == self.tracked_color:
                        self.macro_use_count += 1
                        used_macro = True
                        break

        if used_macro:
            with open(LOG_FILE, "a") as f:
                f.write("[RewardTracker] Macro used: tracked_type=%s, color=%s\n" %
                        (self.tracked_type, self.tracked_color))

        return obs, reward, terminated, truncated, info

# Detect stalling and trigger recovery macro
class UnstuckHandler(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_pos = None
        self.stuck_counter = 0
        self.unstuck_macro_used = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_pos = tuple(self.env.unwrapped.agent_pos)
        self.stuck_counter = 0
        return obs, info

    def step(self, action):
        # Check if the agent is stuck before stepping
        curr_pos = tuple(self.env.unwrapped.agent_pos)
        if curr_pos == self.last_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        # If agent is stuck, override the action with a random one
        if self.stuck_counter >= 2:
            action = self.env.action_space.sample()
            self.unstuck_macro_used += 1
            self.stuck_counter = 0

            with open(LOG_FILE, "a") as f:
                f.write("[UnstuckHandler] Macro used: agent was stuck\n")

        # Now take a single step
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_pos = tuple(self.env.unwrapped.agent_pos)
        return obs, reward, terminated, truncated, info

