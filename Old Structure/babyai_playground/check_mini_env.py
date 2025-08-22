import gymnasium as gym
import minigrid
envs = [env_id for env_id in gym.registry.keys() if "MiniGrid" in env_id]
print(envs)

