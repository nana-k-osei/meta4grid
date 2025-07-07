import gymnasium as gym
import torch
import torch.nn as nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ---- Define the custom feature extractor ----
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),  # Keep input size
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape with dummy input
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )


    def forward(self, observations):
        return self.linear(self.cnn(observations))

# ---- Setup environment ----
env = gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)                   # Converts dict obs to image
#env = Monitor(env)                         # Logs episode rewards
env = Monitor(env, filename="ppo_minigrid_log")  # Logs rewards to CSV
env = DummyVecEnv([lambda: env])           # Vectorized wrapper
env = VecTransposeImage(env)               # Makes obs (C, H, W) for CNN

# ---- Setup PPO training ----
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cuda")  # use "cpu" if no GPU

model.learn(total_timesteps=200_000)  # Adjust as needed

# ---- Save model ----
model.save("ppo_minigrid_empty8x8_v2")
