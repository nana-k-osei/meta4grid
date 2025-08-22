# register_goto_red_blue_ball.py
from gymnasium.envs.registration import register
from minigrid.envs.babyai.goto import GoToRedBlueBall

register(
    id='BabyAI-GoToRedBlueBall-v0',
    entry_point='minigrid.envs.babyai.goto:GoToRedBlueBall',
)
