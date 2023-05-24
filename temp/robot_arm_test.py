import os
import sys
import time
import gym
import deep_sprl.environments
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import MlpPolicy as PPOMlpPolicy

fig_dir = "robot_arm_test"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

render_on = True

env = gym.make("ContextualFetchPickAndPlay6DProduct-v1")
main_env = env.env.env
observation = env.reset()
print(f"initial state: {observation}")
for t in range(50):
    action = np.zeros(4)
    if main_env._rm_state == 0:
        action[:3] = 5*(main_env._locs["loc_1"]-observation[:3])
    elif main_env._rm_state == 1:
        action[:3] = 5*(main_env._locs["loc_2"]-observation[:3])  
    elif main_env._rm_state == 2:
        action[:3] = 5*(main_env._locs["loc_1"]-observation[:3])        
    observation, reward, done, info = env.step(action)
    print("--------------------------------")
    print(f"t: {t}")
    print(f"rm_state: {main_env._rm_state}")
    print(f"observation: {observation}")
    print(f"reward: {reward}")
    print(f"done: {done}")
    print(f"info: {info}")
    if render_on:
        main_env._render_callback()
        main_env._get_viewer("rgb_array").render(500, 500)
        # window size used for old mujoco-py:
        data = main_env._get_viewer("rgb_array").read_pixels(500, 500, depth=False)
        # original image is upside-down, so flip it
        data =  data[::-1, :, :]
        plt.imsave(fname=f"{fig_dir}/robot_arm_t_{t}.png", arr=data)
    if done:
        observation = env.reset()
        break

env.close()