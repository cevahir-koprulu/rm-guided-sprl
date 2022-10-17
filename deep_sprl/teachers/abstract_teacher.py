import gym
import numpy as np
from gym import spaces
from abc import ABC, abstractmethod
from deep_sprl.teachers.util import Buffer
from stable_baselines3.common.vec_env import VecEnv
from collections import defaultdict


class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self):
        pass


class BaseWrapper(gym.Env):

    def __init__(self, env, teacher, discount_factor, context_visible, reward_from_info=False):
        gym.Env.__init__(self)
        self.stats_buffer = Buffer(3, 5000, True)

        self.env = env
        self.teacher = teacher
        self.discount_factor = discount_factor

        if context_visible:
            context = self.teacher.sample()
            low_ext = np.concatenate((self.env.observation_space.low, -np.inf * np.ones_like(context)))
            high_ext = np.concatenate((self.env.observation_space.high, np.inf * np.ones_like(context)))
            self.observation_space = gym.spaces.Box(low=low_ext, high=high_ext)
        else:
            self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.context_visible = context_visible
        self.cur_context = None
        self.cur_initial_state = None

        self.rm_transitions_buffer = {"transitions": dict(),
                                      "num_trajectories": 0}

        self.reward_from_info = reward_from_info

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward):
        pass

    def step(self, action):
        step = self.env.step(action)
        if self.context_visible:
            step = np.concatenate((step[0], self.cur_context)), step[1], step[2], step[3]
        self.update(step)
        # self.render()
        return step

    def get_rm_transitions_buffer(self):
        return self.rm_transitions_buffer

    def reset_rm_transitions_buffer(self):
        self.rm_transitions_buffer = {"transitions": dict(),
                                      "num_trajectories": 0}

    def reset(self):
        if self.cur_context is None:
            self.cur_context = self.teacher.sample()
        self.env.unwrapped.context = self.cur_context.copy()
        obs = self.env.reset()
        self.rm_transitions_buffer["num_trajectories"] += 1

        if self.context_visible:
            obs = np.concatenate((obs, self.cur_context))

        self.cur_initial_state = obs.copy()
        return obs

    def set_current_context(self, context=None):
        self.cur_context = context.copy() if context is not None else None

    def get_current_context(self):
        return self.cur_context.copy()

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def update(self, step):
        reward = step[3]["reward"] if self.reward_from_info else step[1]
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward

        past_rm_state, current_rm_state = self.env.unwrapped.get_rm_transition()
        contexts_for_transition = self.env.unwrapped.get_rm_transition_context_map()[(
            past_rm_state,
            current_rm_state)]
        if tuple(contexts_for_transition) not in self.rm_transitions_buffer["transitions"]:
            self.rm_transitions_buffer["transitions"][tuple(contexts_for_transition)] = {"contexts": list(),
                                                                                         "rewards": list()}
        if len(contexts_for_transition) == 0:
            self.rm_transitions_buffer["transitions"][tuple(contexts_for_transition)]["contexts"].append(
                self.cur_context.copy())
        else:
            self.rm_transitions_buffer["transitions"][tuple(contexts_for_transition)]["contexts"].append(
                self.cur_context.copy()[contexts_for_transition])
        self.rm_transitions_buffer["transitions"][tuple(contexts_for_transition)]["rewards"].append(self.cur_disc * reward)

        self.cur_disc *= self.discount_factor
        self.step_length += 1.
        if step[2]:
            self.done_callback(step, self.cur_initial_state.copy(), self.cur_context.copy(), self.discounted_reward)

            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, self.step_length))
            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.cur_initial_state = None

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_step_length = np.mean(steps)

            return mean_reward, mean_disc_reward, mean_step_length

    def get_buffer(self):
        rewards, disc_rewards, steps = self.stats_buffer.read_buffer()
        return disc_rewards
