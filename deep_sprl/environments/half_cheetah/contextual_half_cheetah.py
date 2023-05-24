"""
This code add event detectors to the Ant3 Environment
"""
import copy
import gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

class ContextualHalfCheetah(gym.Env):

    FLAG_1 = 1.
    FLAG_2 = 4.
    FLAG_3 = 7.

    def __init__(self, context=None, product_cmdp=False, rm_state_onehot=True):
        exclude_current_positions_from_observation = False
        self.env = gym.make("HalfCheetah-v3",
                            exclude_current_positions_from_observation=exclude_current_positions_from_observation).unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.product_cmdp = product_cmdp
        self.rm_state_onehot = rm_state_onehot
        if context is None:
            context = np.array([self.FLAG_1, self.FLAG_2, self.FLAG_3])
        self.context = context
        self._update_env_config()
        self.rm_info = {"num_states": 5,
                        "sink": None,
                        "goal": 4}
        self.rm_transition_context_map = {(0, 0): [0],
                                          (0, 1): [0],
                                          (1, 1): [1],
                                          (1, 2): [1],
                                          (2, 2): [0],
                                          (2, 3): [0],
                                          (3, 3): [2],
                                          (3, 4): [2],
                                          }

        if self.product_cmdp:
            if self.rm_state_onehot:
                low_ext = np.concatenate((self.observation_space.low, np.zeros(self.rm_info["num_states"])))
                high_ext = np.concatenate((self.observation_space.high, np.ones(self.rm_info["num_states"])))
            else:
                low_ext = np.concatenate((self.observation_space.low, np.array(0.)))
                high_ext = np.concatenate((self.observation_space.high, np.array(self.rm_info["num_states"]-1)))
            self.observation_space = gym.spaces.Box(low=low_ext, high=high_ext)

        self.rewards = {
            0: {0: None, 1: 10.},
            1: {1: None, 2: 100.},
            2: {2: None, 3: 500.},
            3: {3: None, 4: 1000.},
            4: {4: None},
        }

        self._rm_state = None
        self._past_rm_state = None
        self._state = None
        self._num_step = 0

    def _update_env_config(self):
        self._flags = np.copy(self.context)

    def get_rm_transition(self):
        return self._past_rm_state, self._rm_state

    def reset(self):
        self._num_step = 0
        self._past_rm_state = None
        self._rm_state = 0
        _state = self.env.reset()
        if self.product_cmdp:
            if self.rm_state_onehot:
                _state_rm_ext = np.zeros(self.rm_info["num_states"])
                _state_rm_ext[self._rm_state] = 1.
                _state = np.concatenate((_state, _state_rm_ext))
            else:
                _state = np.concatenate((_state, np.array(self._rm_state)))
        self._state = np.copy(_state)
        return _state

    def step(self, action):
        self._update_env_config()

        next_state, reward, done, info = self.env.step(action)
        reward = info["reward_ctrl"]

        if self._rm_state == 0:
            next_rm_state = 0
            if info['x_position'] > self._flags[0]:
                next_rm_state = 1
        elif self._rm_state == 1:
            next_rm_state = 1
            if info['x_position'] > self._flags[1]:
                next_rm_state = 2
        elif self._rm_state == 2:
            next_rm_state = 2
            if info['x_position'] < self._flags[0]:
                next_rm_state = 3
        elif self._rm_state == 3:
            next_rm_state = 3
            if info['x_position'] > self._flags[2]:
                next_rm_state = 4
                done = True
        elif self._rm_state == 4:
            next_rm_state = 4

        reward_rm = self.rewards[self._rm_state][next_rm_state]
        if reward_rm is not None:
            reward = reward_rm

        self._num_step += 1
        self._past_rm_state = copy.deepcopy(self._rm_state)
        self._rm_state = copy.deepcopy(next_rm_state)

        info["success"] = next_rm_state == self.rm_info["goal"]
        info["mission"] = next_rm_state

        if self.product_cmdp:
            if self.rm_state_onehot:
                _state_rm_ext = np.zeros(self.rm_info["num_states"])
                _state_rm_ext[self._rm_state] = 1.
                next_state = np.concatenate((next_state, _state_rm_ext))
            else:
                next_state = np.concatenate((next_state, np.array(self._rm_state)))
        self._state = np.copy(next_state)
        return next_state, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode=mode)