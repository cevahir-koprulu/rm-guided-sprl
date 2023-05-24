"""
This code add event detectors to the Ant3 Environment
"""
import copy
import gym
import numpy as np

class ContextualSwimmer(gym.Env):

    FLAG_1 = -1.
    FLAG_2 = 1.

    def __init__(self, context=None, product_cmdp=False, rm_state_onehot=True):
        exclude_current_positions_from_observation = False
        self.env = gym.make("Swimmer-v3",
                            ctrl_cost_weight=1e-5,
                            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                            ).unwrapped
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.product_cmdp = product_cmdp
        self.rm_state_onehot = rm_state_onehot
        if context is None:
            context = np.array([self.FLAG_1, self.FLAG_2])
        self.context = context
        self._update_env_config()
        self.rm_info = {"num_states": 3,
                        "sink": None,
                        "goal": 2}
        self.rm_transition_context_map = {(0, 0): [1],
                                          (0, 1): [1],
                                          (1, 1): [0],
                                          (1, 2): [0],
                                          (2, 2): [],
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
            0: {0: None, 1: 100.},
            1: {1: None, 2: 1000.},
            2: {2: None},
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
        if self._state is not None:
            print(f"RESET - position: {self._state[:2]} | context: {self.context}")
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

        if self._past_rm_state is None:
            print(f"START - position: {info['x_position']} | context: {self.context}")

        if self._rm_state == 0:
            next_rm_state = 0
            if info['x_position'] > self._flags[1]:
                next_rm_state = 1
        elif self._rm_state == 1:
            next_rm_state = 1
            if info['x_position'] < self._flags[0]:
                next_rm_state = 2
                done = True
        elif self._rm_state == 2:
            next_rm_state = 2

        if self._rm_state != next_rm_state:
            print(f"TRANSITION - from {self._rm_state} to {next_rm_state} | context: {self.context} | step: {self._num_step+1}")

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
        