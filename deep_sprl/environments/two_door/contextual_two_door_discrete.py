import copy

import numpy as np
from gym import Env, spaces
from deep_sprl.util.viewer import Viewer


class ContextualTwoDoorDiscrete(Env):

    def __init__(self, context=np.array([0., 0., -2., 0.]), door_widths=[5., 5.], product_cmdp=False, rm_state_onehot=True):
        self.action_space = spaces.Box(np.array([0.]), np.array([1.]))
        self.product_cmdp = product_cmdp
        self.rm_state_onehot = rm_state_onehot
        self.context = context
        self._door_widths = door_widths
        self._chest_shape = [5., 5.]
        self._update_env_config()
        self.rm_info = {"num_states": 6,
                        "sink": 5,
                        "goal": 4}
        self.rm_transition_context_map = {(0, 0): [],
                                              (0, 1): [0],
                                              (0, 5): [0],
                                              (1, 1): [0, 2],
                                              (1, 2): [2],
                                              (1, 5): [0],
                                              (2, 2): [0, 1],
                                              (2, 3): [1],
                                              (2, 5): [0, 1],
                                              (3, 3): [0, 1, 3],
                                              (3, 4): [3],
                                              (3, 5): [0, 1]}

        if self.product_cmdp:
            if self.rm_state_onehot:
                self.observation_space = spaces.Box(np.array([-4., -4., 0., 0., 0., 0., 0., 0.]),
                                                    np.array([4., 4., 1., 1., 1., 1., 1., 1.]))
            else:
                self.observation_space = spaces.Box(np.array([-4., -4., 0.]),
                                                    np.array([4., 4., 5.]))
        else:
            self.observation_space = spaces.Box(np.array([-4., -4.]),
                                                np.array([4., 4.]))

        self.rewards = {0: {0: 0., 1: 1., 5: 0.},
                        1: {1: 0., 2: 2., 5: 0.},
                        2: {2: 0., 3: 3., 5: 0.},
                        3: {3: 0., 4: 4., 5: 0.},
                        4: {4: 0.},
                        5: {5: 0.}}

        self._rm_state = None
        self._past_rm_state = None
        self._state = None
        self._dt = 0.01
        self._num_step = 0
        self._viewer = Viewer(8, 8, background=(255, 255, 255))

    def _discretize_context(self, context):
        p_door1 = round((context[0]+4.)/8*40)
        p_door2 = round((context[1]+4.)/8*40)
        p_chest = round((context[2]+4.)/8*40)
        p_goal = round((context[3]+4.)/8*40)

        return np.array([p_door1, p_door2, p_chest, p_goal])

    def _update_env_config(self):
        context_discrete = self._discretize_context(self.context)
        self._doors = np.array([[context_discrete[0], 30, self._door_widths[0]],
                                [context_discrete[1], 10, self._door_widths[1]]])

        self._chest = np.array([[context_discrete[2], 20.],
                                [context_discrete[2]+self._chest_shape[0], 20.-self._chest_shape[1]]])
        self._goal = np.array([context_discrete[3], 5.])

    def get_rm_transition(self):
        return self._past_rm_state, self._rm_state

    def reset(self):
        self._num_step = 0
        self._past_rm_state = None
        self._rm_state = 0
        self._state = np.array([20., 35., 0.]) if self.product_cmdp else np.array([20., 35.])
        _state = np.copy(self._state)
        _state[:2] = (_state[:2]/40.)*8.-4
        if self.product_cmdp and self.rm_state_onehot:
            _state = np.concatenate((_state[:2], np.zeros(self.rm_info["num_states"])))
            _state[2+self._rm_state] = 1.
        return _state

    def _step_internal(self, state, action, rm_state):
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        act = round(action*3.)
        new_state = np.copy(state[:2])
        if act == 0.:  # UP
            new_state[1] = min(new_state[1]+1, 39.)
        elif act == 1.:  # RIGHT
            new_state[0] = min(new_state[0]+1, 39.)
        elif act == 2.:  # DOWN
            new_state[1] = max(new_state[1]-1, 0.)
        elif act == 3.:  # LEFT
            new_state[0] = max(new_state[0]-1, 0.)
        else:
            print("Invalid action value: "+str(act)+" from "+str(action))
            raise ValueError

        crash = False
        # Door
        if (new_state[1] == self._doors[0, 1] and np.absolute(new_state[0] - self._doors[0, 0]) > self._doors[0, 2]) or (
                new_state[1] == self._doors[1, 1] and (rm_state < 2 or (
                rm_state >= 2 and np.absolute(new_state[0] - self._doors[1, 0]) > self._doors[1, 2]))):
            crash = True
        # No crash
        else:
            pass

        new_rm_state = 0
        # Crash
        if crash:
            new_rm_state = 5
        # Passed first door
        elif rm_state == 0 and new_state[1] <= self._doors[0, 1]:
            new_rm_state = 1
        # Searching for chest
        elif rm_state == 1:
            new_rm_state = 1 # No chest yet
            if self._chest[0, 0] <= new_state[0] <= self._chest[1, 0] and \
                    self._chest[1, 1] <= new_state[1] <= self._chest[0, 1]:
                new_rm_state = 2 # Chest is found
        # Passed second door
        elif rm_state == 2:
            new_rm_state = 2
            if new_state[1] <= self._doors[1, 1]:
                new_rm_state = 3
        elif rm_state == 3:
            new_rm_state = 3
            if self._goal[0] == new_state[0] and self._goal[1] == new_state[1]:
                new_rm_state = 4

        if self.product_cmdp:
            new_state = np.concatenate((new_state, np.array([new_rm_state])))
        return new_state, crash, new_rm_state

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        self._update_env_config()
        new_state, crash, new_rm_state = self._step_internal(self._state, action, self._rm_state)

        reward = self.rewards[self._rm_state][new_rm_state]

        if new_rm_state == self.rm_info["goal"]:
            crash = True

        self._num_step += 1
        self._state = np.copy(new_state)
        self._past_rm_state = copy.deepcopy(self._rm_state)
        self._rm_state = copy.deepcopy(new_rm_state)

        info = {"success": self._rm_state == self.rm_info["goal"]}

        new_state[:2] = (new_state[:2]/40.)*8.-4
        if self.product_cmdp and self.rm_state_onehot:
            new_state = np.concatenate((new_state[:2], np.zeros(self.rm_info["sink"] + 1)))
            new_state[2+self._rm_state] = 1.
        return new_state, reward, crash, info