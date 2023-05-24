import numpy as np
from gym import Env
from .contextual_two_door_discrete import ContextualTwoDoorDiscrete


class ContextualTwoDoorDiscrete2D(Env):

    def __init__(self, context=np.array([0., 0.])):
        self._chest_pos = -2.
        self._goal_pos = 0.
        self.env = ContextualTwoDoorDiscrete(context=np.array([context[0], context[1], self._chest_pos, self._goal_pos]),
                                             door_widths=[2., 2.])
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.env.rm_transition_context_map = {(0, 0): [],
                                              (0, 1): [0],
                                              (0, 5): [0],
                                              (1, 1): [0],
                                              (1, 2): [],
                                              (1, 5): [0],
                                              (2, 2): [0],
                                              (2, 3): [1],
                                              (2, 5): [0, 1],
                                              (3, 3): [0, 1],
                                              (3, 4): [],
                                              (3, 5): [0, 1]}

    def set_context(self, context):
        self.env.context = np.array([context[0], context[1], self._chest_pos, self._goal_pos])

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)

    def get_rm_info(self):
        return self.env.rm_info

    def get_rm_transition_context_map(self):
        return self.env.rm_transition_context_map

    def get_rm_transition(self):
        return self.env.get_rm_transition()
