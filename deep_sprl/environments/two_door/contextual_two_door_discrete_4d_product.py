import numpy as np
from gym import Env
from .contextual_two_door_discrete import ContextualTwoDoorDiscrete


class ContextualTwoDoorDiscrete4DProduct(Env):

    def __init__(self, context=np.array([0., 0., -2., 0.])):
        self.env = ContextualTwoDoorDiscrete(context=np.copy(context),
                                             product_cmdp=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = np.copy(context)

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