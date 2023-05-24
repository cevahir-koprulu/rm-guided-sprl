import os
import gym
from .contextual_fetch_push_and_play import ContextualFetchPushAndPlay

# Ensure we get the path separator correct on windows
GYM_PATH = os.path.abspath(os.path.join(gym.__file__, os.pardir))
MODEL_XML_PATH = os.path.join(GYM_PATH, "envs", "robotics", "assets", "fetch", "push.xml")


class ContextualFetchPushAndPlay4D(gym.Env):
    def __init__(self, context=None):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.25, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
        }
        self.env = ContextualFetchPushAndPlay(
            MODEL_XML_PATH,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_offset=0.0,
            obj_range=0., #0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            context=None, 
            product_cmdp=False,
            rm_state_onehot=True
        )
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = context

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
