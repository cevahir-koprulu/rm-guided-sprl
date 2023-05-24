import numpy as np
from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class SelfPacedWrapper(BaseWrapper):

    def __init__(self, env, sp_teacher, discount_factor, context_visible, reward_from_info=False,
                 use_undiscounted_reward=False, episodes_per_update=50, 
                 rm_guided=False, 
                 use_step_based_update=False, step_based_update_offset=0, step_based_update_interval=0):
        self.use_undiscounted_reward = use_undiscounted_reward
        self.step_based_update_offset = step_based_update_offset
        self.step_based_update_interval = step_based_update_interval
        self.step_based_update_allowed = False
        self.step_based_update_counter = 0
        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible,
                             reward_from_info=reward_from_info,
                             rm_guided=rm_guided,
                             use_step_based_update=use_step_based_update)

        self.context_buffer = Buffer(3, 10000, True)
        self.episodes_per_update = episodes_per_update

    def update_distribution(self):
        __, contexts, returns = self.context_buffer.read_buffer()
        if self.use_step_based_update:
            if self.rm_guided:
                rm_transitions_buffer = self.get_rm_transitions_buffer()
                self.teacher.update_distribution(avg_performance=np.mean(returns), contexts=np.array(contexts), 
                                                 values=np.array(returns), rm_transitions_buffer=rm_transitions_buffer)
                self.reset_rm_transitions_buffer()
            else:
                self.teacher.update_distribution(avg_performance=np.mean(returns), contexts=np.array(contexts), 
                                                 values=np.array(returns))
        else:
            if self.rm_guided:
                rm_transitions_buffer = self.get_rm_transitions_buffer()
                self.teacher.update_distribution(contexts=np.array(contexts), values=np.array(returns), 
                                                 rm_transitions_buffer=rm_transitions_buffer)
                self.reset_rm_transitions_buffer()
            else:
                self.teacher.update_distribution(contexts=np.array(contexts), values=np.array(returns))

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        ret = undiscounted_reward if self.use_undiscounted_reward else discounted_reward
        self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))
        if hasattr(self.teacher, "on_rollout_end"):
                self.teacher.on_rollout_end(cur_context, ret)

        if len(self.context_buffer) >= self.episodes_per_update and not self.use_step_based_update:
            self.update_distribution()

    def step_based_update(self):
        self.step_based_update_counter += 1
        if self.step_based_update_counter >= self.step_based_update_offset*self.step_based_update_interval:
            # print(f"Step based update offset is exceeded at {self.step_based_update_counter}")
            self.step_based_update_allowed = True

        if self.step_based_update_allowed and self.step_based_update_counter >= self.step_based_update_interval:
            # print(f"Update distribution at {self.step_based_update_counter}")
            self.update_distribution()
            self.step_based_update_counter = 0

    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)

