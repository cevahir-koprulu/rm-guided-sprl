import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import numpy as np
import torch
# import tensorflow as tf
# tf.get_logger().setLevel('INFO')
# tf.autograph.set_verbosity(0)
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedTeacherV2, RMguidedSelfPacedTeacher, SelfPacedWrapper
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import GaussianSampler, UniformSampler
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv


class TwoDoorDiscrete2DExperiment(AbstractExperiment):
    LOWER_CONTEXT_BOUNDS = np.array([-4, -4])
    UPPER_CONTEXT_BOUNDS = np.array([4., 4.])

    INITIAL_MEAN = np.array([0., 0.])
    INITIAL_VARIANCE = np.diag(np.square([.5, .5]))

    TARGET_MEAN = np.array([2., 2.])
    TARGET_VARIANCE_TYPES = {
        "narrow": np.diag(np.square([4e-3, 4e-3])),
        "wide": np.diag(np.square([1., 1.])),
                             }
    TARGET_TYPE = "narrow"

    DISCOUNT_FACTOR = 0.98
    STD_LOWER_BOUND = np.array([4e-3, 4e-3])
    KL_THRESHOLD = 8000.
    MAX_KL = 0.05

    ZETA = {Learner.PPO: 1.45, Learner.SAC: 1.1}
    ALPHA_OFFSET = {Learner.PPO: 20, Learner.SAC: 25}
    OFFSET = {Learner.PPO: 5, Learner.SAC: 1}
    PERF_LB = {Learner.PPO: 0.9, Learner.SAC: 3.5}  # self_paced_v2

    STEPS_PER_ITER = 8 * 2048
    NUM_CONTEXT_UPDATES = 11
    LAM = 0.99

    LEARNING_RATE = 0.0003

    # SAC Parameters
    SAC_BUFFER = 10000

    AG_P_RAND = {Learner.PPO: 0.3, Learner.SAC: 0.1}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: 500}

    GG_NOISE_LEVEL = {Learner.PPO: 0.025, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: 100}
    GG_P_OLD = {Learner.PPO: 0.1, Learner.SAC: 0.3}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, **kwargs):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, **kwargs)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        if self.use_product_cmdp:
            env = gym.make("ContextualTwoDoorDiscrete2DProduct-v1")
        else:
            env = gym.make("ContextualTwoDoorDiscrete2D-v1")

        if evaluation or self.curriculum.default():
            teacher = GaussianSampler(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE_TYPES[self.TARGET_TYPE],
                                      (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy()),
                                      seed=self.seed)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.goal_gan():
            samples = np.clip(np.random.multivariate_normal(self.INITIAL_MEAN, self.INITIAL_VARIANCE, size=1000),
                              self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.self_paced_v2() or self.curriculum.rm_guided_self_paced():
            teacher = self.create_self_paced_teacher()
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                                     seed=self.seed)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR,
                                seed=self.seed,
                                verbose=0,
                                policy_kwargs=dict(net_arch=[64, 64],
                                                   activation_fn=torch.nn.Tanh)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER,
                             gae_lam=self.LAM,
                             max_grad_norm=None,
                             batch_size=128,
                             learning_rate=self.LEARNING_RATE,  # 0.000025,
                             ),
                    sac=dict(learning_rate=self.LEARNING_RATE,
                             buffer_size=self.SAC_BUFFER,
                             learning_starts=500,
                             batch_size=64,
                             train_freq=5,
                             target_entropy="auto"))

    def create_experiment(self):
        timesteps = self.NUM_CONTEXT_UPDATES * self.STEPS_PER_ITER
        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env.teacher, SelfPacedTeacher) or isinstance(env.teacher, SelfPacedTeacherV2) or \
                isinstance(env.teacher, RMguidedSelfPacedTeacher):
            sp_teacher = env.teacher
        else:
            sp_teacher = None

        callback_params = {"learner": interface, "env_wrapper": env, "sp_teacher": sp_teacher, "n_inner_steps": 1,
                           "n_offset": self.OFFSET[self.learner], "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER if self.learner.sac() else 1}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return SelfPacedTeacher(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE_TYPES[self.TARGET_TYPE].copy(),
                                    self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy(),
                                    bounds, alpha_fn, max_kl=self.MAX_KL, std_lower_bound=self.STD_LOWER_BOUND,
                                    kl_threshold=self.KL_THRESHOLD, use_avg_performance=True,
                                    seed=self.seed)
        elif self.curriculum.self_paced_v2():
            return SelfPacedTeacherV2(self.TARGET_MEAN.copy(), self.TARGET_VARIANCE_TYPES[self.TARGET_TYPE].copy(),
                                      self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy(),
                                      bounds, self.PERF_LB[self.learner], max_kl=self.MAX_KL,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      use_avg_performance=True, seed=self.seed)
        elif self.curriculum.rm_guided_self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return RMguidedSelfPacedTeacher(self.TARGET_MEAN.copy(),
                                            self.TARGET_VARIANCE_TYPES[self.TARGET_TYPE].copy(),
                                            self.INITIAL_MEAN.copy(), self.INITIAL_VARIANCE.copy(),
                                            bounds, alpha_fn, max_kl=self.MAX_KL, std_lower_bound=self.STD_LOWER_BOUND,
                                            kl_threshold=self.KL_THRESHOLD, use_avg_performance=True,
                                            seed=self.seed)
        else:
            raise ValueError("Invalid self-paced teacher")

    def get_env_name(self):
        return "two_door_discrete_2d_"+self.TARGET_TYPE

    def evaluate_learner(self, path):
        num_context = 100
        num_run = 1

        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env)
        eval_contexts = np.load(f"{os.getcwd()}/eval_contexts/{self.get_env_name()}_eval_contexts.npy")

        if num_context is None:
            num_context = eval_contexts.shape[0]

        num_succ_eps_per_c = np.zeros((num_context, 1))
        for i in range(num_context):
            context = eval_contexts[i, :]
            for j in range(num_run):
                self.eval_env.set_current_context(context)
                obs = self.vec_eval_env.reset()
                done = False
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                if infos[0]["success"]:
                    num_succ_eps_per_c[i, 0] += 1./num_run

        print(f"Successful Eps: {100*np.mean(num_succ_eps_per_c)}%")
        disc_rewards = self.eval_env.get_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j*num_run:(j+1)*num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               self.eval_env.teacher.mean(), self.eval_env.teacher.covariance_matrix(), num_succ_eps_per_c