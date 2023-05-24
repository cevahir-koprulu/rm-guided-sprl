import os
import gym
import torch.nn
import numpy as np
import scipy
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacher, SelfPacedTeacherV2, SelfPacedWrapper, \
    CurrOT, RMguidedSelfPacedTeacher, RMguidedSelfPacedTeacherV2
from deep_sprl.teachers.spl.alpha_functions import PercentageAlphaFunction
from deep_sprl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from deep_sprl.teachers.dummy_wrapper import DummyWrapper
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.teachers.util import Subsampler 
from scipy.stats import multivariate_normal


class Swimmer2DExperiment(AbstractExperiment):
    TARGET_TYPE = "narrow"
    TARGET_MEAN = np.array([-.6, 1.6])
    TARGET_VARIANCES = {
        "narrow": np.diag(np.square([4e-3, 4e-3])),
        "wide": np.diag(np.square([.5, .5])),
    }

    PCMDP = True

    LOWER_CONTEXT_BOUNDS = np.array([-.6, 1.])
    UPPER_CONTEXT_BOUNDS = np.array([0., 1.6])

    def target_log_likelihood(self, cs):
        return multivariate_normal.logpdf(cs, self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE])

    def target_sampler(self, n, rng=None):
        if rng is None:
            rng = np.random

        return rng.multivariate_normal(self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE], size=n)

    INITIAL_MEAN = np.array([0., 1.])
    INITIAL_VARIANCE = np.diag(np.square([.1, .1]))

    STD_LOWER_BOUND = np.array([4e-3, 4e-3])
    KL_THRESHOLD = 8000.
    KL_EPS = 0.1 
    DELTA = 4.0
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 40  # 10
    ZETA = {Learner.PPO: 1.45, Learner.SAC: 1.0}
    ALPHA_OFFSET = {Learner.PPO: 20, Learner.SAC: 5}
    OFFSET = {Learner.PPO: 5, Learner.SAC: 10}
    # PERF_LB = {Learner.PPO: 0.9, Learner.SAC: 3.5}  # self_paced_v2

    NUM_ITER = 200 # 250
    STEPS_PER_ITER = 8 * 2048
    DISCOUNT_FACTOR = 0.99
    LAM = 0.99

    AG_P_RAND = {Learner.PPO: 0.3, Learner.SAC: 0.3}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: 200}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: 1000}

    GG_NOISE_LEVEL = {Learner.PPO: 0.025, Learner.SAC: 0.1}
    GG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: 200}
    GG_P_OLD = {Learner.PPO: 0.1, Learner.SAC: 0.3}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, device)
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)
    
    def create_environment(self, evaluation=False):
        if self.PCMDP:
            env = gym.make("ContextualSwimmer2DProduct-v1")
        else:
            env = gym.make("ContextualSwimmer2D-v1")

        if evaluation or self.curriculum.default():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.goal_gan():
            samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(1000, 2))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.self_paced_v2() or \
            self.curriculum.rm_guided_self_paced() or self.curriculum.rm_guided_self_paced_v2() or \
            self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE, context_visible=True, 
                                   rm_guided=self.curriculum.rm_guided_self_paced() or self.curriculum.rm_guided_self_paced_v2(),
                                   use_step_based_update=self.curriculum.self_paced() or self.curriculum.rm_guided_self_paced(), 
                                   step_based_update_offset=self.OFFSET[self.learner], step_based_update_interval=self.STEPS_PER_ITER)
        elif self.curriculum.acl():
            bins = 50
            teacher = ACL(bins * bins, self.ACL_ETA, eps=self.ACL_EPS, norm_hist_len=2000)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                [bins, bins]))
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.vds():
            teacher = VDS(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.DISCOUNT_FACTOR, self.VDS_NQ,
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": self.STEPS_PER_ITER})
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR,
                                seed=self.seed,
                                verbose=0,
                                device=self.device,
                                policy_kwargs=dict(net_arch=[256, 256],
                                                   activation_fn=torch.nn.ReLU)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER,
                             gae_lambda=self.LAM,
                             max_grad_norm=None,
                             batch_size=128,
                             learning_rate=0.001,  
                             ),
                    sac=dict(learning_rate=0.001,  
                             buffer_size=500000,
                             learning_starts=10000,
                             batch_size=256,
                             train_freq=8,
                             target_entropy="auto"))

    def create_experiment(self):
        timesteps = self.NUM_ITER * self.STEPS_PER_ITER
        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(np.array([0., 0., -3., 0.])[None, :], contexts.shape[0], axis=0),
                 contexts], axis=-1)
            env.teacher.initialize_teacher(env, interface, state_provider)

        callback_params = {"learner": interface, "env_wrapper": env, 
                            "save_interval": 10,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return SelfPacedTeacher(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      use_avg_performance=False)
        elif self.curriculum.self_paced_v2():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      self.INITIAL_VARIANCE.copy(), bounds, perf_lb=self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD)
        elif self.curriculum.rm_guided_self_paced():
            alpha_fn = PercentageAlphaFunction(self.ALPHA_OFFSET[self.learner], self.ZETA[self.learner])
            return RMguidedSelfPacedTeacher(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                            self.INITIAL_VARIANCE.copy(), bounds, alpha_fn, max_kl=self.KL_EPS,
                                            std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                            use_avg_performance=False)
        elif self.curriculum.rm_guided_self_paced_v2():
            return RMguidedSelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                              self.INITIAL_VARIANCE.copy(), bounds, perf_lb=self.DELTA, max_kl=self.KL_EPS,
                                              std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, 
                                             size=(200, self.LOWER_CONTEXT_BOUNDS.shape[0]))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS, self.EP_PER_UPDATE,
                          wb_max_reuse=1)

    def get_env_name(self):
        return f"swimmer_2d_{self.TARGET_TYPE}"

    def evaluate_learner(self, path):
        num_context = 10 # None
        num_run = 1

        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.device)
        eval_path = f"{os.getcwd()}/eval_contexts/{self.get_env_name()}_eval_contexts.npy"
        if os.path.exists(eval_path):
            eval_contexts = np.load(eval_path)
            if num_context is None:
                num_context = eval_contexts.shape[0] 
        else:
            raise ValueError(f"Evaluation context file doesn't exist!")

        num_succ_eps_per_c = np.zeros((num_context, 1))
        for i in range(num_context):
            context = eval_contexts[i, :]
            for j in range(num_run):
                self.eval_env.set_context(context)
                obs = self.vec_eval_env.reset()
                done = False
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                if infos[0]["success"]:
                    num_succ_eps_per_c[i, 0] += 1./num_run

        print(f"Successful Eps: {100*np.mean(num_succ_eps_per_c)}%")
        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               np.exp(self.target_log_likelihood(eval_contexts[:num_context, :])), num_succ_eps_per_c