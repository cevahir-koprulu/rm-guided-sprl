import os
import sys
import numpy as np
from pathlib import Path
from scipy.stats import ttest_ind, sem
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution


# def get_dist_stats(base_dir, num_updates, context_dim=2):
#     dist_stats = []
#     for iteration in range(num_updates):
#         dist = os.path.join(base_dir, "iteration-%d" % int(iteration*5), "context_dist.npy")
#         dist = GaussianTorchDistribution.from_weights(context_dim, np.load(dist))
#         stats = []
#         for c_dim in range(context_dim):
#             stats.append(dist.mean()[c_dim])
#         for c_dim in range(context_dim):
#             stats.append(dist.covariance_matrix()[c_dim, c_dim])

#         dist_stats.append(stats)
#     dist_stats = np.array(dist_stats)
#     return dist_stats

def get_dist_stats(base_dir, iterations, context_dim=2):
    dist_stats = []
    for iteration in iterations:
        dist_path = os.path.join(base_dir, f"iteration-{iteration}", "teacher.npy")
        if os.path.exists(dist_path):
            dist = GaussianTorchDistribution.from_weights(context_dim, np.load(dist_path))
            stats = []
            for c_dim in range(context_dim):
                stats.append(dist.mean()[c_dim])
            for c_dim in range(context_dim):
                stats.append(dist.covariance_matrix()[c_dim, c_dim])

            dist_stats.append(stats)
        else:
            print(f"No curriculum data found: {dist_path}")
            dist_stats = []
            break
    dist_stats = np.array(dist_stats)
    return dist_stats

def calc_stat_var(dist_stats):
    return np.var(dist_stats, axis=0)


def run_welch_t_test(base_log_dir, seeds, env, setting, algorithms):
    context_dim = setting["context_dim"]
    num_iters = setting["num_iters"]
    num_updates_per_iteration = setting["num_updates_per_iteration"]

    iterations=np.arange(0, num_iters, num_updates_per_iteration, dtype=int)

    dist_stats = {}
    stat_vars = {}
    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        algo_name = algorithms[cur_algo]["name"]
        model = algorithms[cur_algo]["model"]

        print("\n"+cur_algo)

        dist_stats[algo_name] = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            dist_stats_seed = get_dist_stats(
                base_dir=base_dir,
                iterations=iterations,
                context_dim=context_dim)
            dist_stats[algo_name].append(dist_stats_seed)
        dist_stats[algo_name] = np.array(dist_stats[algo_name])
        stat_vars[algo_name] = calc_stat_var(dist_stats[algo_name])
        ave_var = np.mean(stat_vars[algo_name], axis=0)
        sem_var = sem(stat_vars[algo_name], axis=0)
        print(f"Average Variance: {ave_var}")
        print(f"SEM of Variance: {sem_var}")

        for i in range(2):
            stat_name = "Mean" if i == 0 else "Var"
            for d in range(context_dim):
                print(f"{stat_name}-{d+1}: {ave_var[i*context_dim+d]:.2e} +- {sem_var[i*context_dim+d]:.2e}")

    algo_list = list(algorithms.keys())
    for i, cur_algo_i in enumerate(algo_list[:-1]):
        for cur_algo_j in algo_list[i+1:]:
            label_i = algorithms[cur_algo_i]["name"]
            label_j = algorithms[cur_algo_j]["name"]
            upt_i = algorithms[cur_algo_i]["target_conv"]
            upt_j = algorithms[cur_algo_j]["target_conv"]
            print(f"\n{label_i} ({upt_i}) vs {label_j} ({upt_j})")
            print(f"{label_i}:{stat_vars[label_i][upt_i, :]}")
            print(f"{label_i}:{stat_vars[label_j][upt_j, :]}")
            print(ttest_ind(stat_vars[label_i][:upt_i, :],
                            stat_vars[label_j][:upt_j, :],
                            equal_var=False,
                            axis=0))


def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    num_seeds = 15
    seeds = [str(i) for i in range(1, num_seeds+1)]
    env = "two_door_discrete_2d_wide"
    # env = "swimmer_2d_narrow"
    # env = "half_cheetah_3d_narrow"

    settings = {
        "two_door_discrete_2d_wide":
            {
                "context_dim": 2,
                "num_updates_per_iteration": 5,
                "num_iters": 350,
            },
        "swimmer_2d_narrow":
            {
                "context_dim": 2,
                "num_updates_per_iteration": 10,
                "num_iters": 200,
            },
        "half_cheetah_3d_narrow": 
            {
                "context_dim": 3,
                "num_updates_per_iteration": 10,
                "num_iters": 250,
            },
    }


    algorithms = {
        "two_door_discrete_2d_wide": {
            "rm_guided": {
                "algorithm": "rm_guided_self_paced",
                "name": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=0.96_PCMDP=True",
                "target_conv": 22,  # update at which the distribution converges to the target
            },
            "intermediate": {
                "algorithm": "self_paced",
                "name": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=1.2_PCMDP=True",
                "target_conv": 36,  # update at which the distribution converges to the target
            },
            "self_paced": {
                "algorithm": "self_paced",
                "name": "SPDL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=1.2_PCMDP=False",
                "target_conv": 36,  # update at which the distribution converges to the target
            },
        },

        "swimmer_2d_narrow": {
            "rm_guided": {
                "algorithm": "rm_guided_self_paced",
                "name": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=1.0_PCMDP=True",
                "target_conv": 13,  # update at which the distribution converges to the target
            },
            "intermediate": {
                "algorithm": "self_paced",
                "name": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=4.0_PCMDP=True",
                "target_conv": 13,  # update at which the distribution converges to the target
            },
            "self_paced": {
                "algorithm": "self_paced",
                "name": "SPDL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=4.0_PCMDP=False",
                "target_conv": 19,  # update at which the distribution converges to the target
            },
        },

        "half_cheetah_3d_narrow": {
            "rm_guided": {
                "algorithm": "rm_guided_self_paced",
                "name": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=1.0_PCMDP=True",
                "target_conv": 20,  # update at which the distribution converges to the target
            },
            "intermediate": {
                "algorithm": "self_paced",
                "name": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=4.0_PCMDP=True",
                "target_conv": 24,  # update at which the distribution converges to the target
            },
        #     "self_paced": {
        #         "algorithm": "self_paced",
        #         "name": "SPDL",
        #         "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=1.2_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS",
        #         "target_conv": -,  # update at which the distribution converges to the target
        #     },
        },
    }

    run_welch_t_test(
        base_log_dir=base_log_dir,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms[env])


if __name__ == "__main__":
    main()