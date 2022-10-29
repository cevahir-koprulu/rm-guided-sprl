import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy.stats import norm
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution


def get_results(base_dir, iterations, get_success=False):
    expected = []
    success = []

    for iteration in iterations:
        perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            disc_rewards = results[:, 1]
            expected.append(np.mean(disc_rewards))

            if get_success:
                successful_eps = results[:, -1]
                success.append(np.mean(successful_eps))
        else:
            print(f"No evaluation data found: {os.path.join(base_dir, 'iteration-0', 'performance.npy')}")
            expected = []
            success = []
            break
    return expected, success


def get_dist_stats(base_dir, iterations, context_dim=2):
    dist_stats = []
    for iteration in iterations:
        dist_path = os.path.join(base_dir, f"iteration-{iteration}", "context_dist.npy")
        dist = GaussianTorchDistribution.from_weights(context_dim, np.load(dist_path))
        stats = []
        for c_dim in range(context_dim):
            stats.append(dist.mean()[c_dim])
        for c_dim in range(context_dim):
            stats.append(dist.covariance_matrix()[c_dim, c_dim])

        dist_stats.append(stats)
    dist_stats = np.array(dist_stats)
    return dist_stats


def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, plot_success, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    num_iters = setting["num_iters"]
    context_dim = setting["context_dim"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    grid_shape = setting["grid_shape"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    axes_info = setting["axes_info"]

    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(grid_shape[0], grid_shape[1], figure=fig)
    axes = []
    for row_no in range(grid_shape[0]):
        for col_no in range(grid_shape[1]):
            axes.append(fig.add_subplot(gs[row_no, col_no]))

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]

        expected = []
        success = []
        dist_stats = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed, success_seed = get_results(
                base_dir=base_dir,
                iterations=iterations,
                get_success=plot_success,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)
            if algorithm[:8] != "goal_gan" and algorithm[:7] != "default":
                dist_stats_seed = get_dist_stats(
                    base_dir=base_dir,
                    iterations=iterations,
                    context_dim=context_dim)
                dist_stats.append(dist_stats_seed)
            if plot_success:
                success.append(success_seed)

        expected = np.array(expected)
        expected_mid = np.median(expected, axis=0)
        expected_low = np.min(expected, axis=0)
        expected_high = np.max(expected, axis=0)
        expected_qlow = np.quantile(expected, 0.25, axis=0)
        expected_qhigh = np.quantile(expected, 0.75, axis=0)


        if algorithm[:8] != "goal_gan" and algorithm[:7] != "default":
            dist_stats = np.array(dist_stats)
            dist_stats = np.swapaxes(dist_stats, 1, 2)
            dist_stats = np.swapaxes(dist_stats, 0, 1)
            dist_stats_mid = np.median(dist_stats, axis=1)
            dist_stats_low = np.min(dist_stats, axis=1)
            dist_stats_high = np.max(dist_stats, axis=1)

            for ax_i in range(context_dim):
                axes[ax_i].plot(iterations, dist_stats_mid[ax_i, :], color=color, label=f"{label}-mean", marker=".")
                axes[ax_i].fill_between(iterations, dist_stats_low[ax_i, :], dist_stats_high[ax_i, :], color=color,
                                        alpha=0.5)
                axes[ax_i].plot(iterations, dist_stats_mid[context_dim+ax_i, :], color=color, label=f"{label}-var",
                                ls="--", marker="x")
                axes[ax_i].fill_between(iterations, dist_stats_low[context_dim+ax_i, :],
                                        dist_stats_high[context_dim+ax_i, :], color=color, alpha=0.5, ls="--")

        axes[context_dim].plot(iterations, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      # marker="^",
                      )
        axes[context_dim].fill_between(iterations, expected_qlow, expected_qhigh, color=color, alpha=0.4)
        axes[context_dim].fill_between(iterations, expected_low, expected_high, color=color, alpha=0.2)

        if plot_success:
            success = np.array(success)
            success_mid = np.median(success, axis=0)
            success_low = np.min(success, axis=0)
            success_high = np.max(success, axis=0)
            success_qlow = np.quantile(success, 0.25, axis=0)
            success_qhigh = np.quantile(success, 0.75, axis=0)

            axes[-1].plot(iterations, success_mid, color=color, linewidth=2.0, label=f"{label}",
                          # marker="^",
                          )
            axes[-1].fill_between(iterations, success_qlow, success_qhigh, color=color, alpha=0.4)
            axes[-1].fill_between(iterations, success_low, success_high, color=color, alpha=0.2)

    markers = [".", "x"]
    linestyles = ["-", "--"]
    labels = ["Mean", "Variance"]
    lines = [Line2D([0], [0], color="black", linestyle=linestyles[i], marker=markers[i]) for i in range(2)]
    for ax_i in range(len(axes)):
        axes[ax_i].set_ylabel(axes_info["ylabel"][ax_i], fontsize=fontsize)
        axes[ax_i].set_ylim(axes_info["ylim"][ax_i])
        axes[ax_i].grid(True)
        if ax_i < context_dim:
            axes[ax_i].legend(lines, labels, fontsize=fontsize*0.8, loc="best", framealpha=1.)
            if grid_shape[0] == 1:
                axes[ax_i].set_xlabel('Context Distribution Update', fontsize=fontsize)
    axes[-1].set_xlabel('Context Distribution Update', fontsize=fontsize)

    colors = []
    labels = []
    for cur_algo in algorithms:
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])
    markers = ["" for i in range(len(algorithms))]
    linestyles = ["-" for i in range(len(algorithms))]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(len(algorithms))]
    lgd = fig.legend(lines, labels, ncol=len(algorithms), loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    plt.savefig(f"{Path(os.getcwd()).parent}/figures/{env}_{figname}{figname_extra}.pdf", dpi=500,
                bbox_inches='tight', bbox_extra_artists=(lgd,))


def main():
    base_log_dir = f"{Path(os.getcwd()).parent}/logs"
    num_updates_per_iteration = 5
    seeds = ["1", "2", "3", "4", "5"]
    env = "half_cheetah_3d_narrow"
    # env = "two_door_discrete_2d_wide"
    # env = "two_door_discrete_4d_narrow"
    figname_extra = "" 
    plot_success = False

    algorithms = {
        #############################
        ## HalfCheetah-v3 & Narrow ##
        #############################
        "half_cheetah_3d_narrow": {
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "sac_GG_FIT_RATE=100_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.3_LR=0.001_ARCH=256_RBS=250000",
                "color": "gold",
            },
            "Default": {
                "algorithm": "default",
                "label": "Default",
                "model": "sac_LR=0.001_ARCH=256_RBS=250000",
                "color": "cyan",
            },
            "Default(P)": {
                "algorithm": "default",
                "label": "Default*",
                "model": "sac_LR=0.001_ARCH=256_RBS=250000_PRODUCTCMDP",
                "color": "magenta",
            },
            "SPRL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "sac_ALPHA_OFFSET=0_MAX_KL=0.05_OFFSET=80_ZETA=4.0_LR=0.001_ARCH=256_RBS=250000_TRUEREWARDS",
                "color": "green",
            },
            "Intermediate": {
                "algorithm": "self_paced",
                "label": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=0_MAX_KL=0.05_OFFSET=80_ZETA=4.0_LR=0.001_ARCH=256_RBS=250000_TRUEREWARDS_PRODUCTCMDP",
                "color": "red",
            },
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=0_MAX_KL=0.05_OFFSET=80_ZETA=1.0_LR=0.001_ARCH=256_RBS=250000_TRUEREWARDS_PRODUCTCMDP",
                "color": "blue",
            },
        },

        ########################
        ## Two-door 2D & Wide ##
        ########################
        "two_door_discrete_2d_wide": {
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "sac_GG_FIT_RATE=100_GG_NOISE_LEVEL=0.05_GG_P_OLD=0.2_LR=0.0003_ARCH=256_RBS=60000",
                "color": "gold",
            },
            "Default": {
                "algorithm": "default",
                "label": "Default",
                "model": "sac_LR=0.0003_ARCH=256_RBS=60000",
                "color": "cyan",
            },
            "Default(P)": {
                "algorithm": "default",
                "label": "Default*",
                "model": "sac_LR=0.0003_ARCH=256_RBS=60000_PRODUCTCMDP",
                "color": "magenta",
            },
            "SPRL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=1.2_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS",
                "color": "green",
            },
            "Intermediate": {
                "algorithm": "self_paced",
                "label": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=1.2_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS_PRODUCTCMDP",
                "color": "red",
            },
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=0.96_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS_PRODUCTCMDP",
                "color": "blue",
            },
        },

        ##########################
        ## Two-door 4D & Narrow ##
        ##########################
        "two_door_discrete_2d_wide": {
            "GoalGAN": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "sac_GG_FIT_RATE=100_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.3_LR=0.0003_ARCH=64_RBS=60000",
                "color": "gold",
            },
            "Default": {
                "algorithm": "default",
                "label": "Default",
                "model": "sac_LR=0.0003_ARCH=64_RBS=60000",
                "color": "cyan",
            },
            "Default(P)": {
                "algorithm": "default",
                "label": "Default*",
                "model": "sac_LR=0.0003_ARCH=64_RBS=60000_PRODUCTCMDP",
                "color": "magenta",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "sac_ALPHA_OFFSET=25_MAX_KL=0.05_OFFSET=5_ZETA=1.2_LR=0.0003_ARCH=64_RBS=60000",
                "color": "green",
            },
            "Intermediate": {
                "algorithm": "self_paced",
                "label": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=25_MAX_KL=0.05_OFFSET=5_ZETA=1.2_LR=0.0003_ARCH=64_RBS=60000_PRODUCTCMDP",
                "color": "red",
            },
            "RM-guided": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=25_MAX_KL=0.05_OFFSET=5_ZETA=1.0_LR=0.0003_ARCH=64_RBS=60000_PRODUCTCMDP",
                "color": "blue",
            },
        }
    }

    settings = {
        "two_door_discrete_2d_wide":
            {
                "context_dim": 2,
                "num_iters": 200,
                "fontsize": 12,
                "figsize": (5 * (3+1*plot_success), 2.5),
                "grid_shape": (1, (3+1*plot_success)),
                "bbox_to_anchor": (.5, 1.25),
                "axes_info": {
                    "ylabel": ['Param-1: Door 1 Position',
                               'Param-2: Door 2 Position',
                               'Expected Discounted Return',
                                'Successful Episodes',
                                ],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             [-.1, 4.],
                             [-0.05, 1.05]],
                    },
            },

        "two_door_discrete_4d_narrow":
            {
                "context_dim": 4,
                "num_iters": 200,
                "fontsize": 12,
                "figsize": (5, 2.1*(5+1*plot_success)),
                "grid_shape": ((5+1*plot_success), 1),
                "bbox_to_anchor": (.5, 1.07),
                "axes_info": {
                    "ylabel": ['Param-1: Door 1 Position',
                               'Param-2: Door 2 Position',
                               'Param-3: Box Position',
                               'Param-4: Goal Position',
                               'Expected Discounted Return',
                               "Successful Episode"],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             [-2.2, 1],
                             [-2.2, 1],
                             [-.1, 4.5],
                             [-0.1, 1.1]],
                }
            },

        "half_cheetah_3d_narrow":
            {
                "context_dim": 3,
                "num_iters": 200,
                "fontsize": 8,
                "figsize": (5, 2.1*(4+1*plot_success)),
                "grid_shape": (4+1*plot_success, 1),
                "bbox_to_anchor": (.5, 1.05),
                "axes_info": {
                    "ylabel": ['Param-1: Flag 1',
                               'Param-2: Flag 2',
                               'Param-3: Flag 3',
                               'Expected Discounted Return',
                               'Successful Episodes',
                               ],
                    "ylim": [[-0.1, 4.1],
                             [-0.1, 7.1],
                             [-1., 11.],
                             [-50., 550.],
                             [-0.05, 1.05]],
                },
            },
    }

    plot_results(
        base_log_dir=base_log_dir,
        num_updates_per_iteration=num_updates_per_iteration,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms[env],
        plot_success=plot_success,
        figname_extra=figname_extra,
        )


if __name__ == "__main__":
    main()