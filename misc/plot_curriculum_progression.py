import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    context_dim = setting["context_dim"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    grid_shape = setting["grid_shape"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    axes_info = setting["axes_info"]

    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

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
        print(algorithm)

        dist_stats = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            dist_stats_seed = get_dist_stats(
                    base_dir=base_dir,
                    iterations=iterations,
                    context_dim=context_dim)
            if len(dist_stats_seed) == 0:
                continue

            dist_stats.append(dist_stats_seed)

        dist_stats = np.array(dist_stats)
        dist_stats = np.swapaxes(dist_stats, 1, 2)
        dist_stats = np.swapaxes(dist_stats, 0, 1)
        print(dist_stats.shape)
        dist_stats_mid = np.median(dist_stats, axis=1)
        # dist_stats_low = np.min(dist_stats, axis=1)
        # dist_stats_high = np.max(dist_stats, axis=1)
        dist_stats_low = np.quantile(dist_stats, 0.25, axis=1)
        dist_stats_high = np.quantile(dist_stats, 0.75, axis=1)

        print(f"mean: {dist_stats[0:3, :, -1].T}")
        print(f"var : {dist_stats[3:6, :, -1].T}")

        for ax_i in range(context_dim):
            

            axes[ax_i].plot(iterations_step, dist_stats_mid[ax_i, :], color=color, label=f"{label}-mean", marker="o")
            axes[ax_i].fill_between(iterations_step, dist_stats_low[ax_i, :], dist_stats_high[ax_i, :], color=color,
                                    alpha=0.5)
            axes[ax_i].plot(iterations_step, dist_stats_mid[context_dim+ax_i, :], color=color, label=f"{label}-var",
                            ls="--", marker="x")
            axes[ax_i].fill_between(iterations_step, dist_stats_low[context_dim+ax_i, :],
                                    dist_stats_high[context_dim+ax_i, :], color=color, alpha=0.5, ls="--")


    markers = ["o", "x"]
    linestyles = ["-", "--"]
    labels = ["Mean", "Variance"]
    lines = [Line2D([0], [0], color="black", linestyle=linestyles[i], marker=markers[i]) for i in range(2)]
    for ax_i in range(len(axes)):
        axes[ax_i].set_ylabel(axes_info["ylabel"][ax_i], fontsize=fontsize)
        axes[ax_i].set_ylim(axes_info["ylim"][ax_i])
        axes[ax_i].grid(True)
        axes[ax_i].legend(lines, labels, fontsize=fontsize*0.8, loc="best", framealpha=1.)
        if grid_shape[0] == 1:
            axes[ax_i].set_xlabel('Number of interactions', fontsize=fontsize)
            axes[ax_i].set_xlim([iterations_step[0], iterations_step[-1]])
            axes[ax_i].ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
        else:
            axes[ax_i].set_xlim([iterations_step[0], iterations_step[-1]])
            if ax_i < len(axes)-1:
                axes[ax_i].set_xticklabels([])


    axes[-1].set_xlabel('Number of interactions', fontsize=fontsize)
    axes[-1].set_xlim([iterations_step[0], iterations_step[-1]])
    axes[-1].ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
    
    colors = []
    labels = []
    num_alg = len(algorithms)
    for cur_algo in algorithms:
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=math.ceil(num_alg/2), loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)
    # lgd = fig.legend(lines, labels, ncol=num_alg, loc="upper center", bbox_to_anchor=bbox_to_anchor,
    #                  fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    print(f"{Path(os.getcwd()).parent}/figures/{env}_{figname}{figname_extra}.png")
    plt.savefig(f"{Path(os.getcwd()).parent}/figures/{env}_{figname}{figname_extra}.pdf", dpi=500,
                bbox_inches='tight',
                bbox_extra_artists=(lgd,))


def main():
    base_log_dir = f"{Path(os.getcwd()).parent}/logs"
    num_updates_per_iteration = 10
    seeds = [str(i) for i in range(1, 11)]
    # seeds = [2, 3, 4, 6, 7, 9, 10]
    # env = "two_door_discrete_2d_wide"
    # env = "swimmer_2d_narrow"
    env = "half_cheetah_3d_narrow"
    figname_extra = "_curriculum_progression"

    algorithms = {
        "two_door_discrete_2d_wide": {
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=0.96_PCMDP=True",
                "color": "blue",
            },
            "Intermediate SPRL": {
                "algorithm": "self_paced",
                "label": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=1.2_PCMDP=True",
                "color": "red",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=1.2_PCMDP=False",
                "color": "green",
            },
        },
        "swimmer_2d_narrow": {
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=1.0_PCMDP=True",
                "color": "blue",
            },
            "Intermediate SPRL": {
                "algorithm": "self_paced",
                "label": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=4.0_PCMDP=True",
                "color": "red",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=4.0_PCMDP=False",
                "color": "green",
            },
        },
        "half_cheetah_3d_narrow": {
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=1.0_PCMDP=True",
                "color": "blue",
            },
            "Intermediate SPRL": {
                "algorithm": "self_paced",
                "label": "Intermediate SPRL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=4.0_PCMDP=True",
                "color": "red",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=4.0_PCMDP=False",
                "color": "green",
            },
        },
    }

    settings = {
        "two_door_discrete_2d_wide":
            {
                "num_iters": 350,
                "steps_per_iter": 16384,
                "context_dim": 2,
                "fontsize": 14,
                "figsize": (5, 2.5*2),
                "grid_shape": (2, 1),
                "bbox_to_anchor": (.5, 1.15),
                "axes_info": {
                    "ylabel": ['Param-1: Door position',
                               'Param-2: Door width',
                               ],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             ],
                    },
            },
        "swimmer_2d_narrow":
            {
                "num_iters": 200,
                "steps_per_iter": 16384,
                "context_dim": 2,
                "fontsize": 14,
                "figsize": (5, 2.5*2),
                "grid_shape": (2, 1),
                "bbox_to_anchor": (.5, 1.15),
                "axes_info": {
                    "ylabel": ['Param-1: Flag 1',
                               'Param-2: Flag 2',
                               ],
                    "ylim": [[-0.65, 0.05],
                             [0.95, 1.65],
                             ],
                },
            },
        "half_cheetah_3d_narrow":
            {
                "num_iters": 250,
                "steps_per_iter": 16384,
                "context_dim": 3,
                "fontsize": 14,
                "figsize": (5, 2.5*3),
                "grid_shape": (3, 1),
                "bbox_to_anchor": (.5, 1.1),
                "axes_info": {
                    "ylabel": ['Param-1: Flag 1',
                               'Param-2: Flag 2',
                               'Param-3: Flag 3',
                               ],
                    "ylim": [[-0.1, 4.1],
                             [-0.1, 7.1],
                             [-1., 11.],
                             ],
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
        figname_extra=figname_extra,
        )


if __name__ == "__main__":
    main()
