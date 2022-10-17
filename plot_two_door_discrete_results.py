import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution


def get_results(base_dir, num_contexts, num_updates):
    if os.path.exists(os.path.join(base_dir, "performance.npy")):
        val_results = np.load(os.path.join(base_dir, "performance.npy"))
        expected = []
        for it in range(num_updates):
            results = val_results[val_results[:, 0] == it * 5]
            disc_rewards = results[:, 1]
            mean = np.mean(disc_rewards)
            expected.append(mean)
    elif os.path.exists(os.path.join(base_dir, "performance.pkl")):
        with open(os.path.join(base_dir, "performance.pkl"), "rb") as f:
            expected = np.array(pickle.load(f))[:num_updates]
    else:
        raise Exception("No evaluation data found: "+base_dir)
    return expected


def get_dist_stats(base_dir, num_updates, context_dim=2):
    dist_stats = []
    for iteration in range(num_updates):
        dist = os.path.join(base_dir, "iteration-%d" % int(iteration*5), "context_dist.npy")
        dist = GaussianTorchDistribution.from_weights(context_dim, np.load(dist))
        stats = []
        for c_dim in range(context_dim):
            stats.append(dist.mean()[c_dim])
        for c_dim in range(context_dim):
            stats.append(dist.covariance_matrix()[c_dim, c_dim])

        dist_stats.append(stats)
    dist_stats = np.array(dist_stats)
    return dist_stats


# def plot_results(seeds, directory, settings, figname, num_iters, num_contexts, context_dim, axes_info):
def plot_results(base_log_dir, num_contexts, seeds, env, setting, algorithms):
    rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

    num_iters = setting["num_iters"]
    context_dim = setting["context_dim"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    grid_shape = setting["grid_shape"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    axes_info = setting["axes_info"]

    num_updates = int(num_iters / 5)
    iterations = np.arange(0, num_updates * 5, 5)

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
        dist_stats = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")

            expected_seed = get_results(
                base_dir=base_dir,
                num_contexts=num_contexts,
                num_updates=num_updates)
            expected.append(expected_seed)
            if cur_algo != "goal_gan":
                dist_stats_seed = get_dist_stats(
                    base_dir=base_dir,
                    num_updates=num_updates,
                    context_dim=context_dim)
                dist_stats.append(dist_stats_seed)

        expected = np.array(expected)
        expected_mid = np.median(expected, axis=0)
        expected_low = np.min(expected, axis=0)
        expected_high = np.max(expected, axis=0)
        expected_qlow = np.quantile(expected, 0.25, axis=0)
        expected_qhigh = np.quantile(expected, 0.75, axis=0)

        if algorithm != "goal_gan":
            dist_stats = np.array(dist_stats)
            dist_stats = np.swapaxes(dist_stats, 1, 2)
            dist_stats = np.swapaxes(dist_stats, 0, 1)
            dist_stats_mid = np.median(dist_stats, axis=1)
            dist_stats_low = np.min(dist_stats, axis=1)
            dist_stats_high = np.max(dist_stats, axis=1)

        if algorithm != "goal_gan":
            for ax_i in range(context_dim):
                axes[ax_i].plot(iterations, dist_stats_mid[ax_i, :], color=color, label=f"{label}-mean", marker=".")
                axes[ax_i].fill_between(iterations, dist_stats_low[ax_i, :], dist_stats_high[ax_i, :], color=color,
                                        alpha=0.5)
                axes[ax_i].plot(iterations, dist_stats_mid[context_dim+ax_i, :],color=color, label=f"{label}-var",
                                ls="--", marker="x")
                axes[ax_i].fill_between(iterations, dist_stats_low[context_dim+ax_i, :],
                                        dist_stats_high[context_dim+ax_i, :], color=color, alpha=0.5, ls="--")

        axes[-1].plot(iterations, expected_mid, color=color, linewidth=2.0, label=f"{label}", marker="^")
        axes[-1].fill_between(iterations, expected_qlow, expected_qhigh, color=color, alpha=0.4)
        axes[-1].fill_between(iterations, expected_low, expected_high, color=color, alpha=0.2)

    markers = [".", "x"]
    linestyles = ["-", "--"]
    labels = ["Mean", "Variance"]
    lines = [Line2D([0], [0], color="black", linestyle=linestyles[i], marker=markers[i]) for i in range(2)]
    for ax_i in range(len(axes)):
        axes[ax_i].set_ylabel(axes_info["ylabel"][ax_i], fontsize=fontsize)
        axes[ax_i].set_ylim(axes_info["ylim"][ax_i])
        axes[ax_i].grid(True)
        if ax_i < context_dim-1:
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
    for cur_algo in algorithms:
        figname += cur_algo + "_vs_"
    plt.savefig(f"{env}_{figname[:-4]}" + ".pdf", dpi=500, bbox_inches='tight', bbox_extra_artists=(lgd,))


def main():
    base_log_dir = "logs"
    num_contexts = 100
    seeds = ["1", "2", "3", "4", "5"]
    env = "two_door_discrete_4d_narrow"

    algorithms = {
        "goal_gan": {
            "algorithm": "goal_gan",
            "label": "GoalGAN",
            "model": "sac_GG_FIT_RATE=100_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.3_PRODUCT_CMDP=False_LR=0.0003_RBS=60000",
            "color": "gold",
        },
        "self_paced": {
            "algorithm": "self_paced",
            "label": "SPDL",
            "model": "sac_ALPHA_OFFSET=25_MAX_KL=0.05_OFFSET=5_PRODUCT_CMDP=False_ZETA=1.2_LR=0.0003_RBS=60000",
            "color": "green",
        },
        "rm_guided": {
            "algorithm": "rm_guided_self_paced",
            "label": "RM-guided SPRL",
            "model": "sac_ALPHA_OFFSET=25_MAX_KL=0.05_OFFSET=5_PRODUCT_CMDP=True_ZETA=1.0_LR=0.0003_RBS=60000",
            "color": "blue",
        },
        "intermediate": {
            "algorithm": "self_paced",
            "label": "Intermediate SPRL",
            "model": "sac_ALPHA_OFFSET=25_MAX_KL=0.05_OFFSET=5_PRODUCT_CMDP=True_ZETA=1.2_LR=0.0003_RBS=60000",
            "color": "red",
        },
    }

    settings = {
        "two_door_discrete_2d_narrow":
            {
                "context_dim": 2,
                "num_iters": 200,
                "fontsize": 12,
                "figsize": (5 * 3, 2.5),
                "grid_shape": (1, 3),
                "bbox_to_anchor": (.5, 1.21),
                "axes_info": {
                    "ylabel": ['Param-1: Door 1 Position',
                               'Param-2: Door 2 Position',
                               'Expected Discounter Return'],
                    "ylim": [[-.7, 2.2],
                             [-.7, 2.2],
                             [-.1, 4.]],
                    },
            },

        "two_door_discrete_2d_wide":
            {
                "context_dim": 2,
                "num_iters": 200,
                "fontsize": 12,
                "figsize": (5 * 3, 2.5),
                "grid_shape": (1, 3),
                "bbox_to_anchor": (.5, 1.21),
                "axes_info": {
                    "ylabel": ['Param-1: Door 1 Position',
                               'Param-2: Door 2 Position',
                               'Expected Discounter Return'],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             [-.1, 4.]],
                    },
            },

        "two_door_discrete_4d_narrow":
            {
                "context_dim": 4,
                "num_iters": 200,
                "fontsize": 8,
                "figsize": (5, 2.1*5),
                "grid_shape": (5, 1),
                "bbox_to_anchor": (.5, 1.05),
                "axes_info": {
                    "ylabel": ['Param-1: Door 1 Position',
                               'Param-2: Door 2 Position',
                               'Param-3: Box Position',
                               'Param-4: Goal Position',
                               'Expected Discounter Return'],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             [-2.2, 1],
                             [-2.2, 1],
                             [-.1, 4.5]],
                }
            }
    }

    plot_results(
        base_log_dir=base_log_dir,
        num_contexts=num_contexts,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms,
        )


if __name__ == "__main__":
    main()