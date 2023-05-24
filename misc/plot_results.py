import os
import sys
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_results(base_dir, iterations, get_success=False):
    expected = []
    success = []
    perf_c = []
    success_c = []
    for iteration in iterations:
        perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            disc_rewards = results[:, 1]
            expected.append(np.mean(disc_rewards))
            perf_c.append(disc_rewards)

            if get_success:
                successful_eps = results[:, -1]
                success.append(np.mean(successful_eps))
                success_c.append(successful_eps)

        else:
            print(f"No evaluation data found: {perf_file}")
            expected = []
            success = []
            break
    return expected, success, perf_c, success_c


def get_dist_stats(base_dir, iterations, context_dim=2):
    dist_stats = []
    for iteration in iterations:
        dist_path = os.path.join(base_dir, f"iteration-{iteration}", "teacher.npy")
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
    steps_per_iter = setting["steps_per_iter"]
    context_dim = setting["context_dim"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    grid_shape = setting["grid_shape"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    axes_info = setting["axes_info"]
    hist_bar_num = setting["hist_bar_num"]

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

        expected = []
        success = []
        dist_stats = []
        perf_c = []
        success_c = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed, success_seed, perf_c_seed, success_c_seed = get_results(
                base_dir=base_dir,
                iterations=iterations,
                get_success=plot_success,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)
            perf_c.append(perf_c_seed)
            success_c.append(success_c_seed)
            if "self_paced" in algorithm:
            # if False:
                dist_stats_seed = get_dist_stats(
                    base_dir=base_dir,
                    iterations=iterations,
                    context_dim=context_dim)
                dist_stats.append(dist_stats_seed)
            if plot_success:
                success.append(success_seed)

        expected = np.array(expected)
        expected_mid = np.median(expected, axis=0)
        expected_qlow = np.quantile(expected, 0.25, axis=0)
        expected_qhigh = np.quantile(expected, 0.75, axis=0)
        expected_low = np.min(expected, axis=0)
        expected_high = np.max(expected, axis=0)
        # expected_mid = np.mean(expected, axis=0)
        # expected_std = np.std(expected, axis=0)
        # expected_qlow = expected_mid-expected_std
        # expected_qhigh = expected_mid+expected_std

        # perf_c = np.moveaxis(np.array(perf_c), 0, -1)
        # perf_c_ave = np.mean(perf_c, axis=2)

        # success_c = np.moveaxis(np.array(success_c), 0, -1)
        # success_c_ave = np.mean(success_c, axis=2)

        if "self_paced" in algorithm:
        # if False:
            dist_stats = np.array(dist_stats)
            dist_stats = np.swapaxes(dist_stats, 1, 2)
            dist_stats = np.swapaxes(dist_stats, 0, 1)
            dist_stats_mid = np.median(dist_stats, axis=1)
            # dist_stats_low = np.min(dist_stats, axis=1)
            # dist_stats_high = np.max(dist_stats, axis=1)
            dist_stats_low = np.quantile(dist_stats, 0.25, axis=1)
            dist_stats_high = np.quantile(dist_stats, 0.75, axis=1)

            # print(dist_stats[:context_dim, :])

            for ax_i in range(context_dim):
                axes[ax_i].plot(iterations_step, dist_stats_mid[ax_i, :], color=color, label=f"{label}-mean", marker=".")
                axes[ax_i].fill_between(iterations_step, dist_stats_low[ax_i, :], dist_stats_high[ax_i, :], color=color,
                                        alpha=0.5)
                axes[ax_i].plot(iterations_step, dist_stats_mid[context_dim+ax_i, :], color=color, label=f"{label}-var",
                                ls="--", marker="x")
                axes[ax_i].fill_between(iterations_step, dist_stats_low[context_dim+ax_i, :],
                                        dist_stats_high[context_dim+ax_i, :], color=color, alpha=0.5, ls="--")

        axes[context_dim].plot(iterations_step, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      marker="^",
                      )
        axes[context_dim].fill_between(iterations_step, expected_qlow, expected_qhigh, color=color, alpha=0.4)
        axes[context_dim].fill_between(iterations_step, expected_low, expected_high, color=color, alpha=0.2)

        # axes[context_dim + 1].hist(perf_c_ave[-1, :], density=True, bins="auto", edgecolor="white", color=color, alpha=0.4)
        # perf_c_p = ([], [])
        # perf_interval = axes_info["ylim"][context_dim][1] / hist_bar_num
        # perf_c_ave_final = perf_c_ave[-1, :]
        # for b in range(hist_bar_num):
        #     num_c = np.sum((perf_interval*b  <= perf_c_ave_final) & (perf_c_ave_final < (b+1)*perf_interval))
        #     perf_c_p[0].append(perf_interval*b+perf_interval/2)
        #     perf_c_p[1].append(num_c / perf_c_ave_final.shape[0])
        # axes[context_dim + 1].bar(perf_c_p[0], perf_c_p[1], edgecolor="white", width=perf_interval, color=color, alpha=0.4)

        if plot_success:
            success = np.array(success)
            success_mid = np.median(success, axis=0)
            success_low = np.min(success, axis=0)
            success_high = np.max(success, axis=0)
            success_qlow = np.quantile(success, 0.25, axis=0)
            success_qhigh = np.quantile(success, 0.75, axis=0)

            axes[-1].plot(iterations_step, success_mid, color=color, linewidth=2.0, label=f"{label}",
                          # marker="^",
                          )
            axes[-1].fill_between(iterations_step, success_qlow, success_qhigh, color=color, alpha=0.4)
            axes[-1].fill_between(iterations_step, success_low, success_high, color=color, alpha=0.2)

            # success_c_p = ([], [])
            # success_interval = 1. / hist_bar_num
            # success_c_ave_final = success_c_ave[-1, :]
            # for b in range(hist_bar_num):
            #     num_c = np.sum((success_interval*b <= success_c_ave_final) & (success_c_ave_final < (b+1)*success_interval))
            #     success_c_p[0].append(success_interval*b+success_interval/2)
            #     success_c_p[1].append(num_c / success_c_ave_final.shape[0])
            # axes[-1].bar(success_c_p[0], success_c_p[1], edgecolor="white", width=success_interval, color=color, alpha=0.4)

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
                # axes[ax_i].set_xlabel('Context Distribution Update', fontsize=fontsize)
                axes[ax_i].set_xlabel('Number of environment interactions', fontsize=fontsize)
    # axes[-1].set_xlabel('Context Distribution Update', fontsize=fontsize)
    axes[-1].set_xlabel('Number of environment interactions', fontsize=fontsize)

    colors = []
    labels = []
    num_alg = 0
    for cur_algo in algorithms:
        num_alg += 1
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=num_alg, loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    fig_path = os.path.join(Path(os.getcwd()).parent, "figures")
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    plt.savefig( os.path.join(fig_path, f"{env}_{figname}{figname_extra}.pdf"), dpi=500,
                bbox_inches='tight', bbox_extra_artists=(lgd,))


def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent,"logs")
    num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(1, 16)]
    # seeds = [str(i) for i in range(1, 11)]
    seeds = [2,3,4,6,7,9,10]
    target_type = "narrow"
    env = f"swimmer_2d_{target_type}"
    figname_extra = f"_TARGET=-0.6_16_seeds={seeds}"
    # target_type = "narrow"
    # env = f"half_cheetah_3d_{target_type}"
    # figname_extra = f"_ARCH=256_2_BUFFER=500000_ITER=250_seeds={seeds}_FINAL"
    # target_type = "wide"
    # env = f"two_door_discrete_2d_{target_type}"
    # figname_extra = "_bar_15seeds_BUFFER=150000_NUM_ITER=350"
    plot_success = True

    algorithms = {
        "two_door_discrete_2d_wide": {
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=0.96_PCMDP=True",
                "color": "blue",
                "aux_color": "",
            },
        },

        "half_cheetah_3d_narrow": {
            "RM-guided SPRL": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=1.0_PCMDP=True",
                "color": "blue",
                "aux_color": "",
            },
        },

        "swimmer_2d_narrow": {
            # "RM-guided SPRL_KL=0.01": {
            #     "algorithm": "rm_guided_self_paced",
            #     "label": "RM-guided SPRL_KL=0.01",
            #     "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.01_OFFSET=10_ZETA=1.0_PCMDP=True",
            #     "color": "blue",
            #     "aux_color": "",
            # },
            # "RM-guided SPRL_KL=0.05": {
            #     "algorithm": "rm_guided_self_paced",
            #     "label": "RM-guided SPRL_KL=0.05",
            #     "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.05_OFFSET=10_ZETA=1.0_PCMDP=True",
            #     "color": "green",
            #     "aux_color": "",
            # # },
            "RM-guided SPRL_KL=0.1": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL_KL=0.1",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=1.0_PCMDP=True",
                "color": "magenta",
                "aux_color": "",
            },
            "Default": {
                "algorithm": "default",
                "label": "Default",
                "model": "sac_PCMDP=True",
                "color": "red",
                "aux_color": "",
            },
        },
    }

    settings = {
        "two_door_discrete_2d":
            {
                "context_dim": 2,
                "num_iters": 350,
                "steps_per_iter": 16384,
                "fontsize": 12,
                "hist_bar_num": 10,
                "figsize": (5 * 2, 2.5 * 2),
                "grid_shape": (2, 2),
                "bbox_to_anchor": (.5, 1.25),
                "axes_info": {
                    "ylabel": ['Param-1: Door position',
                               'Param-2: Door width',
                               'Expected discounted return',
                               'Expected rate of success',
                               ],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             [-.1, 4.],
                             [-0.05, 1.05]],
                    },
            },

        "half_cheetah_3d":
            {
                "context_dim": 3,
                "num_iters": 250,
                "steps_per_iter": 16384,
                "fontsize": 8,
                "hist_bar_num": 10,
                "figsize": (5, 2.5*5),
                "grid_shape": (5, 1),
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

        "swimmer_2d":
            {
                "context_dim": 2,
                "num_iters": 200,
                "steps_per_iter": 16384,
                "fontsize": 8,
                "hist_bar_num": 10,
                "figsize": (5*2, 2.5*2),
                "grid_shape": (2, 2),
                "bbox_to_anchor": (.5, 1.05),
                "axes_info": {
                    "ylabel": ['Param-1: Flag 1',
                               'Param-2: Flag 2',
                               'Expected Discounted Return',
                               'Successful Episodes',
                               ],
                    "ylim": [[-1.0, 0.5],
                             [-0.5, 2.0],
                            # [-0.05, 1.05],
                             [-10., 500.],
                             [-0.05, 1.05]],
                },
            },

    }

    plot_results(
        base_log_dir=base_log_dir,
        num_updates_per_iteration=num_updates_per_iteration,
        seeds=seeds,
        env=env,
        setting=settings[env[:-len(target_type)-1]],
        algorithms=algorithms[env],
        plot_success=plot_success,
        figname_extra=figname_extra,
        )


if __name__ == "__main__":
    main()
