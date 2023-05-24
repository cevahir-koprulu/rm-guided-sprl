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

def get_results(base_dir, iterations, plot_success):
    expected = []
    for iteration in iterations:
        perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            if plot_success:
                perf = results[:, -1]
            else:
                perf = results[:, 1]
            expected.append(np.mean(perf))
        else:
            print(f"No evaluation data found: {perf_file}")
            expected = []
            break
    return expected

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

def plot_results(base_log_dir, settings, algorithms):
    fontsize = 20
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = fontsize

    fig = plt.figure(constrained_layout=True, figsize=(15, 8))
    gs = GridSpec(2, 2, figure=fig)
    axes = []
    for row_no in range(2):
        for col_no in range(2):
            axes.append(fig.add_subplot(gs[row_no, col_no]))
    

    # Curriculum for Two-door
    algorithms_two_door = algorithms["two_door_discrete_2d_wide"]
    settings_two_door = settings["two_door_discrete_2d_wide"]
    iterations_two_door = np.arange(0, settings_two_door["num_iters"], 
                           settings_two_door["num_updates_per_iteration"], dtype=int)
    iterations_step_two_door = iterations_two_door*settings_two_door["steps_per_iter"]
    context_dim_two_door = 2
    for cur_algo in algorithms_two_door:
        algorithm = algorithms_two_door[cur_algo]["algorithm"]
        label = algorithms_two_door[cur_algo]["label"]
        model = algorithms_two_door[cur_algo]["model"]
        color = algorithms_two_door[cur_algo]["color"]
        print(algorithm)

        if "self_paced" in algorithm:
            dist_stats = []
            for seed in settings_two_door["seeds"]:
                base_dir = os.path.join(base_log_dir, "two_door_discrete_2d_wide", algorithm, model, f"seed-{seed}")
                print(base_dir)
                dist_stats_seed = get_dist_stats(
                        base_dir=base_dir,
                        iterations=iterations_two_door,
                        context_dim=2)
                if len(dist_stats_seed) == 0:
                    continue

                dist_stats.append(dist_stats_seed)

            dist_stats = np.array(dist_stats)
            dist_stats = np.swapaxes(dist_stats, 1, 2)
            dist_stats = np.swapaxes(dist_stats, 0, 1)
            dist_stats_mid = np.median(dist_stats, axis=1)
            dist_stats_low = np.quantile(dist_stats, 0.25, axis=1)
            dist_stats_high = np.quantile(dist_stats, 0.75, axis=1)

            for ax_i in range(context_dim_two_door):
                axes[ax_i].plot(iterations_step_two_door, dist_stats_mid[ax_i, :], color=color, label=f"{label}-mean", marker="o")
                axes[ax_i].fill_between(iterations_step_two_door, dist_stats_low[ax_i, :], dist_stats_high[ax_i, :], color=color,
                                        alpha=0.5)
                axes[ax_i].plot(iterations_step_two_door, dist_stats_mid[context_dim_two_door+ax_i, :], color=color, label=f"{label}-var",
                                ls="--", marker="x")
                axes[ax_i].fill_between(iterations_step_two_door, dist_stats_low[context_dim_two_door+ax_i, :],
                                        dist_stats_high[context_dim_two_door+ax_i, :], color=color, alpha=0.5, ls="--")
        
        expected = []
        for seed in settings_two_door["seeds"]:
            base_dir = os.path.join(base_log_dir, "two_door_discrete_2d_wide", algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed = get_results(
                base_dir=base_dir,
                iterations=iterations_two_door,
                plot_success=True,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)

        expected = np.array(expected)
        expected_mid = np.median(expected, axis=0)
        expected_qlow = np.quantile(expected, 0.25, axis=0)
        expected_qhigh = np.quantile(expected, 0.75, axis=0)
        axes[2].plot(iterations_step_two_door, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      marker=".",
                      )
        axes[2].fill_between(iterations_step_two_door, expected_qlow, expected_qhigh, color=color, alpha=0.4)


    algorithms_half_cheetah = algorithms["half_cheetah_3d_narrow"]
    settings_half_cheetah = settings["half_cheetah_3d_narrow"]
    iterations_half_cheetah = np.arange(0, settings_half_cheetah["num_iters"], 
                           settings_half_cheetah["num_updates_per_iteration"], dtype=int)
    iterations_step_half_cheetah = iterations_half_cheetah*settings_half_cheetah["steps_per_iter"]
    context_dim_two_door = 2
    for cur_algo in algorithms_half_cheetah:
        algorithm = algorithms_half_cheetah[cur_algo]["algorithm"]
        label = algorithms_half_cheetah[cur_algo]["label"]
        model = algorithms_half_cheetah[cur_algo]["model"]
        color = algorithms_half_cheetah[cur_algo]["color"]
        print(algorithm)
        expected = []
        for seed in settings_half_cheetah["seeds"]:
            base_dir = os.path.join(base_log_dir, "half_cheetah_3d_narrow", algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed = get_results(
                base_dir=base_dir,
                iterations=iterations_half_cheetah,
                plot_success=True,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)

        expected = np.array(expected)
        expected_mid = np.median(expected, axis=0)
        expected_qlow = np.quantile(expected, 0.25, axis=0)
        expected_qhigh = np.quantile(expected, 0.75, axis=0)
        axes[3].plot(iterations_step_half_cheetah, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      marker=".",
                      )
        axes[3].fill_between(iterations_step_half_cheetah, expected_qlow, expected_qhigh, color=color, alpha=0.4)

    markers = ["o", "x"]
    linestyles = ["-", "--"]
    labels = ["Mean", "Variance"]
    lines = [Line2D([0], [0], color="black", linestyle=linestyles[i], marker=markers[i]) for i in range(2)]
    axes_info = settings_two_door["axes_info"]
    for ax_i in range(2):
        axes[ax_i].set_ylabel(axes_info["ylabel"][ax_i])
        axes[ax_i].set_ylim(axes_info["ylim"][ax_i])
        axes[ax_i].grid(True)
        axes[ax_i].legend(lines, labels, fontsize=fontsize*0.8, loc="best", framealpha=1.)
        axes[ax_i].set_xlim([iterations_step_two_door[0], iterations_step_two_door[-1]])
        axes[ax_i].ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)

    axes[2].set_title("Two-door")
    axes[2].set_ylabel('Rate of task completion')
    axes[2].set_xlabel('Number of interactions')
    axes[2].set_xlim([iterations_step_two_door[0], iterations_step_two_door[-1]])
    axes[2].ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
    
    axes[3].set_title("HalfCheetah")
    axes[3].set_xlabel('Number of interactions', fontsize=fontsize)
    axes[3].set_xlim([iterations_step_half_cheetah[0], iterations_step_half_cheetah[-1]])
    axes[3].ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)

    colors = []
    labels = []
    num_alg = len(algorithms_two_door)
    for cur_algo in algorithms_two_door:
        colors.append(algorithms_two_door[cur_algo]["color"])
        labels.append(algorithms_two_door[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=math.ceil(num_alg/1), loc="upper center", bbox_to_anchor=(.5,1.1),
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    fig_filename = f"AAMAS_{figname}.pdf"
    print(f"{Path(os.getcwd()).parent}/figures/{fig_filename}")
    plt.savefig(f"{Path(os.getcwd()).parent}/figures/{fig_filename}", dpi=500,
                bbox_inches='tight',
                bbox_extra_artists=(lgd,))


def main():
    base_log_dir = f"{Path(os.getcwd()).parent}/logs"
    algorithms = {
        "two_door_discrete_2d_wide": {
            "RM-g": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=10_KL_EPS=0.05_OFFSET=70_ZETA=0.96_PCMDP=True",
                "color": "blue",
            },
            "Inter": {
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
            "DefP": {
                "algorithm": "default",
                "label": "Default*",
                "model": "sac_PCMDP=True",
                "color": "gold",
            },
            "Def": {
                "algorithm": "default",
                "label": "Default",
                "model": "sac_PCMDP=False",
                "color": "cyan",
            },
            "Goal": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "sac_GG_FIT_RATE=100_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.3_PCMDP=False",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
                "label": "ALP-GMM",
                "model": "sac_AG_FIT_RATE=200_AG_MAX_SIZE=1000_AG_P_RAND=0.2_PCMDP=False",
                "color": "magenta",
            },
        },
        "half_cheetah_3d_narrow": {
            "RM-g": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=0_KL_EPS=0.05_OFFSET=80_ZETA=1.0_PCMDP=True",
                "color": "blue",
            },
            "Inter": {
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
            "DefP": {
                "algorithm": "default",
                "label": "Default*",
                "model": "sac_PCMDP=True",
                "color": "gold",
            },
            "Def": {
                "algorithm": "default",
                "label": "Default",
                "model": "sac_PCMDP=False",
                "color": "cyan",
            },
            "Goal": {
                "algorithm": "goal_gan",
                "label": "GoalGAN",
                "model": "sac_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.3_PCMDP=False",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
                "label": "ALP-GMM",
                "model": "sac_AG_FIT_RATE=200_AG_MAX_SIZE=1000_AG_P_RAND=0.3_PCMDP=False",
                "color": "magenta",
            },
        },
    }

    settings = {
        "two_door_discrete_2d_wide":
            {
                "num_iters": 350,
                "steps_per_iter": 16384,
                "num_updates_per_iteration": 5,
                "seeds": [str(i) for i in range(1, 16)],
                "axes_info": {
                    "ylabel": [r'$\mathbf{c}[1]$: Door position',
                               r'$\mathbf{c}[2]$: Door width',
                               ],
                    "ylim": [[-.5, 2.2],
                             [-.5, 2.2],
                             ],
                    },
            },
        "half_cheetah_3d_narrow":
            {
                "num_iters": 250,
                "steps_per_iter": 16384,
                "num_updates_per_iteration": 10,
                "seeds": [str(i) for i in range(1, 11)],
            },
    }

    plot_results(
        base_log_dir=base_log_dir,
        settings=settings,
        algorithms=algorithms,
        )


if __name__ == "__main__":
    main()
