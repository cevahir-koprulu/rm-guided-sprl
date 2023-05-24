import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

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

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra, plot_success):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    ylabel = setting["ylabel"]
    ylim = setting["ylim"]
    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    alg_exp_mid = {}

    results_df = pd.DataFrame(columns=["Algorithm", "Max", "3rd Quartile", "Median", "1st Quartile", "Min"])

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        expected = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_seed = get_results(
                base_dir=base_dir,
                iterations=iterations,
                plot_success=plot_success,
            )
            if len(expected_seed) == 0:
                continue

            expected.append(expected_seed)

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

        # print(f"Max         : {expected_high[-1]}")
        # print(f"3rd quartile: {expected_qhigh[-1]}")
        # print(f"Median      : {expected_mid[-1]}")
        # print(f"1st quartile: {expected_qlow[-1]}")
        # print(f"Min         : {expected_low[-1]}")

        results_df = results_df.append(pd.DataFrame([[label, expected_high[-1], expected_qhigh[-1],
                                                     expected_mid[-1], expected_qlow[-1],
                                                     expected_low[-1]]],
                                                     columns=list(results_df.columns)),
                                                     ignore_index=True)
        
        print(f"Algorithm: {label} || Final results: {expected[:,-1]}")

        alg_exp_mid[cur_algo] = expected_mid[-1]

        plt.plot(iterations_step, expected_mid, color=color, linewidth=2.0, label=f"{label}",
                      marker=".",
                      )
        plt.fill_between(iterations_step, expected_qlow, expected_qhigh, color=color, alpha=0.4)
        # axes[context_dim].fill_between(iterations_step, expected_low, expected_high, color=color, alpha=0.2)

    print(results_df)

    # To DO: Set x ticks!
    plt.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
    plt.xlim([iterations_step[0], iterations_step[-1]])
    if plot_success:
        plt.ylim([-0.1, 1.1])
        plt.ylabel("Expected success ratio")
    else:
        plt.ylim(ylim)
        plt.ylabel(ylabel)
    plt.xlabel("Number of interactions")
    plt.grid(True)

    sorted_alg_exp_mid = [b[0] for b in sorted(enumerate(list(alg_exp_mid.values()), ), key=lambda i: i[1])]
    colors = []
    labels = []
    num_alg = len(algorithms)
    for alg_i in sorted_alg_exp_mid:
        cur_algo = list(alg_exp_mid.keys())[alg_i]
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=math.ceil(num_alg/2), loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    print(f"{Path(os.getcwd()).parent}/figures/{env}_{figname}{figname_extra}.pdf")
    # plt.savefig(f"{Path(os.getcwd()).parent}/figures/{env}_{figname}{figname_extra}.pdf", dpi=500,
    #             bbox_inches='tight', bbox_extra_artists=(lgd,))


def main():
    base_log_dir = f"{Path(os.getcwd()).parent}/logs"
    num_updates_per_iteration = 10
    seeds = [str(i) for i in range(1, 11)]
    # seeds = [2, 3, 4, 6, 7, 9, 10]
    # env = "two_door_discrete_2d_wide"
    # env = "swimmer_2d_narrow"
    env = "half_cheetah_3d_narrow"
    plot_success = True
    figname_extra = "_expected_return" if not plot_success else "_expected_success"

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
        "swimmer_2d_narrow": {
            "RM-g": {
                "algorithm": "rm_guided_self_paced",
                "label": "RM-guided SPRL",
                "model": "sac_ALPHA_OFFSET=5_KL_EPS=0.1_OFFSET=10_ZETA=1.0_PCMDP=True",
                "color": "blue",
            },
            "Inter": {
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
                "fontsize": 20,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.17),
                "ylabel": 'Expected discounted return',
                "ylim": [0., 4.],
            },
        "swimmer_2d_narrow":
            {
                "num_iters": 200,
                "steps_per_iter": 16384,
                "fontsize": 20,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.18),
                "ylabel": 'Expected discounted return',
                "ylim": [-10., 300.],
            },
        "half_cheetah_3d_narrow":
            {
                "num_iters": 250,
                "steps_per_iter": 16384,
                "fontsize": 20,
                "figsize": (10, 5),
                "bbox_to_anchor": (.5, 1.185),
                "ylabel": 'Expected discounted return',
                "ylim": [-50., 500.],
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
        plot_success=plot_success,
        )


if __name__ == "__main__":
    main()
