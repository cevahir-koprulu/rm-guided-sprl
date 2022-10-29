import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import rc
from matplotlib.colors import ListedColormap
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))
from deep_sprl.experiments import TwoDoorDiscrete2DExperiment, TwoDoorDiscrete4DExperiment, CurriculumType

rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

base_log_dir = "logs"
picture_dir = "figures"


def main():
    learner = "sac"
    env = "two_door_discrete_2d"
    target_type = "wide"
    seeds = [1, 2, 3, 4, 5]
    seed_colors = ['crimson', 'chocolate', 'lime', 'midnightblue', 'purple']
    model_iter = 195
    context_no = 0
    contexts = np.load(os.path.join(Path(os.getcwd()).parent, "eval_contexts", f"{env}_{target_type}_eval_contexts.npy"))
    context = contexts[context_no]

    picture_base = os.path.join(Path(os.getcwd()).parent, picture_dir)
    log_dir = os.path.join(Path(os.getcwd()).parent, base_log_dir, f"{env}_{target_type}")
    algorithms = {
        "default": {
            "name": "default",
            "model": "sac_LR=0.0003_ARCH=256_RBS=60000",
            },
        "default(P)": {
            "name": "default",
            "model": "sac_LR=0.0003_ARCH=256_RBS=60000_PRODUCTCMDP",
            },
        "goal_gan": {
            "name": "goal_gan",
            "model": "sac_GG_FIT_RATE=100_GG_NOISE_LEVEL=0.05_GG_P_OLD=0.2_LR=0.0003_ARCH=256_RBS=60000",
            },
        "self_paced": {
            "name": "self_paced",
            "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=1.2_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS",
            },
        "rm_guided": {
            "name": "rm_guided_self_paced",
            "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=0.96_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS_PRODUCTCMDP",
            },
        "intermediate": {
            "name": "self_paced",
            "model": "sac_ALPHA_OFFSET=10_MAX_KL=0.05_OFFSET=70_ZETA=1.2_LR=0.0003_ARCH=256_RBS=60000_TRUEREWARDS_PRODUCTCMDP",
            }
    }

    algos = list(algorithms.keys())
    for algo in algos:
        cur_algo = algorithms[algo]["name"]
        model_name = algorithms[algo]["model"]
        PRODUCTCMDP = model_name[-len("PRODUCTCMDP"):] == "PRODUCTCMDP"

        p_door1 = round((context[0] + 4.) / 8 * 40)
        p_door2 = round((context[1] + 4.) / 8 * 40)
        if env[-2:] == "2d":    
            exp = TwoDoorDiscrete2DExperiment(base_log_dir, cur_algo, learner, {}, 1, use_product_cmdp=PRODUCTCMDP)
            p_box = round((-2 + 4.) / 8 * 40)
            p_goal = round((0. + 4.) / 8 * 40)
        elif env[-2:] == "4d":
            exp = TwoDoorDiscrete4DExperiment(base_log_dir, cur_algo, learner, {}, 1, use_product_cmdp=PRODUCTCMDP)
            p_box = round((context[2] + 4.) / 8 * 40)
            p_goal = round((context[3] + 4.) / 8 * 40)
        else:
            raise ValueError("Invalid environment type")
        type_log_dir = os.path.join(log_dir, cur_algo, model_name)

        N_GRID = 41
        context_discrete = [p_door1, p_door2, p_box, p_goal]
        door_widths = [2, 2]
        box_shape = [5, 5]
        start = np.array([20, 35],
                         dtype=int)
        doors = np.array([[context_discrete[0], 30, door_widths[0]], [context_discrete[1], 10, door_widths[1]]],
                         dtype=int)
        box = np.array([[context_discrete[2], 20], [context_discrete[2]+box_shape[0], 20-box_shape[1]]],
                       dtype=int)
        goal = np.array([context_discrete[3], 5],
                        dtype=int)
        two_door_grid = np.ones((N_GRID, N_GRID))*np.nan

        fig, ax = plt.subplots(1, 1, tight_layout=True)

        # Door 1, Box, Door 2, Goal, Wall, Start, Agent
        two_door_colors = ['blue', 'gold', 'magenta', 'red', 'gray', 'green']
        two_door_cmap = ListedColormap(two_door_colors)
        two_door_cmap.set_bad(color='w', alpha=0)

        # ax.text(start[0]-2., start[1]+1.05, "Start", fontsize=20)
        # ax.text(doors[0, 0]-3., doors[0, 1]+1.05, "Door 1", fontsize=20)
        # ax.text(goal[0]-2., goal[1]+1.05, "Goal", fontsize=20)
        # ax.text(doors[1, 0]-3., doors[1, 1]+1.05, "Door 2", fontsize=20)
        # ax.text(box[0, 0]+1, box[0, 1]+1.05, "Box", fontsize=20)

        start[1] = 40 - start[1]
        doors[:, 1] = 40 - doors[:, 1]
        box[:, 1] = 40 - box[:, 1]
        goal[1] = 40 - goal[1]

        # Start
        two_door_grid[start[0]:start[0]+1, start[1]:start[1]+1] = 5
        # Wall 1
        two_door_grid[:, doors[0, 1]:doors[0, 1]+1] = 4
        # Wall 2
        two_door_grid[:, doors[1, 1]:doors[1, 1]+1] = 4
        # Goal
        two_door_grid[goal[0]:goal[0]+1, goal[1]:goal[1]+1] = 3
        # Door 2
        two_door_grid[doors[1, 0]-doors[1, 2]:doors[1, 0]+doors[1, 2]+1, doors[1, 1]:doors[1, 1]+1] = 2
        # Box
        two_door_grid[box[0, 0]:box[1, 0]+1, min(box[1, 1], box[0, 1]):max(box[1, 1], box[0, 1])+1] = 1
        # Door 1
        two_door_grid[doors[0, 0]-doors[0, 2]:doors[0, 0]+doors[0, 2]+1, doors[0, 1]:doors[0, 1]+1] = 0
        ax.imshow(two_door_grid.T, interpolation='none', cmap=two_door_cmap, extent=[0, N_GRID, 0, N_GRID], zorder=0)

        for seed_i, seed in enumerate(seeds):
            path = os.path.join(type_log_dir, "seed-" + str(seed), "iteration-" + str(model_iter))
            if os.path.exists(path):
                model = exp.learner.load_for_evaluation(os.path.join(path, "model"), exp.vec_eval_env)

                traj = []
                final = []
                done = False
                exp.eval_env.set_current_context(contexts[context_no])
                obs = exp.vec_eval_env.reset()
                x_pos = round((obs[0][0]+4.)/8*40)
                y_pos = round((obs[0][1]+4.)/8*40)
                traj.append([x_pos+0.5, y_pos+0.5])
                while not done:
                    action = model.step(obs, state=None, deterministic=False)
                    obs, reward, done, info = exp.vec_eval_env.step(action)
                    if not info[0]["success"] and not done:
                        x_pos = round((obs[0][0] + 4.) / 8 * 40)
                        y_pos = round((obs[0][1] + 4.) / 8 * 40)
                        traj.append([x_pos + 0.5, y_pos + 0.5])
                    else:
                        terminal_obs = info[0]["terminal_observation"][[0, 1]]
                        x_pos = round((terminal_obs[0] + 4.) / 8 * 40)
                        y_pos = round((terminal_obs[1] + 4.) / 8 * 40)
                        traj.append([x_pos + 0.5, y_pos + 0.5])
                        final.append([x_pos + 0.5, y_pos + 0.5])

                traj=np.array(traj)
                ax.plot(traj[:, 0], traj[:, 1], color=seed_colors[seed_i], alpha=.8, label=f"seed-{seed}", linewidth=2)
                ax.scatter(final[0][0], final[0][1], marker="x", color=seed_colors[seed_i], alpha=.8, linewidth=2)

        ax.axis('off')
        plt.savefig(os.path.join(picture_base,
                                 f"{env}_{target_type}_{algo}.pdf"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()
        # plt.savefig(os.path.join(picture_base, f"{env}_{target_type}.pdf"),  bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    main()
