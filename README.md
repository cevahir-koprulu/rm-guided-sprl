# Reward-Machine-Guided, Self-Paced Reinforcement Learning

Based on the source code for _Self-Paced Deep Reinforcement Learning_ (SPDL) by Klink et al. (2020) and _Curriculum Reinforcement Learning via Constrained Optimal Transport_ (CURROT) by Klink et al. (2022).

Web sources for SPDL:

Source code: https://github.com/psclklnk/spdl

Paper: https://papers.nips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html

Web sources for CURROT:

Source code: https://github.com/psclklnk/currot/tree/icml (ICML branch)

Paper: https://proceedings.mlr.press/v162/klink22a.html


We run our codebase on Ubuntu 20.04.5 LTS with Python 3.9.13

## Installation

We use a conda environment, but provide the required packages in a requirements.txt file so that the user 
can access it on other environment, as well.
```bash
pip install -r requirements.txt
```
In order to run experiments for the customized HalfCheetah-v3 environment, install mujoco-py: https://github.com/openai/mujoco-py

## How to run
To run a single experiment, *run_experiment.py* can be called as follows (you can put additional parameters):
```bash
python run_experiment.py --env two_door_discrete_2d --type rm_guided_self_paced --target_type wide --PRODUCTCMDP --ZETA 0.96 --seed 1 # RM-guided SPRL
python run_experiment.py --env two_door_discrete_2d --type self_paced --target_type wide --PRODUCTCMDP --ZETA 1.2 --seed 1 # Intermediate
python run_experiment.py --env two_door_discrete_2d --type self_paced --target_type wide --PRODUCTCMDP --ZETA 1.2 --seed 1 # SPDL
python run_experiment.py --env two_door_discrete_2d --type default --target_type wide --PRODUCTCMDP --seed 1 # Default*
python run_experiment.py --env two_door_discrete_2d --type default --target_type wide --seed 1 # Default
python run_experiment.py --env two_door_discrete_2d --type goal_gan --target_type wide --PRODUCTCMDP --seed 1 # GoalGAN
```
The results demonstrated in our submitted paper can be run via *run_experiments.py* which include 
78 training+evaluation runs in total.

*sample_and_save_eval_context.py* is used to sample contexts from the target context distributions. 
These samples are saved to be used for evaluation in every experiment so that algorithms can be compared in a fair manner.

## Evaluation
*plot_results.py* is used to get the plots (curricula and expected return progression) 
provided in the submitted paper.



Under *misc* directory, there are two scripts:
1) *run_welch_t_test.py*: Presented results for Welch's t-test are obtained via this script.
2) *visualize_two_door_discrete.py*: We use this script to visualize the trajectories produced in the two-door environment.

## Reproducibility

We provide the data saved during the training of RM-guided SPRL, Intermediate SPRL, SPDL and GoalGAN for Case Study 1,
which is the two-door environment with 2D context space (horizontal positions of doors) and narrow target
distribution.

You can run the evaluation scripts to generate the results for this case study (see *misc* directory):
1) Training progression plots in the submitted paper: *plot_results.py*
2) Welch's t-test results for curricula variance in the submitted paper: *run_welch_t_test.py*
3) Qualitative comparison (supplementary material) with agent paths in the environment: *visualize_two_door_discrete.py*