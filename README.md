# Reward-Machine-Guided, Self-Paced Reinforcement Learning

Cevahir Koprulu and Ufuk Topcu from The University of Texas at Austin

Accepted for the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023).

Based on the source code for _Self-Paced Deep Reinforcement Learning_ (SPDL) by Klink et al. (2020) and _Curriculum Reinforcement Learning via Constrained Optimal Transport_ (CURROT) by Klink et al. (2022).

Web sources for SPDL:

Source code: https://github.com/psclklnk/spdl

Paper: https://papers.nips.cc/paper/2020/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html

Web sources for CURROT:

Source code: https://github.com/psclklnk/currot/tree/icml (ICML branch)

Paper: https://proceedings.mlr.press/v162/klink22a.html


We run our codebase on Ubuntu 20.04.5 LTS with Python 3.9.16

## Installation

We use a conda environment, but provide the required packages in a requirements.txt file so that the user 
can access it on other environment, as well.
```bash
pip install -r requirements.txt
```
In order to run experiments for the customized HalfCheetah-v3 environment, install mujoco-py: https://github.com/openai/mujoco-py

## How to run
To run a single experiment, *run.py* can be called as follows (you can put additional parameters):
```bash
python run.py --env --train --eval two_door_discrete_2d --type rm_guided_self_paced --target_type wide --PCMDP --ZETA 0.96 --seed 1 # RM-guided SPRL
python run.py --env --train --eval two_door_discrete_2d --type self_paced --target_type wide --PCMDP --ZETA 1.2 --seed 1 # Intermediate
python run.py --env --train --eval two_door_discrete_2d --type self_paced --target_type wide --PCMDP --ZETA 1.2 --seed 1 # SPDL
python run.py --env --train --eval two_door_discrete_2d --type default --target_type wide --PCMDP --seed 1 # Default*
python run.py --env --train --eval two_door_discrete_2d --type default --target_type wide --seed 1 # Default
python run.py --env --train --eval two_door_discrete_2d --type goal_gan --target_type wide --seed 1 # GoalGAN
python run.py --env --train --eval two_door_discrete_2d --type alp_gmm --target_type wide --seed 1 # ALP-GMM
```
The results demonstrated in our submitted paper can be run via *run_{environment_name}_experiments.py* by changing environment_name to one of the following:
- two_door_2d_wide
- swimmer_2d_narrow
- half_cheetah_3d_narrow

## Evaluation
Under *misc* directory, there are three scripts:
1) *run_welch_t_test.py*: Presented results for Welch's t-test are obtained via this script.
2) *plot_expected_performance.py*: We use this script to plot the progression of expected return and success during training.
3) *plot_curriculum_progression.py*: We run this script to plot the curricula during training.
4) *sample_eval_contexts.py*: We use this script to sample contexts from the target context distributions and save them for the evaluation of policies.