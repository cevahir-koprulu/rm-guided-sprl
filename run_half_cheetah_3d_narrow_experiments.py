import subprocess
import os
filepath = f"{os.getcwd()}/run.py"
arguments = [

            ################################
            #### Half-Cheetah 3D Narrow ####
            ################################

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '1', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '2', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '3', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '4', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '5', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '6', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '7', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '8', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '9', '--ZETA', '1.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '10', '--ZETA', '1.'],

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '1', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '2', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '3', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '4', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '5', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '6', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '7', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '8', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '9', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '10', '--ZETA', '4.'],

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '1', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '2', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '3', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '4', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '5', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '6', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '7', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '8', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '9', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '10', '--ZETA', '4.'],

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '1'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '2'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '3'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '4'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '5'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '6'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '7'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '8'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '9'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '10'],

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '5'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '6'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '7'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '8'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '9'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '10'],

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '5'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '6'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '7'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '8'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '9'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '10'],

            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '5'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '6'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '7'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '8'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '9'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'alp_gmm', '--TARGET_TYPE', 'narrow', '--seed', '10'],
]

for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])