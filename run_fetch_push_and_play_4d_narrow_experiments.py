import subprocess
import os
filepath = f"{os.getcwd()}/run.py"
arguments = [

            ################################
            #### Half-Cheetah 3D Narrow ####
            ################################

            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '1'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '2'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '3'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '4'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '5'],

            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '1', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '2', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '3', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '4', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '5', '--ZETA', '4.'],

            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '1', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '2', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '3', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '4', '--ZETA', '4.'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '5', '--ZETA', '4.'],

            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '1'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '2'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '3'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '4'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--PCMDP', '--seed', '5'],

            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '5'],

            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            ['--train', '--eval', '--env', 'fetch_push_and_play_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '5'],
]

for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])