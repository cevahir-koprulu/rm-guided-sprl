import subprocess

filepath = r".\run_experiment.py"
arguments = [
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '1', '--ZETA', '1.0', '--true_rewards'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '1', '--ZETA', '1.0', '--true_rewards'],

            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '2', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '3', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '4', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '5', '--ZETA', '1.0'],

            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '1', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '2', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '3', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '4', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '1', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '2', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '3', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '4', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '1'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '3'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '4'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '5'],
            #
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '1', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '2', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '3', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '4', '--ZETA', '1.0'],
            # ['--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '5', '--ZETA', '1.0'],
            #
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '1', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '2', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '3', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '4', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'True', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '1', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '2', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '3', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '4', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '1'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '2'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '3'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '4'],
            # ['--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', '--PRODUCT_CMDP', 'False', '--seed', '5'],
            #
            # ['--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '1', '--ZETA', '0.96'],
            # ['--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '2', '--ZETA', '0.96'],
            # ['--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '3', '--ZETA', '0.96'],
            # ['--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '4', '--ZETA', '0.96'],
            # ['--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '5', '--ZETA', '0.96'],
            #
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '1', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '2', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '3', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '4', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'True', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '1', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '2', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '3', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '4', '--ZETA', '1.2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '1'],
            # ['--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '2'],
            # ['--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '3'],
            # ['--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '4'],
            # ['--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', '--PRODUCT_CMDP', 'False', '--seed', '5'],
]

for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])