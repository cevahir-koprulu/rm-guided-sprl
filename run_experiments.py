import subprocess
import os
filepath = f"{os.getcwd()}/run_experiment.py"
arguments = [

            ################################
            #### Half-Cheetah 3D Narrow ####
            ################################

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '4.', '--OFFSET', '80', '--ALPHA_OFFSET', '0', '--true_rewards'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256'],

            ################################
            #### 2-Door Discrete 2D Wide ###
            ################################

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '1', '--ZETA', '0.96', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '2', '--ZETA', '0.96', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '3', '--ZETA', '0.96', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '4', '--ZETA', '0.96', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '5', '--ZETA', '0.96', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '1', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '2', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '3', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '4', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '5', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--seed', '1', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--seed', '2', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--seed', '3', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--seed', '4', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--TARGET_TYPE', 'wide', '--seed', '5', '--ZETA', '1.2', '--OFFSET', '70', '--ALPHA_OFFSET', '10', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '1'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '3'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '4'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--product_cmdp', '--seed', '5'],
            
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--seed', '1'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--seed', '2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--seed', '3'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--seed', '4'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--TARGET_TYPE', 'wide', '--seed', '5'],
            
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--TARGET_TYPE', 'wide', '--seed', '1'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--TARGET_TYPE', 'wide', '--seed', '2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--TARGET_TYPE', 'wide', '--seed', '3'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--TARGET_TYPE', 'wide', '--seed', '4'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--TARGET_TYPE', 'wide', '--seed', '5'],
            
            ################################
            ### 2-Door Discrete 4D Narrow ##
            ################################

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '1', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '2', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '3', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '4', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '5', '--ZETA', '1.0'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '1', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '2', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '3', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '4', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '5', '--ZETA', '1.2'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '1', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '2', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '3', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '4', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--TARGET_TYPE', 'narrow', '--seed', '5', '--ZETA', '1.2'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '1'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '3'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '4'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--product_cmdp', '--seed', '5'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--TARGET_TYPE', 'narrow', '--seed', '5'],

            # ['--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '1'],
            # ['--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '2'],
            # ['--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '3'],
            # ['--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '4'],
            # ['--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--TARGET_TYPE', 'narrow', '--seed', '5'],
]

for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])