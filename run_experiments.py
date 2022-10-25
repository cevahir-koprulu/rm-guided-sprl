import subprocess
import os
filepath = f"{os.getcwd()}/run_experiment.py"
arguments = [

            ################################
            #### Half-Cheetah 3D Narrow ####
            ################################

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '2.', '--true_rewards'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '2.', '--true_rewards'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '2.', '--true_rewards'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '2.', '--true_rewards'],
            ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '2.', '--true_rewards'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'self_paced', '--target_type', 'narrow', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'narrow', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],

            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'goalgan', '--seed', '1', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'goalgan', '--seed', '2', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'goalgan', '--seed', '3', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'goalgan', '--seed', '4', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],
            # ['--train', '--eval', '--env', 'half_cheetah_3d', '--type', 'default', '--target_type', 'goalgan', '--seed', '5', '--LEARNING_RATE', '0.001', '--ARCH', '256', '--ZETA', '1.', '--true_rewards'],

            ################################
            ### 2-Door Discrete 2D Narrow ##
            ################################

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '1', '--ZETA', '1.0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '2', '--ZETA', '1.0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '3', '--ZETA', '1.0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '4', '--ZETA', '1.0', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', '--seed', '5', '--ZETA', '1.0', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '1', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '3', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '4', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '5', '--true_rewards'],
            
            ################################
            #### 2-Door Discrete 2D Wide ###
            ################################

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.0'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'rm_guided_self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.0'],
            #
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', 'False', '--seed', '1', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', 'False', '--seed', '2', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', 'False', '--seed', '3', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', 'False', '--seed', '4', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'self_paced', '--target_type', 'wide', 'False', '--seed', '5', '--ZETA', '1.2'],

            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.2'],
            
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', 'False', '--seed', '1', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', 'False', '--seed', '2', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', 'False', '--seed', '3', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', 'False', '--seed', '4', '--ZETA', '1.2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'default', '--target_type', 'wide', 'False', '--seed', '5', '--ZETA', '1.2'],
            #
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', 'False', '--seed', '1'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', 'False', '--seed', '2'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', 'False', '--seed', '3'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', 'False', '--seed', '4'],
            # ['--train', '--eval', '--env', 'two_door_discrete_2d', '--type', 'goal_gan', '--target_type', 'wide', 'False', '--seed', '5'],
            
            ################################
            ### 2-Door Discrete 4D Narrow ##
            ################################

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '0.96', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '0.96', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '0.96', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '0.96', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'rm_guided_self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '0.96', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'self_paced', '--target_type', 'narrow', 'False', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', '--product_cmdp', 'True', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '1', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '2', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '3', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '4', '--ZETA', '1.2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'default', '--target_type', 'narrow', 'False', '--seed', '5', '--ZETA', '1.2', '--true_rewards'],

            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '1', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '2', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '3', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '4', '--true_rewards'],
            # ['--train', '--eval', '--env', 'two_door_discrete_4d', '--type', 'goal_gan', '--target_type', 'narrow', 'False', '--seed', '5', '--true_rewards'],
]

for args in arguments:
    subprocess.call(["python", filepath] + [arg for arg in args])