import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
from deep_sprl.util.parameter_parser import parse_parameters


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="self_paced",
                        choices=["default", "random",
                                 "self_paced", "self_paced_v2",
                                 "alp_gmm", "goal_gan",
                                 "rm_guided_self_paced"])
    parser.add_argument("--learner", type=str, default="sac",
                        choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="two_door_discrete_2d",
                        choices=["two_door_discrete_2d", "two_door_discrete_4d", "half_cheetah_3d"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--true_rewards", action="store_true", default=True)
    parser.add_argument("--product_cmdp", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    if args.type == "self_paced" or args.type == "rm_guided_self_paced":
        import torch
        torch.set_num_threads(1)

    if args.env == "two_door_discrete_2d":
        from deep_sprl.experiments import TwoDoorDiscrete2DExperiment
        exp = TwoDoorDiscrete2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed,
                                          use_true_rew=args.true_rewards, use_product_cmdp=args.product_cmdp)
    elif args.env == "two_door_discrete_4d":
        from deep_sprl.experiments import TwoDoorDiscrete4DExperiment
        exp = TwoDoorDiscrete4DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed,
                                          use_true_rew=args.true_rewards, use_product_cmdp=args.product_cmdp)
    elif args.env == "half_cheetah_3d":
        from deep_sprl.experiments import HalfCheetah3DExperiment
        exp = HalfCheetah3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed,
                                          use_true_rew=args.true_rewards, use_product_cmdp=args.product_cmdp)
    else:
        raise ValueError(f"Environment {args.env} does not exist")

    if args.train:
        exp.train()
    if args.eval:
        exp.evaluate()


if __name__ == "__main__":
    main()
