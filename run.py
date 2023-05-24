import argparse
from deep_sprl.util.parameter_parser import parse_parameters
import deep_sprl.environments
import torch


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="self_paced",
                        choices=["default", "random", 
                        "self_paced", "self_paced_v2",
                        "rm_guided_self_paced", "rm_guided_self_paced_v2",
                        "alp_gmm", "goal_gan", "wasserstein", 
                        "plr", "vds", 
                        ])
    parser.add_argument("--learner", type=str, default="sac", choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="two_door_discrete_2d",
                        choices=["two_door_discrete_2d", "two_door_discrete_4d", 
                        "half_cheetah_3d", "fetch_push_and_play_4d", "swimmer_2d"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--PCMDP", action="store_true", default=False)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument("--device", type=str, default="cpu")

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)
    parameters["PCMDP"] = args.PCMDP
    
    torch.set_num_threads(args.n_cores)

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.env == "two_door_discrete_2d":
        from deep_sprl.experiments import TwoDoorDiscrete2DExperiment
        exp = TwoDoorDiscrete2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "two_door_discrete_4d":
        from deep_sprl.experiments import TwoDoorDiscrete4DExperiment
        exp = TwoDoorDiscrete4DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "half_cheetah_3d":
        from deep_sprl.experiments import HalfCheetah3DExperiment
        exp = HalfCheetah3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "fetch_push_and_play_4d":
        from deep_sprl.experiments import FetchPushAndPlay4DExperiment
        exp = FetchPushAndPlay4DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "swimmer_2d":
        from deep_sprl.experiments import Swimmer2DExperiment
        exp = Swimmer2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)


    if args.train:
        exp.train()

    if args.eval:
        exp.evaluate()


if __name__ == "__main__":
    main()
