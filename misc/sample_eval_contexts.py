import sys
sys.path.insert(0, '..')
import os
import numpy as np
from deep_sprl.experiments.two_door_discrete_2d_experiment import TwoDoorDiscrete2DExperiment
from deep_sprl.experiments.two_door_discrete_4d_experiment import TwoDoorDiscrete4DExperiment
from deep_sprl.experiments.half_cheetah_3d_experiment import HalfCheetah3DExperiment
from deep_sprl.experiments.fetch_push_and_play_4d_experiment import FetchPushAndPlay4DExperiment
from deep_sprl.experiments.swimmer_2d_experiment import Swimmer2DExperiment
from pathlib import Path


def sample_contexts(target_sampler, bounds, num_contexts):
    lower_bounds = bounds["lower_bounds"]
    upper_bounds = bounds["upper_bounds"]
    contexts = np.clip(target_sampler(n=num_contexts), lower_bounds, upper_bounds)
    return contexts

def main():
    ##################################
    num_contexts = 100
    eval_context_dir = f"{Path(os.getcwd()).parent}/eval_contexts"
    target_type = "narrow"
    env = f"swimmer_2d_{target_type}"
    # target_type = "narrow"
    # env = f"fetch_push_and_play_4d_{target_type}"
    # target_type = "narrow"
    # env = f"half_cheetah_3d_{target_type}"
    # target_type = "wide"
    # env = f"two_door_discrete_2d_{target_type}"
    ##################################

    if not os.path.exists(eval_context_dir):
        os.makedirs(eval_context_dir)

    if env[:-len(target_type)-1] == "two_door_discrete_2d":
        exp = TwoDoorDiscrete2DExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="sac",
                                          parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "two_door_discrete_4d":
        exp = TwoDoorDiscrete4DExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="sac",
                                          parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "half_cheetah_3d":
        exp = HalfCheetah3DExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="sac", 
                                      parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "swimmer_2d":
        exp = Swimmer2DExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="sac", 
                                      parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "fetch_push_and_play_4d":
        exp = FetchPushAndPlay4DExperiment(base_log_dir="logs", curriculum_name="self_paced", learner_name="sac", 
                                      parameters={"TARGET_TYPE": target_type}, seed=1, device="cpu")
    else:
        raise ValueError("Invalid environment")

    bounds = {
        "lower_bounds": exp.LOWER_CONTEXT_BOUNDS,
        "upper_bounds": exp.UPPER_CONTEXT_BOUNDS,
    }

    contexts = sample_contexts(target_sampler=exp.target_sampler,
                                bounds=bounds,
                                num_contexts=num_contexts,)
    print(contexts)
    np.save(f"{eval_context_dir}/{env}_eval_contexts", contexts)

if __name__ == "__main__":
    main()