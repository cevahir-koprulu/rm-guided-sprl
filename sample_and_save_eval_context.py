import os
import numpy as np
from deep_sprl.experiments.two_door_discrete_2d_experiment import TwoDoorDiscrete2DExperiment
from deep_sprl.experiments.two_door_discrete_4d_experiment import TwoDoorDiscrete4DExperiment
from deep_sprl.experiments.half_cheetah_3d_experiment import HalfCheetah3DExperiment
from pathlib import Path

def sample_contexts(setting, num_contexts):
    target_mean = setting["target_mean"]
    target_covariance = setting["target_covariance"]
    lower_bounds = setting["lower_bounds"]
    upper_bounds = setting["upper_bounds"]
    contexts = []
    for c in range(num_contexts):
        contexts.append(np.clip(np.random.multivariate_normal(mean=target_mean, cov=target_covariance, ),
                                lower_bounds, upper_bounds))
    return np.array(contexts)


def main():
    num_contexts = 100
    eval_context_dir = f"{Path(os.getcwd())}/eval_contexts"
    env = "two_door_discrete_2d"
    target_type = "wide"
    if not os.path.exists(eval_context_dir):
        os.makedirs(eval_context_dir)
    exp = None
    if env == "two_door_discrete_2d":
        exp = TwoDoorDiscrete2DExperiment
    elif env == "two_door_discrete_4d":
        exp = TwoDoorDiscrete4DExperiment
    elif env == "half_cheetah_3d":
        exp = HalfCheetah3DExperiment
    else:
        raise ValueError("Invalid environment")

    target_setting = {
                "target_mean": exp.TARGET_MEAN,
                "target_covariance": exp.TARGET_VARIANCE_TYPES[target_type],
                "lower_bounds": exp.LOWER_CONTEXT_BOUNDS,
                "upper_bounds": exp.UPPER_CONTEXT_BOUNDS,
    }

    contexts = sample_contexts(setting=target_setting,
                               num_contexts=num_contexts)
    print(contexts)
    np.save(f"{eval_context_dir}/{env}_{target_type}_eval_contexts", contexts)


if __name__ == "__main__":
    main()