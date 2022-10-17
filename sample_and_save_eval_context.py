import numpy as np


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
    eval_context_dir = "eval_contexts"
    env = "two_door_discrete_2d_narrow"
    settings = {
        "two_door_discrete_2d_narrow":
            {
                "target_mean": np.array([2., 2.]),
                "target_covariance": np.diag([16e-6, 16e-6]),
                "lower_bounds": np.array([-4., -4.]),
                "upper_bounds": np.array([4., 4.]),
            },

        "two_door_discrete_2d_wide":
            {
                "target_mean": np.array([2., 2.]),
                "target_covariance": np.diag([1., 1.]),
                "lower_bounds": np.array([-4., -4.]),
                "upper_bounds": np.array([4., 4.]),
            },

        "two_door_discrete_4d_narrow":
            {
                "target_mean": np.array([2., 2., -2., -2.]),
                "target_covariance": np.diag([16e-6, 16e-6, 16e-6, 16e-6]),
                "lower_bounds": np.array([-4., -4., -4., -4.]),
                "upper_bounds": np.array([4., 4., 4., 4.]),
            }
    }

    contexts = sample_contexts(setting=settings[env],
                               num_contexts=num_contexts)
    np.save(f"{eval_context_dir}\\{env}_eval_contexts", contexts)


if __name__ == "__main__":
    main()