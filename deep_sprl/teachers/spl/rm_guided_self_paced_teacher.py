import torch
import numpy as np
from deep_sprl.util.torch import to_float_tensor
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from scipy.optimize import minimize, NonlinearConstraint, Bounds

class AbstractRMguidedSelfPacedTeacher:

    def __init__(self, init_mean, flat_init_chol, target_mean, flat_target_chol, alpha_function, max_kl):
        self.context_dist = GaussianTorchDistribution(init_mean, flat_init_chol, use_cuda=False, dtype=torch.float64)
        self.target_dist = GaussianTorchDistribution(target_mean, flat_target_chol, use_cuda=False, dtype=torch.float64)

        self.alpha_function = alpha_function
        self.max_kl = max_kl

        self.iteration = 0

    def target_context_kl(self, numpy=True):
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist.distribution_t,
                                                      self.target_dist.distribution_t).detach()
        if numpy:
            kl_div = kl_div.numpy()
        return kl_div

    def save(self, path):
        weights = self.context_dist.get_weights()
        np.save(path, weights)

    def load(self, path):
        self.context_dist.set_weights(np.load(path))

    @staticmethod
    def get_dists_power_set(contexts_power_set, dist):
        dists_power_set_dict = {}
        for context_subset in contexts_power_set:
            if len(context_subset) == 0:
                continue
            else:
                dist_mean = dist.mean()
                dist_covariance = dist.covariance_matrix()
                mu_active = dist_mean[context_subset]
                covariance_active = np.zeros((len(context_subset), len(context_subset)))
                for i_, i in enumerate(context_subset):
                    for j_, j in enumerate(context_subset):
                        covariance_active[i_, j_] = dist_covariance[i, j]
                covariance_active_flat = GaussianTorchDistribution.flatten_matrix(covariance_active,
                                                                                  tril=False)
                dists_power_set_dict[tuple(context_subset)] = GaussianTorchDistribution(mu_active,
                                                                                        covariance_active_flat,
                                                                                        use_cuda=False,
                                                                                        dtype=torch.float64)
        return dists_power_set_dict

    def _compute_rm_guided_context_loss(self, dist, rm_transitions_buffer, alpha_cur_t, rm_transition_contexts,
                                        old_c_log_prob_t_dict):
        dists_power_set_dict = self.get_dists_power_set(rm_transition_contexts, dist)
        weighted_return = 0.
        for context_subset in rm_transitions_buffer["transitions"]:
            if len(context_subset) != 0:
               active_dist = dists_power_set_dict[context_subset]
               importance_weight = \
                   torch.exp(active_dist.log_pdf_t(rm_transitions_buffer["transitions"][context_subset]["contexts"]) -
                             old_c_log_prob_t_dict[context_subset]).detach().numpy()
            else:
                importance_weight = np.ones(len(rm_transitions_buffer["transitions"][context_subset]["rewards"]))
            weighted_return += np.sum(importance_weight * rm_transitions_buffer["transitions"][context_subset]["rewards"])
        kl_div = torch.distributions.kl.kl_divergence(dist.distribution_t, self.target_dist.distribution_t)
        return weighted_return / rm_transitions_buffer["num_trajectories"] - alpha_cur_t * kl_div


class RMguidedSelfPacedTeacher(AbstractTeacher, AbstractRMguidedSelfPacedTeacher):

    def __init__(self, target_mean, target_variance, initial_mean, initial_variance, context_bounds, alpha_function,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed

        # The bounds that we show to the outside are limited to the interval [-1, 1], as this is typically better for
        # neural nets to deal with
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        self.use_avg_performance = use_avg_performance

        if std_lower_bound is not None and kl_threshold is None:
            raise RuntimeError("Error! Both Lower Bound on standard deviation and kl threshold need to be set")
        else:
            if std_lower_bound is not None:
                if isinstance(std_lower_bound, np.ndarray):
                    if std_lower_bound.shape[0] != self.context_dim:
                        raise RuntimeError("Error! Wrong dimension of the standard deviation lower bound")
                elif std_lower_bound is not None:
                    std_lower_bound = np.ones(self.context_dim) * std_lower_bound
            self.std_lower_bound = std_lower_bound
            self.kl_threshold = kl_threshold

        # Create the initial context distribution
        if isinstance(initial_variance, np.ndarray):
            flat_init_chol = GaussianTorchDistribution.flatten_matrix(initial_variance, tril=False)
        else:
            flat_init_chol = GaussianTorchDistribution.flatten_matrix(initial_variance * np.eye(self.context_dim),
                                                                      tril=False)

        # Create the target distribution
        if isinstance(target_variance, np.ndarray):
            flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance, tril=False)
        else:
            flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance * np.eye(self.context_dim),
                                                                        tril=False)

        super(RMguidedSelfPacedTeacher, self).__init__(initial_mean, flat_init_chol, target_mean, flat_target_chol,
                                                       alpha_function, max_kl)

    def update_distribution(self, avg_performance, contexts, values, rm_transitions_buffer=None):
        self.iteration += 1

        old_context_dist = GaussianTorchDistribution.from_weights(self.context_dim, self.context_dist.get_weights(),
                                                                  dtype=torch.float64)
        contexts_t = to_float_tensor(contexts, use_cuda=False, dtype=torch.float64)

        # Estimate the value of the state after the policy update
        c_val_t = to_float_tensor(values, use_cuda=False, dtype=torch.float64)

        # Add the penalty term
        cur_kl_t = self.target_context_kl(numpy=False)
        if self.use_avg_performance:
            alpha_cur_t = self.alpha_function(self.iteration, avg_performance, cur_kl_t)
        else:
            alpha_cur_t = self.alpha_function(self.iteration, torch.mean(c_val_t).detach(), cur_kl_t)

        # Define the KL-Constraint
        def kl_con_fn(x):
            dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, dist.distribution_t)
            return kl_div.detach().numpy()

        def kl_con_grad_fn(x):
            dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
            kl_div = torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, dist.distribution_t)
            mu_grad, chol_flat_grad = torch.autograd.grad(kl_div, dist.parameters())
            return np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()])

        kl_constraint = NonlinearConstraint(kl_con_fn, -np.inf, self.max_kl, jac=kl_con_grad_fn, keep_feasible=True)
        constraints = [kl_constraint]

        if self.kl_threshold is not None and self.target_context_kl() > self.kl_threshold:
            # Define the variance constraint as bounds
            cones = np.ones_like(self.context_dist.get_weights())
            lb = -np.inf * cones.copy()
            lb[self.context_dim: 2 * self.context_dim] = np.log(self.std_lower_bound)
            ub = np.inf * cones.copy()
            bounds = Bounds(lb, ub, keep_feasible=True)
            x0 = np.clip(self.context_dist.get_weights().copy(), lb, ub)
        else:
            bounds = None
            x0 = self.context_dist.get_weights().copy()

        rm_transition_contexts = [list(x) for x in set(tuple(x) for x in list(
            rm_transitions_buffer["transitions"].keys()))]
        old_dists_power_set_dict = self.get_dists_power_set(rm_transition_contexts, old_context_dist)
        old_c_log_prob_t_dict = dict()
        for context_subset in rm_transitions_buffer["transitions"]:
            if len(context_subset) == 0:
                continue
            active_old_dist = old_dists_power_set_dict[context_subset]
            rm_transitions_buffer["transitions"][context_subset]["contexts"] = \
                to_float_tensor(np.array(rm_transitions_buffer["transitions"][context_subset]["contexts"]),
                                use_cuda=False, dtype=torch.float64)
            rm_transitions_buffer["transitions"][context_subset]["rewards"] = \
                np.array(rm_transitions_buffer["transitions"][context_subset]["rewards"])
            old_c_log_prob_t_dict[context_subset] = \
                active_old_dist.log_pdf_t(rm_transitions_buffer["transitions"][context_subset]["contexts"]).detach()

        # Define the objective plus Jacobian
        def objective(x):
            dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
            val = self._compute_rm_guided_context_loss(dist, rm_transitions_buffer, alpha_cur_t,
                                                       rm_transition_contexts, old_c_log_prob_t_dict)
            mu_grad, chol_flat_grad = torch.autograd.grad(val, dist.parameters())

            return -val.detach().numpy(), \
                   -np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()]).astype(np.float64)

        res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                       constraints=constraints, options={"gtol": 1e-4, "xtol": 1e-6})

        if res.success:
            self.context_dist.set_weights(res.x)
        else:
            # If it was not a success, but the objective value was improved and the bounds are still valid, we still
            # use the result
            old_f = objective(self.context_dist.get_weights())[0]
            kl_ok = kl_con_fn(res.x) <= self.max_kl
            std_ok = bounds is None or (np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub))
            if kl_ok and std_ok and res.fun < old_f:
                self.context_dist.set_weights(res.x)
            else:
                print("Warning! Context optimihation unsuccessful - will keep old values. Message: %s" % res.message)

    def sample(self):
        sample = self.context_dist.sample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])