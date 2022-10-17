import torch
import numpy as np
from deep_sprl.util.torch import to_float_tensor
from deep_sprl.util.gaussian_torch_distribution import GaussianTorchDistribution
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import os
import pickle
import time


class AbstractSelfPacedTeacher:

    def __init__(self, init_mean, flat_init_chol, target_mean, flat_target_chol, max_kl):
        self.context_dist = GaussianTorchDistribution(init_mean, flat_init_chol, use_cuda=False, dtype=torch.float64)
        self.target_dist = GaussianTorchDistribution(target_mean, flat_target_chol, use_cuda=False, dtype=torch.float64)

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

    def _compute_context_kl(self, old_context_dist):
        return torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, self.context_dist.distribution_t)

    def _compute_expected_performance(self, dist, cons_t, old_c_log_prob_t, c_val_t):
        con_ratio_t = torch.exp(dist.log_pdf_t(cons_t) - old_c_log_prob_t)
        return torch.mean(con_ratio_t * c_val_t)


class SelfPacedTeacherV2(AbstractTeacher, AbstractSelfPacedTeacher):

    def __init__(self, target_mean, target_variance, initial_mean, initial_variance, context_bounds, perf_lb,
                 max_kl=0.1, std_lower_bound=None, kl_threshold=None, use_avg_performance=False):

        # The bounds that we show to the outside are limited to the interval [-1, 1], as this is typically better for
        # neural nets to deal with
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        self.use_avg_performance = use_avg_performance
        self.perf_lb = perf_lb
        self.perf_lb_reached = False

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

        super(SelfPacedTeacherV2, self).__init__(initial_mean, flat_init_chol, target_mean, flat_target_chol, max_kl)

    def update_distribution(self, avg_performance, contexts, values):
        self.iteration += 1

        old_context_dist = GaussianTorchDistribution.from_weights(self.context_dim, self.context_dist.get_weights(),
                                                                  dtype=torch.float64)
        contexts_t = to_float_tensor(contexts, use_cuda=False, dtype=torch.float64)
        old_c_log_prob_t = old_context_dist.log_pdf_t(contexts_t).detach()

        # Estimate the value of the state after the policy update
        c_val_t = to_float_tensor(values, use_cuda=False, dtype=torch.float64)

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

        # Define the performance constraint
        def perf_con_fn(x):
            dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
            perf = self._compute_expected_performance(dist, contexts_t, old_c_log_prob_t, c_val_t)
            return perf.detach().numpy()

        def perf_con_grad_fn(x):
            dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
            perf = self._compute_expected_performance(dist, contexts_t, old_c_log_prob_t, c_val_t)
            mu_grad, chol_flat_grad = torch.autograd.grad(perf, dist.parameters())
            return np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()])

        perf_constraint = NonlinearConstraint(perf_con_fn, self.perf_lb, np.inf, jac=perf_con_grad_fn,
                                              keep_feasible=True)

        if self.kl_threshold is not None and self.target_context_kl() > self.kl_threshold:
            # Define the variance constraint as bounds
            cones = np.ones_like(self.context_dist.get_weights())
            lb = -np.inf * cones.copy()
            lb[self.context_dim: 2 * self.context_dim] = np.log(self.std_lower_bound)
            ub = np.inf * cones.copy()
            bounds = Bounds(lb, ub, keep_feasible=True)

            # If the bounds are active, clip the standard deviation to be in bounds (because we may re-introduce
            # bounds after they have previously been removed)
            x0 = np.clip(self.context_dist.get_weights().copy(), lb, ub)
        else:
            x0 = self.context_dist.get_weights().copy()
            bounds = None

        try:
            if kl_con_fn(x0) >= self.max_kl:
                print("Warning! KL-Bound of x0 violates constraint already")

            print(perf_con_fn(x0))
            if perf_con_fn(x0) >= self.perf_lb:
                print("Optimizing KL")
                self.perf_lb_reached = True
                constraints = [kl_constraint, perf_constraint]

                # Define the objective plus Jacobian
                def objective(x):
                    dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
                    kl_div = torch.distributions.kl.kl_divergence(dist.distribution_t, self.target_dist.distribution_t)
                    mu_grad, chol_flat_grad = torch.autograd.grad(kl_div, dist.parameters())

                    return kl_div.detach().numpy(), \
                           np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()]).astype(
                               np.float64)

                res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                               constraints=constraints, options={"gtol": 1e-4, "xtol": 1e-6})
            # Only do the optimization of the context distribution if the performance threshold has not yet been reached
            # even once
            elif not self.perf_lb_reached:
                print("Optimizing performance")
                constraints = [kl_constraint]

                # Define the objective plus Jacobian
                def objective(x):
                    dist = GaussianTorchDistribution.from_weights(self.context_dim, x, dtype=torch.float64)
                    perf = self._compute_expected_performance(dist, contexts_t, old_c_log_prob_t, c_val_t)
                    mu_grad, chol_flat_grad = torch.autograd.grad(perf, dist.parameters())

                    return -perf.detach().numpy(), \
                           -np.concatenate([mu_grad.detach().numpy(), chol_flat_grad.detach().numpy()]).astype(
                               np.float64)

                res = minimize(objective, x0, method="trust-constr", jac=True, bounds=bounds,
                               constraints=constraints, options={"gtol": 1e-4, "xtol": 1e-6})
            else:
                res = None
        except Exception as e:
            os.makedirs("opt_errors", exist_ok=True)
            with open(os.path.join("opt_errors", "error_" + str(time.time())), "wb") as f:
                pickle.dump((self.context_dist.get_weights(), contexts, values), f)
            print("Exception occurred during optimization! Storing state and re-raising!")
            raise e

        if res is not None and res.success:
            self.context_dist.set_weights(res.x)
        elif res is not None:
            # If it was not a success, but the objective value was improved and the bounds are still valid, we still
            # use the result
            old_f = objective(self.context_dist.get_weights())[0]
            cons_ok = True
            for con in constraints:
                cons_ok = cons_ok and con.lb <= con.fun(res.x) <= con.ub

            std_ok = bounds is None or (np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub))
            if cons_ok and std_ok and res.fun < old_f:
                self.context_dist.set_weights(res.x)
            else:
                print(
                    "Warning! Context optimihation unsuccessful - will keep old values. Message: %s" % res.message)

    def sample(self):
        sample = self.context_dist.sample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])