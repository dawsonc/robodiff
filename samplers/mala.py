from functools import reduce
from math import sqrt

import torch
from torch.optim import Optimizer


class MALAOptimizer(Optimizer):
    """Implements the MALA sampling algorithm. Only supports one parameter group.

    Implementation based on torch.optim.LBFGS.
    """

    def __init__(self, params, lr=1e-3):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(MALAOptimizer, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError(
                "MALA doesn't support per-parameter options " "(parameter groups)"
            )

        self._params = self.param_groups[0]["params"]

        self._state = self.state[self._params[0]]
        if "function_calls" not in self._state:
            self._state["function_calls"] = 0
        if "num_accepts" not in self._state:
            self._state["num_accepts"] = 0
        if "steps" not in self._state:
            self._state["steps"] = 0

        self._numel_cache = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(
                lambda total, p: total + p.numel(), self._params, 0
            )
        return self._numel_cache

    def _acceptance_probability(
        self,
        old_state,
        old_loss,
        old_grad,
        new_state,
        new_loss,
        new_grad,
        lr,
    ):
        """Compute the probability of accepting a move from old state to new state."""
        # The likelihood is e^-loss, so the loglikelihood is -loss
        log_likelihood_ratio = old_loss - new_loss  # log(e^-new_loss / e^-old_loss)

        log_transition_old_to_new = -torch.norm(
            new_state - old_state + lr * old_grad
        ) ** 2 / (4 * lr)
        log_transition_new_to_old = -torch.norm(
            old_state - new_state + lr * new_grad
        ) ** 2 / (4 * lr)
        log_transition_ratio = log_transition_new_to_old - log_transition_old_to_new

        log_acceptance_probability = log_likelihood_ratio + log_transition_ratio
        acceptance_probability = torch.exp(log_acceptance_probability)
        acceptance_probability = torch.min(torch.tensor(1.0), acceptance_probability)
        return acceptance_probability

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _add_to_params(self, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.add_(update[offset : offset + numel].view_as(p))
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("Closure must be provided for MALA optimizer")

        # Make sure that the closure is always evaluated with gradients
        closure = torch.enable_grad()(closure)

        # MALA
        # Re-use loss and gradients from a previous step if they are available
        if "loss" in self._state and "grad" in self._state:
            loss = self._state["loss"]
            grad = self._state["grad"]
        else:
            # Get the original loss
            loss = closure()
            self._state["function_calls"] += 1
            grad = self._gather_flat_grad()

        # Get the learning rate and loss scale (assume only one param group)
        group = self.param_groups[0]
        lr = group["lr"]

        # Update the parameters using Langevin dynamics
        noise = torch.randn_like(grad)

        old_state = torch.cat([p.data.detach().clone() for p in group["params"]])
        proposed_move = -lr * grad + sqrt(2 * lr) * noise
        self._add_to_params(proposed_move)
        new_state = torch.cat([p.data.detach().clone() for p in group["params"]])

        # Use the Metropolis acceptance criterion to decide whether to accept
        # the update. This requires evaluating the loss again.
        new_loss = closure()
        new_grad = self._gather_flat_grad()
        acceptance_prob = self._acceptance_probability(
            old_state,
            loss,
            grad,
            new_state,
            new_loss,
            new_grad,
            lr,
        )
        if torch.rand(1).item() < acceptance_prob:
            # Accept the update
            loss = new_loss
            self._state["num_accepts"] += 1
            self._state["loss"] = new_loss
            self._state["grad"] = new_grad
        else:
            # Reject the update, revert the parameters
            self._add_to_params(-proposed_move)
            self._state["loss"] = loss
            self._state["grad"] = grad

        self._state["steps"] += 1

        print(
            f"Acceptance rate: {self._state['num_accepts'] / self._state['steps']:.2}"
        )

        return loss
