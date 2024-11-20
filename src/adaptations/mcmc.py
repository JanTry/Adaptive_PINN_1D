from typing import Callable

import torch
from src.adaptations.adaptation_interface import AdaptationInterface
from src.adaptations.de import mirror_bounds


class MetropolisHastingsAdaptation(AdaptationInterface):
    def __init__(self, proposal_std: float = 0.001, k: int = 2) -> None:
        self.proposal_std = proposal_std
        self.k = k

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        x = old_x.detach().clone().requires_grad_(False)[1:-1]
        proposed_x = x + torch.randn_like(x) * self.proposal_std
        proposed_x = mirror_bounds(proposed_x, self.x_range[0], self.x_range[1])
        current_loss = loss_function(x).pow(self.k).reshape(-1)
        proposed_loss = loss_function(proposed_x).pow(self.k).reshape(-1)
        acceptance_prob = torch.exp(-(proposed_loss - current_loss))
        acceptance_prob = torch.min(acceptance_prob, torch.tensor(1.0, device=x.device))
        random_vals = torch.rand_like(acceptance_prob)
        accept_mask = random_vals < acceptance_prob
        refined_x = torch.where(accept_mask, proposed_x, x)
        refined_x = torch.cat([old_x[0:1], refined_x, old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)
