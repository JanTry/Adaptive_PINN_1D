from typing import Callable

import torch
from src.adaptations.adaptation_interface import AdaptationInterface


class RandomRAdaptation(AdaptationInterface):
    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = (
            torch.empty(old_x.shape[0] - 2, old_x.shape[1])
            .uniform_(*self.x_range)
            .to(old_x.device)
        )
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return "random_r"
