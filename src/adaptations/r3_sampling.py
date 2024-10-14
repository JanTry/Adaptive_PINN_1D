from typing import Callable
from src.helpers.adaptation_interface import AdaptationInterface
import src.params.params as params

import torch


class R3Adaptation:
    def __init__(
        self,
        x_range: tuple[float, float],
        base_points: torch.Tensor,
        max_number_of_points=params.NUM_MAX_POINTS,
    ) -> None:
        self.x_range = x_range
        self.max_iterations = params.DEFAULT_R3_MAX_ITERATIONS

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        x = old_x.detach().clone().requires_grad_(True)
        for _ in range(self.max_iterations):
            residual_function_values = loss_function(x).abs().reshape(-1)
            threshold = residual_function_values.mean()
            retained_x = x[residual_function_values > threshold]
            random_x = torch.empty(
                old_x.numel() - retained_x.numel(), old_x.shape[1]
            ).uniform_(*self.x_range)
            x = torch.cat([retained_x, random_x])
        return x
