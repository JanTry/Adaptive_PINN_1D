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
        # We don't want to refine the boundary points:
        x = old_x.detach().clone().requires_grad_(True)[1:-1]
        for _ in range(self.max_iterations):
            residual_function_values = loss_function(x).abs().reshape(-1)
            threshold = residual_function_values.mean()
            retained_x = x[residual_function_values > threshold]
            num_points_to_sample = x.shape[0] - retained_x.shape[0]
            if num_points_to_sample > 0:
                random_x = (
                    torch.empty(num_points_to_sample, x.shape[1])
                    .uniform_(*self.x_range)
                    .to(x.device)
                )
                x = torch.cat([retained_x, random_x])
            else:
                x = retained_x
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)
