from typing import Callable

import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D

DEFAULT_R3_MAX_ITERATIONS = 1


class R3Adaptation1D(AdaptationInterface1D):
    """
    Daw, Arka, et al. "Mitigating propagation failures in physics-informed neural
    networks using retain-resample-release (r3) sampling."
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_R3_MAX_ITERATIONS,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = old_x.detach().clone().requires_grad_(True)[1:-1]
        for _ in range(self.max_iterations):
            residual_function_values = loss_function(x).abs().reshape(-1)
            threshold = residual_function_values.mean()
            retained_x = x[residual_function_values > threshold]
            num_points_to_sample = x.shape[0] - retained_x.shape[0]
            if num_points_to_sample > 0:
                random_x = torch.empty(num_points_to_sample, x.shape[1]).uniform_(*self.x_range).to(x.device)
                x = torch.cat([retained_x, random_x])
            else:
                x = retained_x
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return "r3" if self.max_iterations == DEFAULT_R3_MAX_ITERATIONS else f"r3_{self.max_iterations}"
