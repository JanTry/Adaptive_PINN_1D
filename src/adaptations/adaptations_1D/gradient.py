from math import sqrt
from typing import Callable

import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D
from src.adaptations.adaptations_1D.de import mirror_bounds


class GradientDescentAdaptation1D(AdaptationInterface1D):
    def __init__(self, tau: float = 0.002, k: int = 2) -> None:
        self.tau = tau
        self.k = k

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = old_x.detach().clone().requires_grad_(True)[1:-1]
        residual_function_values = loss_function(x).abs().pow(self.k).reshape(-1)
        residual_gradient_values = torch.autograd.grad(
            residual_function_values,
            x,
            grad_outputs=torch.ones_like(residual_function_values),
        )[0]
        x = x + self.tau / 2 * residual_gradient_values.reshape(-1, 1)
        x = mirror_bounds(x, self.x_range[0], self.x_range[1])
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().requires_grad_(True)

    def __str__(self) -> str:
        return "gradient_descent"


class LangevinAdaptation1D(AdaptationInterface1D):
    def __init__(self, beta: float = 0.001, tau: float = 0.002, k: int = 2) -> None:
        self.beta = beta
        self.tau = tau
        self.k = k

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = old_x.detach().clone().requires_grad_(True)[1:-1]
        residual_function_values = loss_function(x).abs().pow(self.k).reshape(-1)
        residual_gradient_values = torch.autograd.grad(
            residual_function_values,
            x,
            grad_outputs=torch.ones_like(residual_function_values),
        )[0]
        x = (
            x
            + self.tau / 2 * residual_gradient_values.reshape(-1, 1)
            + self.beta * sqrt(self.tau) * torch.randn(x.shape[1], device=x.device, dtype=x.dtype)
        )
        x = mirror_bounds(x, self.x_range[0], self.x_range[1])
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().requires_grad_(True)

    def __str__(self) -> str:
        return "langevin"
