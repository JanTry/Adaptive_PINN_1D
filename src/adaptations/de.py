from typing import Callable

import torch
from src.adaptations.adaptation_interface import AdaptationInterface

DEFAULT_DE_MAX_ITERATIONS = 1
DEFAULT_DE_F = 0.8
DEFAULT_DE_CR = 0.9


def mirror_bounds(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Handles boundary constraints using mirror method.
    When a value exceeds the boundary, it is reflected back into the valid range.
    """
    range_size = upper - lower
    x_shifted = x - lower
    quotient = torch.floor(x_shifted / range_size)
    remainder = x_shifted % range_size
    mirrored = torch.where(quotient % 2 == 1, range_size - remainder, remainder)
    return mirrored + lower


class DEAdaptation(AdaptationInterface):
    def __init__(
        self,
        max_iterations: int = DEFAULT_DE_MAX_ITERATIONS,
        f: float = DEFAULT_DE_F,
        cr: float = DEFAULT_DE_CR,
    ) -> None:
        self.max_iterations = max_iterations
        self.f = f
        self.cr = cr

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = old_x.detach().clone()[1:-1]
        x = x.requires_grad_(True)

        population_size, number_of_dimensions = x.shape

        for _ in range(self.max_iterations):
            r1, r2, r3 = self.__generate_indices(population_size, x.device)
            v = x[r1] + self.f * (x[r2] - x[r3])
            v = mirror_bounds(v, self.x_range[0], self.x_range[1])
            rand = torch.rand(population_size, number_of_dimensions, device=x.device)
            mask = rand < self.cr
            u = torch.where(mask, v, x)
            f_u = loss_function(u).abs().view(-1)
            f_x = loss_function(x).abs().view(-1)
            improved = f_u >= f_x
            x = torch.where(improved.unsqueeze(1), u, x)

        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __generate_indices(self, population_size: int, device: torch.device):
        idxs = torch.arange(population_size, device=device)
        idxs_repeat = idxs.repeat(population_size, 1)
        idxs_no_i = idxs_repeat[idxs_repeat != idxs.unsqueeze(1)].view(population_size, population_size - 1)

        r1 = torch.zeros(population_size, dtype=torch.long, device=device)
        r2 = torch.zeros(population_size, dtype=torch.long, device=device)
        r3 = torch.zeros(population_size, dtype=torch.long, device=device)

        for i in range(population_size):
            perm = torch.randperm(population_size - 1, device=device)
            r1[i], r2[i], r3[i] = idxs_no_i[i][perm[:3]]

        return r1, r2, r3

    def __str__(self) -> str:
        return "de" if self.max_iterations == DEFAULT_DE_MAX_ITERATIONS else f"de_{self.max_iterations}"


class StaticDEAdaptation(AdaptationInterface):
    def __init__(
        self,
        max_iterations: int = DEFAULT_DE_MAX_ITERATIONS,
        f: float = DEFAULT_DE_F,
        cr: float = DEFAULT_DE_CR,
    ) -> None:
        self.max_iterations = max_iterations
        self.f = f
        self.cr = cr

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        x = torch.linspace(
            self.x_range[0],
            self.x_range[1],
            steps=old_x.shape[0] - 2,
            device=old_x.device,
        ).reshape(-1, 1)
        x = x.requires_grad_(True)

        population_size, number_of_dimensions = x.shape

        for _ in range(self.max_iterations):
            r1, r2, r3 = self.__generate_indices(population_size, x.device)
            v = x[r1] + self.f * (x[r2] - x[r3])
            v = mirror_bounds(v, self.x_range[0], self.x_range[1])
            rand = torch.rand(population_size, number_of_dimensions, device=x.device)
            mask = rand < self.cr
            u = torch.where(mask, v, x)
            f_u = loss_function(u).abs().view(-1)
            f_x = loss_function(x).abs().view(-1)
            improved = f_u >= f_x
            x = torch.where(improved.unsqueeze(1), u, x)

        refined_x = torch.cat(
            [
                old_x[:1],
                x,
                old_x[-1:],
            ]
        ).sort(
            dim=0
        )[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __generate_indices(self, population_size: int, device: torch.device):
        idxs = torch.arange(population_size, device=device)
        idxs_repeat = idxs.repeat(population_size, 1)
        idxs_no_i = idxs_repeat[idxs_repeat != idxs.unsqueeze(1)].view(population_size, population_size - 1)

        r1 = torch.zeros(population_size, dtype=torch.long, device=device)
        r2 = torch.zeros(population_size, dtype=torch.long, device=device)
        r3 = torch.zeros(population_size, dtype=torch.long, device=device)

        for i in range(population_size):
            perm = torch.randperm(population_size - 1, device=device)
            r1[i], r2[i], r3[i] = idxs_no_i[i][perm[:3]]

        return r1, r2, r3

    def __str__(self) -> str:
        return "static_de" if self.max_iterations == DEFAULT_DE_MAX_ITERATIONS else f"static_de_{self.max_iterations}"


class SHADEAdaptation(AdaptationInterface):
    def __init__(
        self,
        max_iterations: int = DEFAULT_DE_MAX_ITERATIONS,
        cr: float = DEFAULT_DE_CR,
        f: float = DEFAULT_DE_F,
    ) -> None:
        self.max_iterations = max_iterations
        self.cr = cr
        self.f = f

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = old_x.detach().clone()[1:-1]
        x = x.requires_grad_(True)

        fitness = loss_function(x).abs().view(-1)
        sorted_indices = torch.argsort(fitness, descending=True)
        x = x[sorted_indices]

        population_size, number_of_dimensions = x.shape

        for _ in range(self.max_iterations):
            # Mutation: DE/current-to-pbest/1
            p_best_indices = torch.randint(0, max(1, int(0.1 * population_size)), (population_size,))
            r1, r2 = self.__generate_indices(population_size, x.device)
            v = x + self.f * (x[p_best_indices] - x + x[r1] - x[r2])
            v = mirror_bounds(v, self.x_range[0], self.x_range[1])

            rand = torch.rand(population_size, number_of_dimensions, device=x.device)
            mask = rand < self.cr
            u = torch.where(mask, v, x)

            f_u = loss_function(u).abs().view(-1)
            f_x = loss_function(x).abs().view(-1)
            improved = f_u >= f_x
            x = torch.where(improved.unsqueeze(1), u, x)

        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __generate_indices(self, population_size: int, device: torch.device):
        idxs = torch.arange(population_size, device=device)
        idxs_repeat = idxs.repeat(population_size, 1)
        idxs_no_i = idxs_repeat[idxs_repeat != idxs.unsqueeze(1)].view(population_size, population_size - 1)

        r1 = torch.zeros(population_size, dtype=torch.long, device=device)
        r2 = torch.zeros(population_size, dtype=torch.long, device=device)

        for i in range(population_size):
            perm = torch.randperm(population_size - 1, device=device)
            r1[i], r2[i] = idxs_no_i[i][perm[:2]]

        return r1, r2

    def __str__(self) -> str:
        return "SHADE" if self.max_iterations == DEFAULT_DE_MAX_ITERATIONS else f"SHADE_{self.max_iterations}"


class GBDEAdaptation(AdaptationInterface):
    def __init__(
        self,
        max_iterations: int = DEFAULT_DE_MAX_ITERATIONS,
        f: float = DEFAULT_DE_F,
        cr: float = DEFAULT_DE_CR,
        gradient_steps: int = 1,
        gradient_lr: float = 0.001,  # Learning rate for gradient refinement
        k: float = 2.0,  # Residual power factor
    ) -> None:
        self.max_iterations = max_iterations
        self.f = f
        self.cr = cr
        self.gradient_steps = gradient_steps
        self.gradient_lr = gradient_lr
        self.k = k

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # x = old_x.detach().clone()[1:-1]
        # x = x.requires_grad_(True)
        x = torch.linspace(
            self.x_range[0],
            self.x_range[1],
            steps=old_x.shape[0] - 2,
            device=old_x.device,
        ).reshape(-1, 1)
        x = x.requires_grad_(True)

        population_size, number_of_dimensions = x.shape

        for _ in range(self.max_iterations):
            r1, r2, r3 = self.__generate_indices(population_size, x.device)
            v = x[r1] + self.f * (x[r2] - x[r3])
            v = mirror_bounds(v, self.x_range[0], self.x_range[1])

            rand = torch.rand(population_size, number_of_dimensions, device=x.device)
            mask = rand < self.cr
            u = torch.where(mask, v, x)

            u = self.apply_gradient_refinement(u, loss_function)

            f_u = loss_function(u).abs().view(-1)
            f_x = loss_function(x).abs().view(-1)
            improved = f_u >= f_x
            x = torch.where(improved.unsqueeze(1), u, x)

        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True)

    def apply_gradient_refinement(self, candidates: torch.Tensor, loss_function: Callable) -> torch.Tensor:
        refined_candidates = candidates.clone().detach().requires_grad_(True)

        for _ in range(self.gradient_steps):
            residual_function_values = loss_function(refined_candidates).abs().pow(self.k).view(-1)

            residual_gradient_values = torch.autograd.grad(
                residual_function_values,
                refined_candidates,
                grad_outputs=torch.ones_like(residual_function_values),
                retain_graph=True,
            )[0]

            refined_candidates = refined_candidates + self.gradient_lr * residual_gradient_values
            refined_candidates = mirror_bounds(refined_candidates, self.x_range[0], self.x_range[1])
        return refined_candidates.detach().requires_grad_(True)

    def __generate_indices(self, population_size: int, device: torch.device):
        idxs = torch.arange(population_size, device=device)
        idxs_repeat = idxs.repeat(population_size, 1)
        idxs_no_i = idxs_repeat[idxs_repeat != idxs.unsqueeze(1)].view(population_size, population_size - 1)

        r1 = torch.zeros(population_size, dtype=torch.long, device=device)
        r2 = torch.zeros(population_size, dtype=torch.long, device=device)
        r3 = torch.zeros(population_size, dtype=torch.long, device=device)

        for i in range(population_size):
            perm = torch.randperm(population_size - 1, device=device)
            r1[i], r2[i], r3[i] = idxs_no_i[i][perm[:3]]

        return r1, r2, r3

    def __str__(self) -> str:
        return "gb-de" if self.max_iterations == DEFAULT_DE_MAX_ITERATIONS else f"gb-de_{self.max_iterations}"
