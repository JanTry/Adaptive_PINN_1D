from typing import Callable
import torch
import src.params.params as params


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


class DEAdaptation:
    def __init__(
        self,
        x_range: tuple[float, float],
        base_points: torch.Tensor,
        max_number_of_points: int = params.NUM_MAX_POINTS,
    ) -> None:
        self.x_range = x_range
        self.max_iterations = params.DEFAULT_DE_MAX_ITERATIONS
        self.F = params.DEFAULT_DE_F
        self.CR = params.DEFAULT_DE_CR

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        # Exclude boundary points
        x = old_x.detach().clone()[1:-1]
        x = x.requires_grad_(True)

        population_size, number_of_dimensions = x.shape

        for _ in range(self.max_iterations):
            r1, r2, r3 = self.__generate_indices(population_size, x.device)
            v = x[r1] + self.F * (x[r2] - x[r3])
            v = mirror_bounds(v, self.x_range[0], self.x_range[1])
            rand = torch.rand(population_size, number_of_dimensions, device=x.device)
            mask = rand < self.CR
            u = torch.where(mask, v, x)
            f_u = loss_function(u).abs().view(-1)
            f_x = loss_function(x).abs().view(-1)
            improved = f_u >= f_x
            x = torch.where(improved.unsqueeze(1), u, x)

        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __generate_indices(self, population_size: int, device: torch.device):
        idxs = torch.arange(population_size, device=device)
        idxs_repeat = idxs.repeat(population_size, 1)
        idxs_no_i = idxs_repeat[idxs_repeat != idxs.unsqueeze(1)].view(
            population_size, population_size - 1
        )

        r1 = torch.zeros(population_size, dtype=torch.long, device=device)
        r2 = torch.zeros(population_size, dtype=torch.long, device=device)
        r3 = torch.zeros(population_size, dtype=torch.long, device=device)

        for i in range(population_size):
            perm = torch.randperm(population_size - 1, device=device)
            r1[i], r2[i], r3[i] = idxs_no_i[i][perm[:3]]

        return r1, r2, r3
