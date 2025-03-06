from typing import Callable
import torch
import numpy as np
import cma
from src.adaptations.adaptation_interface import AdaptationInterface
import src.params.params as params

DEFAULT_EVAL_CUTOFF = 500
DEFAULT_N_STEPS = 1
RANDOM_SEED = 42


class CMAAdaptation(AdaptationInterface):
    def __init__(self, n_steps: int = DEFAULT_N_STEPS) -> None:
        self.n_steps = n_steps

    def run_cma(
        self,
        x_range: tuple[float, float],
        loss_function: Callable,
        number_of_points: int,
        n_steps: int,
    ) -> np.ndarray:
        """
        Run CMA-ES for `n_steps` iterations starting from a mean in the middle of x_range,
        and produce `number_of_points` samples using ask().
        """

        def objective_function(x: np.ndarray) -> float:
            # Convert to torch tensor
            x_tensor = (
                torch.tensor(x)
                .float()
                .reshape(-1, 1)
                .requires_grad_(True)
                .to(params.DEVICE)
            )
            return float(
                loss_function(x_tensor)
                .abs()
                .detach()
                .to("cpu")
                .numpy()
                .reshape(-1)
                .item()
            )

        x_min, x_max = x_range
        x0 = np.array([(x_min + x_max) / 2.0])
        sigma = (x_max - x_min) * 0.3  # Initial step-size, somewhat arbitrary

        lambda_ = number_of_points

        cma_opts = {
            "seed": RANDOM_SEED,
            "popsize": lambda_,
            "bounds": [[x_min] * 1, [x_max] * 1],
            "verbose": -9,
        }

        es = cma.CMAEvolutionStrategy(x0, sigma, cma_opts)

        for _ in range(n_steps):
            X = es.ask()
            fitness_values = [objective_function(x) for x in X]
            es.tell(X, fitness_values)

        final_solutions = es.ask()

        return np.array(final_solutions)

    def refine(self, loss_function: Callable, old_x: torch.Tensor):
        self.validate_problem_details()
        number_of_points = self.max_number_of_points
        cma_points = self.run_cma(
            x_range=self.x_range,
            loss_function=loss_function,
            number_of_points=number_of_points - 2,
            n_steps=self.n_steps,
        )

        x = torch.tensor(cma_points, device=old_x.device, dtype=old_x.dtype)
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True).to(old_x.device)

    def __str__(self) -> str:
        base_str = "cma"
        if self.n_steps != DEFAULT_N_STEPS:
            base_str += f"_steps{self.n_steps}"
        return base_str
