from abc import ABC, abstractmethod
from typing import Callable
import torch
from src.adaptations.adaptation_interface import AdaptationInterface
from src.adaptations.de import mirror_bounds

DEFAULT_DE_MAX_ITERATIONS = 1


class Crossover(ABC):
    @abstractmethod
    def apply(self, parents: torch.Tensor) -> torch.Tensor:
        pass


class Mutation(ABC):
    @abstractmethod
    def apply(self, offspring: torch.Tensor) -> torch.Tensor:
        pass

    def set_bounds(self, x_range: tuple[float, float]) -> None:
        self.x_range = x_range


class Selection(ABC):
    @abstractmethod
    def apply(self, population: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        pass


class ArithmeticCrossover(Crossover):
    def apply(self, parents: torch.Tensor) -> torch.Tensor:
        population_size, num_genes = parents.shape
        if population_size % 2 != 0:
            parents = parents[:-1]
        parents_reshaped = parents.view(population_size // 2, 2, num_genes)
        alpha = torch.rand(population_size // 2, 1, device=parents.device)
        offspring = (
            alpha * parents_reshaped[:, 0, :] + (1 - alpha) * parents_reshaped[:, 1, :]
        )
        offspring = offspring.view(population_size // 2, num_genes)
        return offspring


class GaussianMutation(Mutation):
    def __init__(self, mean: float = 0.0, std: float = 0.1, mutation_rate: float = 0.1):
        self.mean = mean
        self.std = std
        self.mutation_rate = mutation_rate

    def apply(self, offspring: torch.Tensor) -> torch.Tensor:
        mutation_mask = (
            torch.rand(offspring.shape, device=offspring.device) < self.mutation_rate
        )
        noise = torch.normal(
            self.mean, self.std, size=offspring.shape, device=offspring.device
        )
        return torch.where(mutation_mask, offspring + noise, offspring)


class UniformMutation(Mutation):
    def __init__(
        self,
        mutation_rate: float = 0.1,
    ):
        self.mutation_rate = mutation_rate

    def apply(self, offspring: torch.Tensor) -> torch.Tensor:
        mutation_mask = (
            torch.rand(offspring.shape, device=offspring.device) < self.mutation_rate
        )
        random_values = torch.empty(offspring.shape, device=offspring.device).uniform_(
            self.x_range[0], self.x_range[1]
        )
        return torch.where(mutation_mask, random_values, offspring)


class TournamentSelection(Selection):
    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size

    def apply(self, population: torch.Tensor, fitness: torch.Tensor) -> torch.Tensor:
        population_size = population.shape[0]
        indices = torch.randint(
            0,
            population_size,
            (population_size * 2, self.tournament_size),
            device=population.device,
        )
        best_indices = torch.argmax(fitness[indices], dim=1)
        selected_indices = indices[torch.arange(indices.size(0)), best_indices]
        return population[selected_indices]


class GA:
    def __init__(
        self,
        max_iterations: int = 1,
        crossover: Crossover = ArithmeticCrossover(),
        mutation: Mutation = UniformMutation(mutation_rate=0.9),
        selection: Selection = TournamentSelection(),
        elitism_rate: float = 0.05,
    ):
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.max_iterations = max_iterations
        self.elitism_rate = elitism_rate

    def __call__(
        self,
        loss_function: Callable,
        population: torch.Tensor,
        x_range: tuple[float, float] = (-1.0, 1.0),
    ) -> torch.Tensor:
        self.mutation.set_bounds(x_range)
        population_size, num_genes = population.shape
        num_elites = max(1, int(self.elitism_rate * population_size))

        for _ in range(self.max_iterations):
            f_x = loss_function(x).abs().view(-1)
            _, sorted_indices = torch.sort(f_x, descending=True)
            elites = x[sorted_indices[:num_elites]].clone()
            parents = self.selection.apply(x, f_x)
            offspring = self.crossover.apply(parents)
            mutated_offspring = self.mutation.apply(offspring)
            random_indices = torch.randperm(mutated_offspring.size(0))[
                : (population_size - num_elites)
            ]
            x = torch.cat(
                [
                    elites,
                    mutated_offspring[random_indices],
                ],
                dim=0,
            )
            x = mirror_bounds(x, x_range[0], x_range[1])

        return x


class GAAdaptation(AdaptationInterface):
    def __init__(
        self,
        max_iterations: int = DEFAULT_DE_MAX_ITERATIONS,
        crossover: Crossover = ArithmeticCrossover(),
        mutation: Mutation = UniformMutation(mutation_rate=0.9),
        selection: Selection = TournamentSelection(),
        elitism_rate: float = 0.05,
    ) -> None:
        self.max_iterations = max_iterations
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.elitism_rate = elitism_rate

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        self.mutation.set_bounds(self.x_range)
        x = old_x.detach().clone()[1:-1]
        # x = torch.empty_like(x).uniform_(*self.x_range)
        x = x.requires_grad_(True)

        population_size, num_genes = x.shape
        num_elites = max(1, int(self.elitism_rate * population_size))

        for _ in range(self.max_iterations):
            f_x = loss_function(x).abs().view(-1)
            _, sorted_indices = torch.sort(f_x, descending=True)
            elites = x[sorted_indices[:num_elites]].clone()
            parents = self.selection.apply(x, f_x)
            offspring = self.crossover.apply(parents)
            mutated_offspring = self.mutation.apply(offspring)
            random_indices = torch.randperm(mutated_offspring.size(0))[
                : (population_size - num_elites)
            ]
            x = torch.cat(
                [
                    elites,
                    mutated_offspring[random_indices],
                ],
                dim=0,
            )
            x = mirror_bounds(x, self.x_range[0], self.x_range[1])

        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]])
        refined_x = refined_x.sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return f"ga_{self.elitism_rate}_{self.max_iterations}_{self.mutation.mutation_rate}"
