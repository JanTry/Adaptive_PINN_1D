from typing import Callable

import numpy as np
import numpy.linalg as nla
import src.params.params as params
import torch
from pyhms import (
    SEA,
    DELevelConfig,
    DontStop,
    EALevelConfig,
    EvalCutoffProblem,
    FunctionProblem,
    MetaepochLimit,
    SHADELevelConfig,
    SingularProblemEvalLimitReached,
    hms,
)
from pyhms.core.individual import Individual
from pyhms.core.population import Population
from pyhms.demes.abstract_deme import AbstractDeme
from pyhms.demes.single_pop_eas.multiwinner import CCGreedyPolicy, MultiwinnerSelection, UtilityFunction
from pyhms.sprout.sprout_filters import DemeCandidates, DemeLevelCandidatesFilter
from pyhms.sprout.sprout_mechanisms import BestPerDeme, LevelLimit, SproutMechanism
from src.adaptations.adaptation_interface import AdaptationInterface

DEFAULT_EVAL_CUTOFF = 500
DEFAULT_N_DEMES = 10
RANDOM_SEED = 42


class FarEnough(DemeLevelCandidatesFilter):
    def __init__(self, min_distance: float, norm_ord: int = 2) -> None:
        super().__init__()
        self.min_distance = min_distance
        self.norm_ord = norm_ord

    def _is_far_enough(self, ind: Individual, centroid: np.ndarray):
        return nla.norm(ind.genome - centroid, ord=self.norm_ord) > self.min_distance

    def __call__(
        self,
        candidates: dict[AbstractDeme, DemeCandidates],
        tree,
    ) -> dict[AbstractDeme, DemeCandidates]:
        for deme in candidates.keys():
            child_siblings = [sibling for sibling in tree.levels[deme.level + 1]]
            child_seeds = candidates[deme].individuals
            for sibling in child_siblings:
                child_seeds = [ind for ind in child_seeds if self._is_far_enough(ind, sibling.centroid)]
            candidates[deme].individuals = child_seeds
        return candidates


def stochastic_universal_sampling(population: Population, k: int) -> np.ndarray:
    fitnesses = population.fitnesses
    min_fitness = np.min(fitnesses)
    if min_fitness < 0:
        fitnesses = fitnesses - min_fitness
    total_fitness = np.sum(fitnesses)
    if total_fitness == 0:
        raise ValueError("Total fitness is zero after shifting. Cannot perform sampling.")
    probabilities = fitnesses / total_fitness
    cumulative_probabilities = np.cumsum(probabilities)
    step_size = 1.0 / k
    start_point = np.random.uniform(0, step_size)
    pointers = [start_point + i * step_size for i in range(k)]
    selected_indices = []
    current_pointer = 0
    for pointer in pointers:
        while pointer > cumulative_probabilities[current_pointer]:
            current_pointer += 1
        selected_indices.append(current_pointer)
    return population.genomes[selected_indices]


def naive_sampling(population: Population, k: int) -> np.ndarray:
    indices = np.random.choice(population.size, k, replace=False)
    return population.genomes[indices]


def run_hms(
    x_range: tuple[float, float],
    loss_function: Callable,
    number_of_points: int,
    eval_cutoff: int,
) -> np.ndarray:
    def objective_function(x: np.ndarray) -> float:
        x_tensor = torch.tensor(x).float().reshape(-1, 1).requires_grad_(True).to(params.DEVICE)
        return loss_function(x_tensor).abs().detach().to("cpu").numpy().reshape(-1).item()

    function_problem = FunctionProblem(objective_function, maximize=True, bounds=np.array([x_range]), use_cache=True)
    problem_with_cutoff = EvalCutoffProblem(function_problem, eval_cutoff)
    alpha = 0.01
    beta = 0.5
    config = [
        SHADELevelConfig(
            generations=1,
            problem=problem_with_cutoff,
            pop_size=number_of_points // 2,
            lsc=DontStop(),
            memory_size=2,
        ),
        EALevelConfig(
            ea_class=SEA,
            generations=1,
            problem=problem_with_cutoff,
            pop_size=number_of_points // 8,
            mutation_std=(x_range[1] - x_range[0]) * alpha * beta**2,
            lsc=MetaepochLimit(5),
        ),
    ]
    global_stop_condition = SingularProblemEvalLimitReached(eval_cutoff)
    level_limit = 5
    sprout_condition = SproutMechanism(
        BestPerDeme(),
        [FarEnough((x_range[1] - x_range[0] / DEFAULT_N_DEMES), 2)],
        [LevelLimit(level_limit)],
    )
    hms_tree = hms(
        config,
        global_stop_condition,
        sprout_condition,
        {"random_seed": RANDOM_SEED},
    )
    all_individuals = hms_tree.all_individuals
    return naive_sampling(Population.from_individuals(all_individuals), number_of_points)


class HMSAdaptation(AdaptationInterface):
    def __init__(self, eval_cutoff: int = DEFAULT_EVAL_CUTOFF) -> None:
        self.eval_cutoff = eval_cutoff

    def refine(self, loss_function: Callable, old_x: torch.Tensor):
        self.validate_problem_details()
        x_np = run_hms(
            x_range=self.x_range,
            loss_function=loss_function,
            number_of_points=self.max_number_of_points - 2,
            eval_cutoff=self.eval_cutoff,
        )
        x = torch.tensor(x_np, device=old_x.device, dtype=old_x.dtype)
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort(dim=0)[0]
        return refined_x.detach().clone().requires_grad_(True).to(old_x.device)

    def __str__(self) -> str:
        return "hms_sus" if self.eval_cutoff == DEFAULT_EVAL_CUTOFF else f"hms_sus_{self.eval_cutoff}"
