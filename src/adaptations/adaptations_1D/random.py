from enum import Enum
from typing import Callable

import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D

DEFAULT_EVAL_CUTOFF = 5000


class SelectionMethod(str, Enum):
    ROULETTE = "roulette"
    TOURNAMENT = "tournament"


def roulette_select(x: torch.Tensor, y: torch.Tensor, num_select: int) -> torch.Tensor:
    min_score = y.min()
    if min_score < 0:
        y = y - min_score

    cumulative_probs = torch.cumsum(y, dim=0)
    total_sum = cumulative_probs[-1]
    selection_points = torch.rand(num_select, device=x.device) * total_sum
    selected_indices = torch.searchsorted(cumulative_probs, selection_points)
    return x[selected_indices]


def tournament_select(x: torch.Tensor, y: torch.Tensor, k: int) -> torch.Tensor:
    num_candidates = x.size(0)
    tournaments = torch.randint(0, num_candidates, (num_candidates, k))
    tournament_scores = y[tournaments]
    best_indices_in_tournament = torch.argmax(tournament_scores, dim=1)
    selected_indices = tournaments[torch.arange(num_candidates), best_indices_in_tournament]
    return x[selected_indices]


SELECTION_METHOD_TO_FUNCTION = {
    SelectionMethod.ROULETTE: roulette_select,
    SelectionMethod.TOURNAMENT: tournament_select,
}


class RandomSearchWithSelection(AdaptationInterface1D):
    def __init__(
        self,
        eval_cutoff: int = DEFAULT_EVAL_CUTOFF,
        selection_method: SelectionMethod = SelectionMethod.ROULETTE,
    ) -> None:
        self.eval_cutoff = eval_cutoff
        self.selection_method = selection_method
        self.selector = SELECTION_METHOD_TO_FUNCTION.get(selection_method)

    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        old_y = loss_function(old_x).abs().reshape(-1)
        random_x = (
            torch.empty(self.eval_cutoff, old_x.shape[1]).uniform_(*self.x_range).requires_grad_(True).to(old_x.device)
        )
        random_y = loss_function(random_x).abs().reshape(-1)
        selected_x = roulette_select(
            torch.cat([random_x, old_x]).reshape(-1),
            torch.cat([random_y, old_y]),
            old_x.shape[0] - 2,
        )
        refined_x = torch.cat([old_x[0:1], selected_x.reshape(-1, 1), old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return (
            f"random_{self.selection_method.value}"
            if self.eval_cutoff == DEFAULT_EVAL_CUTOFF
            else f"random_{self.selection_method.value}_{self.eval_cutoff}"
        )


class RandomRAdaptation1D(AdaptationInterface1D):
    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        # Exclude boundary points
        x = torch.empty(old_x.shape[0] - 2, old_x.shape[1]).uniform_(*self.x_range).to(old_x.device)
        refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
        return refined_x.detach().clone().requires_grad_(True)

    def __str__(self) -> str:
        return "random_r"
