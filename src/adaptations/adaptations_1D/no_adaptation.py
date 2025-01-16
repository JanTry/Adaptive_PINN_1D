from typing import Callable

import torch
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface


class NoAdaptation(AdaptationInterface):
    def refine(self, loss_function: Callable, old_x: torch.Tensor) -> torch.Tensor:
        self.validate_problem_details()
        return old_x

    def __str__(self) -> str:
        return "no_adaptation"
