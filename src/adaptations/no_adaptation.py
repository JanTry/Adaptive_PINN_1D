from typing import Callable
from src.helpers.adaptation_interface import AdaptationInterface
import src.params.params as params

import torch


class NoAdaptation(AdaptationInterface):
    def __init__(self, x_range, base_points: torch.Tensor, max_number_of_points):
        pass

    def refine(self, loss_function: Callable, old_x: torch.Tensor):
        return old_x
