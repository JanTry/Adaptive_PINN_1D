from typing import Callable
from src.helpers.adaptation_interface import AdaptationInterface
import src.params.params as params

import torch


class MiddlePointAdaptation(AdaptationInterface):
    def __init__(self, x_range: [float, float], max_number_of_points: int=params.NUM_MAX_POINTS):
        self.x_range = x_range
        self.max_number_of_points = max_number_of_points
        # TODO -> Implement

    def refine(self, loss_function: Callable, points: torch.Tensor):
        # TODO -> Implement
        pass
