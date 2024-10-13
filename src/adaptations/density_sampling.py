from typing import Callable
from src.helpers.adaptation_interface import AdaptationInterface
import src.params.params as params

import torch


class DensitySamplingAdaptation(AdaptationInterface):
    def __init__(self, x_range, base_points: torch.Tensor, max_number_of_points=params.NUM_MAX_POINTS, ):
        self.x_range = x_range
        self.base_points = base_points
        self.max_number_of_points = max_number_of_points
        # TODO -> Implement

    def refine(self, loss_function: Callable, points: torch.Tensor):
        return points
        # TODO -> Implement
