from abc import ABC, abstractmethod
from typing import Callable

import torch


class AdaptationInterface(ABC):
    @abstractmethod
    def __init__(self, x_range: [float, float], base_points: torch.Tensor, max_number_of_points: int):
        pass

    @abstractmethod
    def refine(self, loss_function: Callable, previous_points: torch.Tensor):
           pass
