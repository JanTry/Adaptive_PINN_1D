from abc import ABC, abstractmethod
from typing import Callable


class Adaptation(ABC):

    def __init__(self, x_range, max_number_of_points):
        self.x_range = x_range
        self.max_number_of_points = max_number_of_points

    @abstractmethod
    def get_new_points(self, loss_function: Callable):
           pass
