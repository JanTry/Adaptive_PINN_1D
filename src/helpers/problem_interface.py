from abc import ABC, abstractmethod
from src.base.pinn_core import PINN

import torch


class ProblemInterface(ABC):
    @abstractmethod
    def __init__(self):
        """
        Constructor used by factory method
        """
        pass

    @abstractmethod
    def get_range(self) -> [float, float]:
        """
        :return: range of x (omega)
        """
        pass

    @abstractmethod
    def exact_solution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to be used for model validation after the whole PINN learning process is complete
        :param x: tensor with x values that need calculation
        :return: exact solution value
        """
        pass

    @abstractmethod
    def f_inner_loss(self, x: torch.Tensor, pinn: PINN) -> torch.Tensor:
        """
        Calculation of loss function for points that are not on the boundary
        :param x: list of x locations of points
        :param pinn: pinn approximator
        :return: loss function values at given points
        """
        pass


    @abstractmethod
    def compute_loss(self, x: torch.Tensor, pinn: PINN) -> torch.Tensor:
        """
        Calculate final loss for all x elements
        Check the final line for the formula
        :param x: list of x locations of points. Exactly 1 point on each of the boundaries is required
        :param pinn: pinn approximator
        :return: value of the loss function (not just a sum ;) )
        """
        pass

