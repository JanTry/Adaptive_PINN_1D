import torch
from src.base.pinn_2D_core import PINN_2D, dfdx, dfdy, f
from src.helpers.problem_interface import ProblemInterface2D
from src.helpers.separate_boundary_points_2D import separate_boundary_points_2D
from src.params.params_2D import DEVICE
from torch import cos, pi, pow, tan


class Tan05Problem2D(ProblemInterface2D):
    def __init__(self):
        self.x_range = torch.tensor([0.0, pi / 2.0], device=DEVICE)
        self.y_range = torch.tensor([0.0, pi / 2.0], device=DEVICE)

    def get_range(self) -> (torch.Tensor, torch.Tensor):
        return self.x_range, self.y_range

    def exact_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return tan(x - 0.5) * tan(y - 0.5)

    def f_inner_loss(self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D) -> torch.Tensor:
        return (
            dfdx(pinn, x, y, order=2)
            + dfdy(pinn, x, y, order=2)
            - 2 * tan(x - 0.5) * tan(y - 0.5) * (1 / pow(cos(x - 0.5), 2) + 1 / pow(cos(y - 0.5), 2))
        )

    def __f_boundary_loss_mean(self, x, y, pinn):
        left_boundary_x = torch.masked_select(x, torch.isin(x, self.x_range[0])).reshape((-1, 1))
        left_boundary_y = torch.masked_select(y, torch.isin(x, self.x_range[0])).reshape((-1, 1))
        right_boundary_x = torch.masked_select(x, torch.isin(x, self.x_range[1])).reshape((-1, 1))
        right_boundary_y = torch.masked_select(y, torch.isin(x, self.x_range[1])).reshape((-1, 1))
        top_boundary_x = torch.masked_select(x, torch.isin(y, self.y_range[1])).reshape((-1, 1))
        top_boundary_y = torch.masked_select(y, torch.isin(y, self.y_range[1])).reshape((-1, 1))
        bottom_boundary_x = torch.masked_select(x, torch.isin(y, self.y_range[0])).reshape((-1, 1))
        bottom_boundary_y = torch.masked_select(y, torch.isin(y, self.y_range[0])).reshape((-1, 1))

        left_boundary_loss = f(pinn, left_boundary_x, left_boundary_y) + 0.546302 * tan(left_boundary_y - 0.5)
        bottom_boundary_loss = f(pinn, bottom_boundary_x, bottom_boundary_y) + 0.546302 * tan(bottom_boundary_x - 0.5)
        #
        top_boundary_loss = (
            dfdx(pinn, top_boundary_x, top_boundary_y)
            + dfdy(pinn, top_boundary_x, top_boundary_y)
            - 1.64466 / pow(cos(top_boundary_x - 0.5), 2)
            - 4.35069 * tan(top_boundary_x - 0.5)
        )
        right_boundary_loss = (
            dfdy(pinn, right_boundary_x, right_boundary_y)
            + dfdx(pinn, right_boundary_x, right_boundary_y)
            - 1.64466 / pow(cos(right_boundary_y - 0.5), 2)
            - 4.35069 * tan(right_boundary_y - 0.5)
        )
        #
        #
        return (
            left_boundary_loss.pow(2).mean()
            + right_boundary_loss.pow(2).mean()
            + bottom_boundary_loss.pow(2).mean()
            + top_boundary_loss.pow(2).mean()
        )

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D) -> torch.Tensor:
        inner_x, inner_y, boundary_x, boundary_y = separate_boundary_points_2D(x, y, self.x_range, self.y_range)

        interior_loss = self.f_inner_loss(x=inner_x, y=inner_y, pinn=pinn)
        boundary_loss = self.__f_boundary_loss_mean(x=boundary_x, y=boundary_y, pinn=pinn)

        return interior_loss.pow(2).mean() + boundary_loss
