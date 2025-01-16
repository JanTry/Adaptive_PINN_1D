from typing import Callable

import torch


def exit_criterion_1D(base_x: torch.Tensor, loss_fun: Callable, tol: float):
    x = base_x.detach().clone().requires_grad_(True)

    for x1, x2 in zip(x[:-1], x[1:]):
        int_x = torch.linspace(x1.item(), x2.item(), 20).requires_grad_(True).reshape(-1, 1).to(x.device)
        int_y = loss_fun(x=int_x) ** 2
        el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
        # el_loss = quad(loss_np, x1.item(), x2.item())[0] / (x2 - x1)
        if el_loss > tol:
            return False

    return True
