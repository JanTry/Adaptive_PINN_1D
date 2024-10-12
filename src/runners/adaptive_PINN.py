import os
import time
import src.params.params as params
import torch
import logging

from functools import partial
from src.enums.adaptations import EAdaptations
from src.enums.problems import EProblems


def train_PINN(run_id: int, problem:EProblems=params.PROBLEM, adaptation:EAdaptations=params.ADAPTATION, save_results:bool=True):
    """
    Basic 1D PINN training based on src/params/params.py file

    :param run_id: run identification number. Will overwrite any previous result with the same id and save path params
    :param problem: problem enum value. Based on that, a proper class from src/problems will be used
    :param adaptation: adaptation enum value. Based on that, a proper class from src/adaptations will be used
    :param save_results: pretty self-explanatory ;)
    :return:
    """
    logging.log(logging.DEBUG, f'Starting PINN training run {run_id} for: {problem.value}, {adaptation.value}')

    x = torch.linspace(
        X_INI, X_FIN, steps=NUM_MAX_POINTS, requires_grad=True, device=DEVICE
    ).reshape(-1, 1)
