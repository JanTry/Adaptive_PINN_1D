import torch

import src.params.params as params

from src.adaptations.density_sampling import DensitySamplingAdaptation
from src.adaptations.middle_point import MiddlePointAdaptation
from src.adaptations.no_adaptation import NoAdaptation
from src.enums.adaptations import EAdaptations
from src.enums.problems import EProblems
from src.problems.diffusion import DiffusionProblem
from src.problems.p07_001 import P07001Problem
from src.problems.p07_01 import P0701Problem
from src.problems.tan_03 import Tan03Problem
from src.problems.tan_01 import Tan01Problem


def problem_factory(problem: EProblems):
    problem_classes = {
        EProblems.DIFFUSION: DiffusionProblem,
        EProblems.TAN_01: Tan01Problem,
        EProblems.TAN_03: Tan03Problem,
        EProblems.P07_01: P0701Problem,
        EProblems.P07_001: P07001Problem,
    }

    return problem_classes[problem]()


def adaptation_factory(adaptation: EAdaptations, base_points: torch.Tensor, x_range: [float, float],
                       max_number_of_points: int=params.NUM_MAX_POINTS):
    adaptation_classes = {
        EAdaptations.NO_ADAPTATION: NoAdaptation,
        EAdaptations.MIDDLE_POINT: MiddlePointAdaptation,
        EAdaptations.DENSITY_SAMPLING: DensitySamplingAdaptation,
    }

    return adaptation_classes[adaptation](x_range, base_points, max_number_of_points)
