from src.enums.problems import Problems1D
from src.problems.problems_1D.diffusion import DiffusionProblem1D
from src.problems.problems_1D.p07_001 import P07001Problem1D
from src.problems.problems_1D.p07_01 import P0701Problem1D
from src.problems.problems_1D.tan_01 import Tan01Problem1D
from src.problems.problems_1D.tan_03 import Tan03Problem1D


def problem_factory_1D(problem: Problems1D):
    problem_classes = {
        Problems1D.DIFFUSION: DiffusionProblem1D,
        Problems1D.TAN_01: Tan01Problem1D,
        Problems1D.TAN_03: Tan03Problem1D,
        Problems1D.P07_01: P0701Problem1D,
        Problems1D.P07_001: P07001Problem1D,
    }

    return problem_classes[problem]()
