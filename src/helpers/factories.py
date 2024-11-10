from src.enums.problems import EProblems
from src.problems.diffusion import DiffusionProblem
from src.problems.p07_001 import P07001Problem
from src.problems.p07_01 import P0701Problem
from src.problems.tan_01 import Tan01Problem
from src.problems.tan_03 import Tan03Problem


def problem_factory(problem: EProblems):
    problem_classes = {
        EProblems.DIFFUSION: DiffusionProblem,
        EProblems.TAN_01: Tan01Problem,
        EProblems.TAN_03: Tan03Problem,
        EProblems.P07_01: P0701Problem,
        EProblems.P07_001: P07001Problem,
    }

    return problem_classes[problem]()
