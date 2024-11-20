from src.enums.problems import EProblems
from src.helpers.factories import problem_factory
from src.helpers.problem_interface import ProblemInterface
from src.problems.diffusion import DiffusionProblem
from src.problems.p07_001 import P07001Problem
from src.problems.p07_01 import P0701Problem
from src.problems.tan_01 import Tan01Problem
from src.problems.tan_03 import Tan03Problem


def test_diffusion_type():
    problem = EProblems.DIFFUSION
    problem_class = problem_factory(problem)
    assert problem_class.__class__.__name__ == DiffusionProblem.__name__
    assert issubclass(problem_class.__class__, ProblemInterface)


def test_tan01_type():
    problem = EProblems.TAN_01
    problem_class = problem_factory(problem)
    assert problem_class.__class__.__name__ == Tan01Problem.__name__
    assert issubclass(problem_class.__class__, ProblemInterface)


def test_tan001_type():
    problem = EProblems.TAN_03
    problem_class = problem_factory(problem)
    assert problem_class.__class__.__name__ == Tan03Problem.__name__
    assert issubclass(problem_class.__class__, ProblemInterface)


def test_p0701_type():
    problem = EProblems.P07_01
    problem_class = problem_factory(problem)
    assert problem_class.__class__.__name__ == P0701Problem.__name__
    assert issubclass(problem_class.__class__, ProblemInterface)


def test_p07001_type():
    problem = EProblems.P07_001
    problem_class = problem_factory(problem)
    assert problem_class.__class__.__name__ == P07001Problem.__name__
    assert issubclass(problem_class.__class__, ProblemInterface)
