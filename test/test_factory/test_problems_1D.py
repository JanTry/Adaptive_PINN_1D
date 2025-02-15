from src.enums.problems import Problems1D
from src.helpers.factories import problem_factory_1D
from src.helpers.problem_interface import ProblemInterface1D
from src.problems.problems_1D.diffusion import DiffusionProblem1D
from src.problems.problems_1D.p07_001 import P07001Problem1D
from src.problems.problems_1D.p07_01 import P0701Problem1D
from src.problems.problems_1D.tan_01 import Tan01Problem1D
from src.problems.problems_1D.tan_03 import Tan03Problem1D


def test_diffusion_type():
    problem = Problems1D.DIFFUSION
    problem_class = problem_factory_1D(problem)
    assert problem_class.__class__.__name__ == DiffusionProblem1D.__name__
    assert issubclass(problem_class.__class__, ProblemInterface1D)


def test_tan01_type():
    problem = Problems1D.TAN_01
    problem_class = problem_factory_1D(problem)
    assert problem_class.__class__.__name__ == Tan01Problem1D.__name__
    assert issubclass(problem_class.__class__, ProblemInterface1D)


def test_tan001_type():
    problem = Problems1D.TAN_03
    problem_class = problem_factory_1D(problem)
    assert problem_class.__class__.__name__ == Tan03Problem1D.__name__
    assert issubclass(problem_class.__class__, ProblemInterface1D)


def test_p0701_type():
    problem = Problems1D.P07_01
    problem_class = problem_factory_1D(problem)
    assert problem_class.__class__.__name__ == P0701Problem1D.__name__
    assert issubclass(problem_class.__class__, ProblemInterface1D)


def test_p07001_type():
    problem = Problems1D.P07_001
    problem_class = problem_factory_1D(problem)
    assert problem_class.__class__.__name__ == P07001Problem1D.__name__
    assert issubclass(problem_class.__class__, ProblemInterface1D)
