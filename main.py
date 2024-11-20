from src.adaptations import (
    DEAdaptation,
    DensitySamplingAdaptation,
    HMSAdaptation,
    MiddlePointAdaptation,
    NoAdaptation,
    R3Adaptation,
    RandomSearchWithSelection,
    SelectionMethod,
)
from src.enums.problems import EProblems
from src.plots.plot_specific_run import plot_specific_run
from src.runners.adaptive_PINN import train_PINN

PROBLEM_TYPES = [
    EProblems.DIFFUSION,
    EProblems.TAN_03,
    EProblems.P07_01,
]

ADAPTATIONS = [
    DEAdaptation(),
    MiddlePointAdaptation(),
    DensitySamplingAdaptation(),
    RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
    HMSAdaptation(),
    NoAdaptation(),
    R3Adaptation(),
    RandomSearchWithSelection(selection_method=SelectionMethod.ROULETTE),
    RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
]
types_to_time = {}
NUM_RUNS = 10

for problem_type in PROBLEM_TYPES:
    for adaptation in ADAPTATIONS:
        for i in range(NUM_RUNS):
            train_PINN(i, problem_type=problem_type, adaptation=adaptation)
            plot_specific_run(
                run_id=i,
                problem_type=problem_type,
                adaptation=adaptation,
                plot_training_points=True,
            )
