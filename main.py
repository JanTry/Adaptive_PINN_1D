from src.adaptations.adaptations_1D import (
    DEAdaptation1D,
    DensitySamplingAdaptation1D,
    HMSAdaptation1D,
    MiddlePointAdaptation1D,
    NoAdaptation1D,
    R3Adaptation1D,
    RandomSearchWithSelection,
    SelectionMethod,
)
from src.enums.problems import Problems1D
from src.plots.plots_1D.plot_specific_run import plot_specific_run_1D
from src.runners.adaptive_PINN_1D import train_PINN_1D

PROBLEM_TYPES = [
    Problems1D.DIFFUSION,
    Problems1D.TAN_03,
    Problems1D.P07_01,
]

ADAPTATIONS = [
    DEAdaptation1D(),
    MiddlePointAdaptation1D(),
    DensitySamplingAdaptation1D(),
    RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
    HMSAdaptation1D(),
    NoAdaptation1D(),
    R3Adaptation1D(),
    RandomSearchWithSelection(selection_method=SelectionMethod.ROULETTE),
    RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
]
types_to_time = {}
NUM_RUNS = 10

for problem_type in PROBLEM_TYPES:
    for adaptation in ADAPTATIONS:
        for i in range(NUM_RUNS):
            train_PINN_1D(i, problem_type=problem_type, adaptation=adaptation)
            plot_specific_run_1D(
                run_id=i,
                problem_type=problem_type,
                adaptation=adaptation,
                plot_training_points=True,
            )
