import json

from src.adaptations import (
    DEAdaptation,
    MiddlePointAdaptation,
    R3Adaptation,
    NoAdaptation,
    HMSAdaptation,
    DensitySamplingAdaptation,
)
from src.enums.problems import EProblems
from src.plots.plot_specific_run import plot_specific_run
from src.runners.adaptive_PINN import train_PINN

PROBLEM_TYPES = [
    # EProblems.DIFFUSION,
    # EProblems.TAN_03,
    EProblems.P07_001,
]
ADAPTATIONS = [
    # NoAdaptation(),
    # MiddlePointAdaptation(),
    # DensitySamplingAdaptation(),
    R3Adaptation(),
    # HMSAdaptation(),
    DEAdaptation(),
]
types_to_time = {}
NUM_RUNS = 15

for problem_type in PROBLEM_TYPES:
    for adaptation in ADAPTATIONS:
        for i in range(NUM_RUNS):
            train_PINN(i, problem_type=problem_type, adaptation=adaptation)
        # for i in range(NUM_RUNS):
        #     plot_specific_run(
        #         run_id=i,
        #         problem_type=problem_type,
        #         adaptation=adaptation_type,
        #         plot_training_points=True,
        #     )
