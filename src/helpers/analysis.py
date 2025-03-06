import os

import pandas as pd
import src.params.params as params
import torch
from src.adaptations import (
    DEAdaptation,
    DensitySamplingAdaptation,
    GAAdaptation,
    GBDEAdaptation,
    GradientDescentAdaptation,
    HMSAdaptation,
    LangevinAdaptation,
    MiddlePointAdaptation,
    NoAdaptation,
    R3Adaptation,
    RandomRAdaptation,
    RandomSearchWithSelection,
    SelectionMethod,
    SHADEAdaptation,
)
from src.adaptations.adaptation_interface import AdaptationInterface
from src.enums.problems import EProblems
from src.plots.plot_specific_run import N_ITERS_FILE, TIME_FILE

ALL_ADAPTATIONS = [
    GAAdaptation(elitism_rate=0.1),
    GAAdaptation(max_iterations=5, elitism_rate=0.1),
    GradientDescentAdaptation(),
    LangevinAdaptation(),
    RandomRAdaptation(),
    NoAdaptation(),
    MiddlePointAdaptation(),
    DensitySamplingAdaptation(),
    R3Adaptation(),
    HMSAdaptation(),
    DEAdaptation(),
    SHADEAdaptation(),
    GBDEAdaptation(),
    RandomSearchWithSelection(selection_method=SelectionMethod.ROULETTE),
    RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
]


def get_path(problem_type: EProblems, adaptation: AdaptationInterface) -> str:
    return os.path.join(
        "results",
        problem_type.value,
        str(adaptation),
        f"L{params.LAYERS}_N{params.NEURONS}_" f"P{params.NUM_MAX_POINTS}_E{params.NUMBER_EPOCHS}",
        f"LR{params.LEARNING_RATE}_TOL{params.TOLERANCE}",
    )


def extract_df_from_results(
    adaptations: list[AdaptationInterface] = ALL_ADAPTATIONS,
) -> pd.DataFrame:
    all_rows = []

    for problem_type in EProblems:
        for adaptation in adaptations:
            try:
                path = get_path(problem_type, adaptation)
                runs = sorted(
                    [dir for dir in os.listdir(path) if str.isdigit(dir)],
                    key=lambda x: int(x),
                )
                iterations = [torch.load(os.path.join(path, run, N_ITERS_FILE)) for run in runs]
                times = [torch.load(os.path.join(path, run, TIME_FILE)) for run in runs]
                all_rows.extend(
                    [
                        {
                            "run_id": run,
                            "iterations": iter,
                            "time": time,
                            "problem": problem_type.value,
                            "adaptation": str(adaptation),
                        }
                        for run, iter, time in zip(runs, iterations, times)
                    ]
                )
            except FileNotFoundError:
                pass
    return pd.DataFrame(all_rows)
