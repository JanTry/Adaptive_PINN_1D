from src.enums.problems import EProblems
from src.adaptations.adaptation_interface import AdaptationInterface
from src.adaptations import (
    HMSAdaptation,
    R3Adaptation,
    NoAdaptation,
    DEAdaptation,
    DensitySamplingAdaptation,
    MiddlePointAdaptation,
)
import os
import src.params.params as params
import pandas as pd
import torch

N_ITERS_FILE = "n_iters.pt"
TIME_FILE = "exec_time.pt"
POINT_FILE = "point_data.pt"

ALL_ADAPTATIONS = [
    NoAdaptation(),
    MiddlePointAdaptation(),
    DensitySamplingAdaptation(),
    R3Adaptation(),
    HMSAdaptation(),
    DEAdaptation(),
]


def get_path(problem_type: EProblems, adaptation: AdaptationInterface) -> str:
    return os.path.join(
        "results",
        problem_type.value,
        str(adaptation),
        f"L{params.LAYERS}_N{params.NEURONS}_"
        f"P{params.NUM_MAX_POINTS}_E{params.NUMBER_EPOCHS}",
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
                iterations = [
                    torch.load(os.path.join(path, run, N_ITERS_FILE)) for run in runs
                ]
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
