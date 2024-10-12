from src.enums.adaptations import EAdaptations
from src.enums.problems import EProblems
import logging

PROBLEM = EProblems.DIFFUSION
ADAPTATION = EAdaptations.NO_ADAPTATION

# Collocation points limits
NUM_BASE_POINTS = 20 # DEF 20
NUM_MAX_POINTS = 200 # DEF 200
NUM_TEST_POINTS = 20 # DEF 20


# PINN settings
LAYERS = 3  # DEF 3
NEURONS = 15  # DEF 15
LEARNING_RATE = 0.005  # DEF 0.005


# RUN settings
TOLERANCE = 1e-4  # DEF 1e-4, error tolerance
NUMBER_EPOCHS = 1000  # DEF 1000, number of EPOCHS per 1 adaptation iteration
MAX_ITERS = 1000  # DEF 1000, maximum number of iterations for a run
LOG_LEVEL = logging.DEBUG  # DEF logging.INFO


# Problem/adaptation specific values
MAX_DEPTH = 10 # DEF 10, required for some of the adaptations
EPSILON = 0.1 # DEF 0.1, required for the Advection Diffusion problem





