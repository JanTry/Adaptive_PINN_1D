from enum import Enum


class EProblems(str, Enum):
    DIFFUSION = "diffusion"  # Advection diffusion
    TAN_01 = "tan_01"  # tan(x+0.1)
    TAN_03 = "tan_03"  # tan(x+0.3)
    P07_01 = "P07_01"  # (x+0.1)^0.7
    P07_001 = "P07_001"  # (x+0.01)^0.7
