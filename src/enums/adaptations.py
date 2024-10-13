from enum import Enum

class EAdaptations(str, Enum):
    NO_ADAPTATION = 'no_adaptation'
    MIDDLE_POINT = 'middle_point'  #  The old version os the adaptation
    DENSITY_SAMPLING = 'density_sampling'  #  Pick randomly in elements based on the error for each element

