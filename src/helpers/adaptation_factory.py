from src.adaptations.density_sampling import DensitySamplingAdaptation
from src.adaptations.middle_point import MiddlePointAdaptation
from src.adaptations.no_adaptation import NoAdaptation
from src.enums.adaptations import EAdaptations

def adaptation_factory(adaptation: EAdaptations):
    adaptation_classes = {
        EAdaptations.NO_ADAPTATION: NoAdaptation,
        EAdaptations.MIDDLE_POINT: MiddlePointAdaptation,
        EAdaptations.DENSITY_SAMPLING: DensitySamplingAdaptation,
    }

    return adaptation_classes[adaptation]
