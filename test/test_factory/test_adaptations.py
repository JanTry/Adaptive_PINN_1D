import torch
from torch.fx.node import base_types

from src.adaptations.density_sampling import DensitySamplingAdaptation
from src.adaptations.middle_point import MiddlePointAdaptation
from src.adaptations.no_adaptation import NoAdaptation
from src.enums.adaptations import EAdaptations
from src.helpers.factories import adaptation_factory
from src.helpers.adaptation_interface import AdaptationInterface


def test_no_adaptation_type():
    adaptation = EAdaptations.NO_ADAPTATION
    adaptation_class = adaptation_factory(adaptation, torch.Tensor([0,1]), [0, 1])
    assert (adaptation_class.__class__.__name__ == NoAdaptation.__name__)
    assert issubclass(adaptation_class.__class__, AdaptationInterface)


def test_middle_point_type():
    adaptation = EAdaptations.MIDDLE_POINT
    adaptation_class = adaptation_factory(adaptation, torch.Tensor([0,1]), [0, 1])
    assert (adaptation_class.__class__.__name__ == MiddlePointAdaptation.__name__)
    assert issubclass(adaptation_class.__class__, AdaptationInterface)


def test_density_sampling_type():
    adaptation = EAdaptations.DENSITY_SAMPLING
    adaptation_class = adaptation_factory(adaptation, torch.Tensor([0,1]), [0, 1])
    assert (adaptation_class.__class__.__name__ == DensitySamplingAdaptation.__name__)
    assert issubclass(adaptation_class.__class__, AdaptationInterface)


