from dataclasses import dataclass

import numpy as np

from ._envmap_internal import VoitEnvmap
from ._vectors import Vec3


@dataclass
class LightConf:
    """
    Specify the lighting configuration of an object. All lights share the same coordinate system.

    Parameters
    ----------
    envmap
        The environment map.
    dir_light_dir_and_intensity
        The direction and frequency-wise intensity of the directional light.
    """

    envmap: VoitEnvmap | None
    dir_light_dir_and_color: tuple[Vec3, Vec3] | None
