"""
ac2D - Active Contour 2D Module

This package provides functionality for working with active contours in 2D images,
including force calculations, interpolation methods, point selection, and more.
"""

from .forces import *
from .interpolation import *
from .points_selection import *
from .utils import *
from .vfc import *

# `from ac2D import *`
__all__ = [
    # forces exports
    "compute_A",
    "balloon_force",
    "smooth_forces",
    # interpolation exports
    "interp_snake2",
    "splines_interpolation2d",
    # vfc exports
    "create_vfc_kernel_2d",
    "apply_vfc_2d",
    # points_selection exports
    "subdivision",
    "polygon_parity",
    "dist_points",
    "init_rectangle",
    "init_circle",
    "init_elipse",
    # utils exports
    "resize_with_aspect_ratio",
]

# Version info
__version__ = "0.1.0"
