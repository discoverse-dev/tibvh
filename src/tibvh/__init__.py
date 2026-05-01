"""TIBVH - Taichi-based Linear BVH Implementation"""

import warnings

from . import geometry, lbvh
from .geometry import aabb_generator, geom_intersection, utils
from .lbvh.aabb import AABB
from .lbvh.lbvh import LBVH

__version__ = "0.1.4"
__author__ = "Yufei Jia"
__email__ = "jyf23@mails.tsinghua.edu.cn"

__title__ = "tibvh"
__description__ = (
    "A high-performance Taichi-based Linear BVH implementation for 3D geometry "
    "processing"
)
__url__ = "https://github.com/discoverse-dev/tibvh"
__license__ = "MIT"

__all__ = [
    "AABB",
    "LBVH",
    "__author__",
    "__email__",
    "__version__",
    "aabb_generator",
    "geom_intersection",
    "geometry",
    "lbvh",
    "utils",
]

# Initialize Taichi backend on import
try:
    import taichi as ti
    from taichi.lang.impl import get_runtime

    # Initialize with CPU by default. Users can still call ti.init(...) first.
    if get_runtime().prog is None:
        ti.init(arch=ti.cpu, debug=False)
except ImportError:
    warnings.warn(
        "Taichi not found. Please install taichi: pip install taichi>=1.6.0",
        ImportWarning,
    )
except Exception as e:
    warnings.warn(
        f"Failed to initialize Taichi: {e}. "
        "You may need to manually initialize with ti.init()",
        RuntimeWarning,
    )
