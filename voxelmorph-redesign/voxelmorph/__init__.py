# ---- voxelmorph ----
# unsupervised learning for image registration

from . import generators
from . import py
from .py.utils import default_unet_features


# import backend-dependent submodules
backend = py.utils.get_backend()
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    from . import torch
    from .torch import layers
    from .torch import networks
    from .torch import losses
else:
    # tensorflow is default backend
    from . import tf
    from .tf import layers
    from .tf import networks
    from .tf import losses
