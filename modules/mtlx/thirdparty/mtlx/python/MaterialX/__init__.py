from .main import *
from .colorspace import *

try:
    from .legacy import *
except ImportError:
    pass

__version__ = getVersionString()
