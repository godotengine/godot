"""Mock SCons package"""
from . import Script
from . import Util
from . import Variables

# Export commonly used items
from .Script import (
    Variables, EnumVariable, BoolVariable, PathVariable, ListVariable,
    Action, Builder, Program, StaticLibrary, SharedLibrary, Object,
    ARGUMENTS, COMMAND_LINE_TARGETS, BUILD_TARGETS, DEFAULT_TARGETS,
    GetOption, SetOption, AddOption, GetBuildFailures, Progress, Exit
)
from .Util import (
    WhereIs, is_String, is_List, is_Dict,
    flatten, to_String, to_List, to_Dict
)

# Version information
__version__ = "4.0.0"