"""Mock SCons.Script module"""
from tools.cmakeconverter.interpreter.variables import Variables, EnumVariable, BoolVariable, PathVariable, ListVariable
from tools.cmakeconverter.interpreter.builders import Action, Builder, Program, StaticLibrary, SharedLibrary, Object

# Global variables
ARGUMENTS = {}
COMMAND_LINE_TARGETS = []
BUILD_TARGETS = []
DEFAULT_TARGETS = []

# Functions
def GetOption(name: str):
    """Get a command-line option"""
    return None

def SetOption(name: str, value):
    """Set a command-line option"""
    pass

def AddOption(*args, **kwargs):
    """Add a command-line option"""
    pass

def GetBuildFailures():
    """Get list of build failures"""
    return []

def Progress(*args, **kwargs):
    """Set up progress display"""
    pass

def Exit(msg=None):
    """Exit SCons with optional message"""
    pass