"""Type hints for SCons interpreter"""
from typing import Any, Dict, List, Optional, Union, Callable
from .base import Node, FileNode, DirNode
from .environment import SConsEnvironment
from .builders import Action, Builder
from .variables import Variables, EnumVariable, BoolVariable, PathVariable, ListVariable

# Global functions
def GetSConsVersion() -> str:
    return "4.0.0"

def EnsurePythonVersion(major: int, minor: int) -> None:
    pass

def EnsureSConsVersion(major: int, minor: int) -> None:
    pass

def Exit(msg: str = None) -> None:
    pass

def GetLaunchDir() -> str:
    return ""

def SConscriptChdir(chdir: bool) -> None:
    pass

# Environment functions
def AddPostAction(target: Union[str, Node], action: Union[str, Callable]) -> None:
    pass

def AddPreAction(target: Union[str, Node], action: Union[str, Callable]) -> None:
    pass

def Alias(target: str, source: List[Union[str, Node]] = None) -> None:
    pass

def AlwaysBuild(*targets: Union[str, Node]) -> None:
    pass

def CacheDir(path: str) -> None:
    pass

def Clean(target: Union[str, Node], source: Union[str, Node]) -> None:
    pass

def Command(target: Union[str, List[str]], source: Union[str, List[str]], action: Union[str, Callable]) -> List[Node]:
    return []

def Decider(function: Union[str, Callable]) -> None:
    pass

def Depends(target: Union[str, Node], dependency: Union[str, Node]) -> None:
    pass

def Dir(name: str) -> DirNode:
    return DirNode(name)

def Entry(name: str) -> Node:
    return Node(name)

def Execute(action: Union[str, Callable]) -> int:
    return 0

def File(name: str) -> FileNode:
    return FileNode(name)

def FindFile(file: str, dirs: List[str]) -> Optional[FileNode]:
    return None

def Glob(pattern: str) -> List[Node]:
    return []

def Install(dir: str, source: Union[str, List[str]]) -> List[Node]:
    return []

def InstallAs(target: Union[str, List[str]], source: Union[str, List[str]]) -> List[Node]:
    return []

def Value(value: Any) -> Node:
    return Node(str(value))

def VariantDir(variant_dir: str, src_dir: str, duplicate: bool = True) -> None:
    pass

# Make all these available at module level
__all__ = [
    'GetSConsVersion', 'EnsurePythonVersion', 'EnsureSConsVersion',
    'Exit', 'GetLaunchDir', 'SConscriptChdir',
    'AddPostAction', 'AddPreAction', 'Alias', 'AlwaysBuild',
    'CacheDir', 'Clean', 'Command', 'Decider', 'Depends',
    'Dir', 'Entry', 'Execute', 'File', 'FindFile',
    'Glob', 'Install', 'InstallAs', 'Value', 'VariantDir',
]