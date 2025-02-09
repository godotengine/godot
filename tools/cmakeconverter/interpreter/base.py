"""Base classes and exceptions for the SCons interpreter"""
from typing import Any, Dict, List, Optional, Union
import os
import sys

class SConsError(Exception):
    """Base exception for SCons interpretation errors"""
    pass

class UnsupportedFeatureError(SConsError):
    """Raised when encountering SCons features that are not yet implemented"""
    pass

class BuildError(SConsError):
    """Raised when a build operation fails"""
    pass

def EnsureSConsVersion(major: int, minor: int) -> None:
    """Ensure SCons version meets minimum requirements"""
    print(f"Required SCons version: {major}.{minor}")

def EnsurePythonVersion(major: int, minor: int) -> None:
    """Ensure Python version meets minimum requirements"""
    current_major = sys.version_info.major
    current_minor = sys.version_info.minor
    if current_major < major or (current_major == major and current_minor < minor):
        raise SConsError(f"Python {major}.{minor} or later required, current version: {current_major}.{current_minor}")

class Node:
    """Base class for SCons nodes (files, directories, etc.)"""
    def __init__(self, path: str):
        self.path = path
        self._abspath = os.path.abspath(path)
        
    def abspath(self) -> str:
        return self._abspath
        
    def exists(self) -> bool:
        """Check if the node exists in the filesystem"""
        return os.path.exists(self._abspath)
        
    def __str__(self) -> str:
        return self.path
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path!r})"

class FileNode(Node):
    """Represents a file in the build system"""
    pass

class DirNode(Node):
    """Represents a directory in the build system"""
    pass