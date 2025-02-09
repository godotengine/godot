"""Builder implementations for SCons interpreter"""
from typing import Any, Dict, List, Optional, Union, Callable
import sys
from .base import Node, FileNode, DirNode, SConsError

class Action:
    """Represents a build action"""
    def __init__(self, action: Union[str, Callable], *args, **kwargs):
        self.action = action
        self.args = args
        self.kwargs = kwargs
        
    def __call__(self, target: List[Node], source: List[Node], env: Any) -> int:
        """Execute the action"""
        if isinstance(self.action, str):
            # String action - substitute variables and execute command
            cmd = self.action
            # TODO: Implement variable substitution
            return 0
        else:
            # Python function action
            return self.action(target, source, env)
            
    def __str__(self) -> str:
        if isinstance(self.action, str):
            return self.action
        return f"Python function: {self.action.__name__}"

class Builder:
    """Base class for builders"""
    def __init__(self, action: Optional[Action] = None, prefix: str = '', suffix: str = '',
                 src_suffix: str = '', target_suffix: str = '', **kwargs):
        self.action = action
        self.prefix = prefix
        self.suffix = suffix
        self.src_suffix = src_suffix
        self.target_suffix = target_suffix
        self.kwargs = kwargs
        
    def __call__(self, env: Any, target: Union[str, List[str]], 
                 source: Optional[Union[str, List[str]]] = None, **kwargs) -> List[Node]:
        """Build the target"""
        if isinstance(target, str):
            target = [target]
        if isinstance(source, str):
            source = [source]
            
        # Convert targets and sources to nodes
        target_nodes = [FileNode(t) for t in target]
        source_nodes = [FileNode(s) for s in (source or [])]
        
        # Execute the action if one exists
        if self.action:
            result = self.action(target_nodes, source_nodes, env)
            if result != 0:
                raise SConsError(f"Builder action failed with code {result}")
                
        return target_nodes

class Program(Builder):
    """Builder for executable programs"""
    def __init__(self, **kwargs):
        super().__init__(suffix='.exe' if sys.platform == 'win32' else '', **kwargs)
        
    def __call__(self, env: Any, target: Union[str, List[str]], 
                 source: Optional[Union[str, List[str]]] = None, **kwargs) -> List[Node]:
        """Build the target"""
        if isinstance(target, str):
            target = [target]
        if isinstance(source, str):
            source = [source]
            
        # Convert targets and sources to nodes
        target_nodes = [FileNode(t) for t in target]
        source_nodes = [FileNode(s) for s in (source or [])]
        
        # Execute the action if one exists
        if self.action:
            result = self.action(target_nodes, source_nodes, env)
            if result != 0:
                raise SConsError(f"Builder action failed with code {result}")
                
        return target_nodes

class StaticLibrary(Builder):
    """Builder for static libraries"""
    def __init__(self, **kwargs):
        super().__init__(prefix='lib', suffix='.a', **kwargs)

class SharedLibrary(Builder):
    """Builder for shared libraries"""
    def __init__(self, **kwargs):
        if sys.platform == 'win32':
            prefix = ''
            suffix = '.dll'
        elif sys.platform == 'darwin':
            prefix = 'lib'
            suffix = '.dylib'
        else:
            prefix = 'lib'
            suffix = '.so'
        super().__init__(prefix=prefix, suffix=suffix, **kwargs)

class Object(Builder):
    """Builder for object files"""
    def __init__(self, **kwargs):
        super().__init__(suffix='.o', **kwargs)