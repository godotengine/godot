"""SCons interpreter package"""
from typing import Any, Dict, List, Optional, Union
import os
import sys

from .base import (
    SConsError, UnsupportedFeatureError, BuildError,
    EnsureSConsVersion, EnsurePythonVersion,
    Node, FileNode, DirNode
)
from .environment import SConsEnvironment
from .variables import Variable, EnumVariable, BoolVariable, PathVariable, ListVariable, Variables
from .builders import Action, Builder, Program, StaticLibrary, SharedLibrary, Object
from .utils import (
    WhereIs, is_String, is_List, is_Dict,
    flatten, to_String, to_List, to_Dict
)

class SConsInterpreter:
    """Main SCons interpreter class"""
    def __init__(self, project_root: Optional[str] = None):
        self.default_env = SConsEnvironment()
        self.environments: Dict[str, SConsEnvironment] = {}
        self.variables: Dict[str, Any] = {}
        self.targets: Dict[str, Any] = {}
        self.current_script_dir = ""
        self.project_root = project_root
        self._dict = {}
        
        if project_root:
            # Add project root to Python path for imports
            sys.path.insert(0, project_root)
            
        # Initialize basic SCons variables
        self.variables.update({
            'ARGUMENTS': {},
            'COMMAND_LINE_TARGETS': [],
            'BUILD_DIR': 'build',
            'ENV': os.environ.copy(),
        })
        
    def Environment(self, **kwargs) -> SConsEnvironment:
        """Create a new construction environment"""
        env = SConsEnvironment()
        
        # Process tools
        tools = kwargs.get('tools', ['default'])
        if tools:
            env.tools.extend(tools)
            
        # Process other variables
        for key, value in kwargs.items():
            if key != 'tools':
                env.variables[key] = value
                
        return env
    
    def interpret_file(self, filepath: str) -> None:
        """Interpret a SCons script file"""
        try:
            self.current_script_dir = os.path.dirname(filepath)
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Create a new global namespace for execution
            opts = Variables()
            
            global_dict = {
                'Environment': self.Environment,
                'ARGUMENTS': self.variables.get('ARGUMENTS', {}),
                'COMMAND_LINE_TARGETS': self.variables.get('COMMAND_LINE_TARGETS', []),
                'DefaultEnvironment': lambda: self.default_env,
                'Export': self.Export,
                'Import': self.Import,
                'SConscript': self.SConscript,
                'EnsureSConsVersion': EnsureSConsVersion,
                'EnsurePythonVersion': EnsurePythonVersion,
                'Variables': lambda *args: opts,  # Return the same instance
                'EnumVariable': EnumVariable,
                'BoolVariable': BoolVariable,
                'PathVariable': PathVariable,
                'ListVariable': ListVariable,
                'File': lambda x: FileNode(x),
                'Dir': lambda x: DirNode(x),
                'Action': Action,
                'Builder': Builder,
                'Program': Program,
                'StaticLibrary': StaticLibrary,
                'SharedLibrary': SharedLibrary,
                'Object': Object,
                'WhereIs': WhereIs,
                'Help': lambda x: None,  # We don't need to process help text for CMake
            }
            
            # Execute the SCons script
            exec(content, global_dict)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise SConsError(f"Error interpreting {filepath}: {str(e)}")
    
    def Export(self, *args, **kwargs):
        """Export variables to other SConscript files"""
        if len(args) == 1 and isinstance(args[0], str):
            # Export('var1 var2 var3')
            var_names = args[0].split()
            for var in var_names:
                if var in self._dict:
                    self.variables[f"exported_{var}"] = self._dict[var]
                elif var in self.variables:
                    self.variables[f"exported_{var}"] = self.variables[var]
                else:
                    # Create an empty export
                    self.variables[f"exported_{var}"] = None
        elif len(args) == 1 and isinstance(args[0], dict):
            # Export({'var1': value1, 'var2': value2})
            for key, value in args[0].items():
                self.variables[f"exported_{key}"] = value
        elif kwargs:
            # Export(var1=value1, var2=value2)
            for key, value in kwargs.items():
                self.variables[f"exported_{key}"] = value
        else:
            raise SConsError("Unsupported Export() format")
    
    def Import(self, *args):
        """Import variables from other SConscript files"""
        if len(args) == 1 and isinstance(args[0], str):
            # Import('var1 var2 var3')
            var_names = args[0].split()
            for var in var_names:
                exported_var = f"exported_{var}"
                if exported_var in self.variables:
                    self.variables[var] = self.variables[exported_var]
                    self._dict[var] = self.variables[exported_var]
                else:
                    # Create an empty import
                    self.variables[var] = None
                    self._dict[var] = None
        else:
            raise SConsError("Unsupported Import() format")
    
    def SConscript(self, scripts, *args, **kwargs):
        """Process subsidiary SConscript files"""
        if isinstance(scripts, str):
            scripts = [scripts]
        
        # Get exports from kwargs
        exports = kwargs.get('exports', [])
        if isinstance(exports, str):
            exports = [exports]
        elif isinstance(exports, dict):
            exports = list(exports.keys())
            
        # Export variables
        for var in exports:
            if var in self.variables:
                self.variables[f"exported_{var}"] = self.variables[var]
            elif var in self._dict:
                self.variables[f"exported_{var}"] = self._dict[var]
            else:
                self.variables[f"exported_{var}"] = None
        
        for script in scripts:
            script_path = os.path.join(self.current_script_dir, script)
            if not os.path.exists(script_path):
                raise SConsError(f"SConscript file not found: {script_path}")
            
            # Create a new global namespace for the script
            opts = Variables()
            
            global_dict = {
                'env': self.default_env,
                'Environment': self.Environment,
                'ARGUMENTS': self.variables.get('ARGUMENTS', {}),
                'COMMAND_LINE_TARGETS': self.variables.get('COMMAND_LINE_TARGETS', []),
                'DefaultEnvironment': lambda: self.default_env,
                'Export': self.Export,
                'Import': self.Import,
                'SConscript': self.SConscript,
                'EnsureSConsVersion': EnsureSConsVersion,
                'EnsurePythonVersion': EnsurePythonVersion,
                'Variables': lambda *args: opts,  # Return the same instance
                'EnumVariable': EnumVariable,
                'BoolVariable': BoolVariable,
                'PathVariable': PathVariable,
                'ListVariable': ListVariable,
                'File': lambda x: FileNode(x),
                'Dir': lambda x: DirNode(x),
                'Action': Action,
                'Builder': Builder,
                'Program': Program,
                'StaticLibrary': StaticLibrary,
                'SharedLibrary': SharedLibrary,
                'Object': Object,
                'WhereIs': WhereIs,
                'Help': lambda x: None,  # We don't need to process help text for CMake
            }
            
            # Execute the script
            with open(script_path, 'r') as f:
                content = f.read()
            exec(content, global_dict)