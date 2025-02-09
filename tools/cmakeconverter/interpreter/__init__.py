"""SCons interpreter package"""
from typing import Any, Dict, List, Optional, Union
import os
import sys

from .base import (
    SConsError, UnsupportedFeatureError, BuildError,
    EnsureSConsVersion, EnsurePythonVersion,
    Node, FileNode, DirNode
)
from .functions import Glob, Value, Run, CommandNoCache
from . import detect
from . import platform
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
        self._variant_dir = None  # Current variant directory
        self._default_env_instance = None  # Singleton instance for DefaultEnvironment()
        
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
        
    def _create_global_namespace(self, env: Optional[SConsEnvironment] = None) -> Dict[str, Any]:
        """Create a new global namespace for script execution"""
        opts = Variables()
        
        if env is None:
            env = self.default_env
            
        if self._default_env_instance is None:
            self._default_env_instance = SConsEnvironment()
            
        def Return(*args):
            """Store return value in global namespace"""
            if len(args) == 1:
                global_dict['__return_value__'] = args[0]
            elif len(args) > 1:
                global_dict['__return_value__'] = args
            
        # Import SCons module
        from tools.cmakeconverter import SCons
        
        global_dict = {
            'env': env,
            'Environment': self.Environment,
            'ARGUMENTS': self.variables.get('ARGUMENTS', {}),
            'COMMAND_LINE_TARGETS': self.variables.get('COMMAND_LINE_TARGETS', []),
            'DefaultEnvironment': lambda: self._default_env_instance,
            'Export': self.Export,
            'Import': self.Import,
            'SConscript': self.SConscript,
            'Return': Return,  # Add Return function
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
            'SCons': SCons,  # Add SCons module
            'Glob': Glob,  # Add Glob function
            'Value': Value,  # Add Value function
            'Run': Run,  # Add Run function
            'CommandNoCache': CommandNoCache,  # Add CommandNoCache function
            'detect': detect,  # Add detect module
            'platform_list': platform.platform_list,  # Add platform variables
            'platform_opts': platform.platform_opts,
            'platform_flags': platform.platform_flags,
            'platform_doc_class_path': platform.platform_doc_class_path,
            'platform_exporters': platform.platform_exporters,
            'platform_apis': platform.platform_apis,
            'sys': sys,  # Add sys module
            'os': os,  # Add os module
            'glob': __import__('glob'),  # Add glob module
            'OrderedDict': __import__('collections').OrderedDict,  # Add OrderedDict
            'pickle': __import__('pickle'),  # Add pickle module
            'ModuleType': __import__('types').ModuleType,  # Add ModuleType
            'module_from_spec': __import__('importlib.util').module_from_spec,  # Add module_from_spec
            'spec_from_file_location': __import__('importlib.util').spec_from_file_location,  # Add spec_from_file_location
            '_helper_module': lambda name, path: None,  # Mock _helper_module function
        }
        print(f"DEBUG: Global namespace created with keys: {list(global_dict.keys())}")
        return global_dict
        
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
            prev_dir = self.current_script_dir
            self.current_script_dir = os.path.dirname(filepath)
            
            # Add SCons module to Python path
            converter_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if converter_root not in sys.path:
                sys.path.insert(0, converter_root)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Execute the SCons script
            global_dict = self._create_global_namespace()
            exec(content, global_dict)
            
            self.current_script_dir = prev_dir
            
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
    
    def SConscript(self, scripts, *args, **kwargs) -> Any:
        """Process subsidiary SConscript files"""
        if isinstance(scripts, str):
            scripts = [scripts]
            
        # Get variant directory
        variant_dir = kwargs.get('variant_dir')
        if variant_dir:
            prev_variant_dir = self._variant_dir
            self._variant_dir = variant_dir
            
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
                
        # Process each script
        results = []
        for script in scripts:
            script_path = os.path.join(self.current_script_dir, script)
            if not os.path.exists(script_path):
                raise SConsError(f"SConscript file not found: {script_path}")
                
            # Save current directory
            prev_dir = self.current_script_dir
            self.current_script_dir = os.path.dirname(script_path)
            
            # Create a new global namespace for the script
            global_dict = self._create_global_namespace()
            
            # Execute the script
            with open(script_path, 'r') as f:
                content = f.read()
            exec(content, global_dict)
            
            # Check for Return value
            if 'Return' in content:
                result = global_dict.get('__return_value__')
                if result is not None:
                    results.append(result)
                
            # Restore current directory
            self.current_script_dir = prev_dir
            
        # Restore variant directory
        if variant_dir:
            self._variant_dir = prev_variant_dir
            
        # Return results
        if not results:
            return None
        elif len(results) == 1:
            return results[0]
        else:
            return results