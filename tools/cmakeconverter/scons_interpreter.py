from typing import Any, Dict, List, Optional, Union
import os
import sys
from mock_scons import Action, Builder, Node, Scanner, __version__ as scons_raw_version

class SConsError(Exception):
    """Base exception for SCons interpretation errors"""
    pass

class UnsupportedFeatureError(SConsError):
    """Raised when encountering SCons features that are not yet implemented"""
    pass

def EnsureSConsVersion(major: int, minor: int) -> None:
    """Ensure SCons version meets minimum requirements"""
    # For now, we just log this requirement
    print(f"Required SCons version: {major}.{minor}")

def EnsurePythonVersion(major: int, minor: int) -> None:
    """Ensure Python version meets minimum requirements"""
    current_major = sys.version_info.major
    current_minor = sys.version_info.minor
    if current_major < major or (current_major == major and current_minor < minor):
        raise SConsError(f"Python {major}.{minor} or later required, current version: {current_major}.{current_minor}")

class Variable:
    """Base class for SCons variables"""
    def __init__(self, key: str, help: str, default: Any):
        self.key = key
        self.help = help
        self.default = default
        
    def convert(self, value: Any) -> Any:
        return value

class EnumVariable(Variable):
    """Variable with enumerated values"""
    def __init__(self, key: str, help: str, default: str, allowed_values: tuple, map=None):
        super().__init__(key, help, default)
        self.allowed_values = allowed_values
        self.map = map or {}
        
    def convert(self, value: Any) -> Any:
        if value not in self.allowed_values and value not in self.map:
            raise SConsError(f"Invalid value for {self.key}: {value}. Allowed values: {self.allowed_values}")
        return self.map.get(value, value)

class BoolVariable(Variable):
    """Variable with boolean values"""
    def __init__(self, key: str, help: str, default: bool):
        super().__init__(key, help, default)
        
    def convert(self, value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in ('1', 'true', 'yes', 'on')
        return bool(value)

class Variables:
    """Container for SCons variables"""
    def __init__(self, files=None, args=None):
        self.variables: Dict[str, Variable] = {}
        self.values: Dict[str, Any] = {}
        self.files = files or []
        self.args = args or {}
        
    def Add(self, *args, **kwargs):
        """Add a new variable"""
        if len(args) == 1 and isinstance(args[0], Variable):
            var = args[0]
            self.variables[var.key] = var
            self.values[var.key] = var.default
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            key, help, default = args[0]
            var = Variable(key, help, default)
            self.variables[key] = var
            self.values[key] = default
        else:
            raise SConsError("Unsupported variable format")
            
    def Update(self, env):
        """Update environment with variable values"""
        # First apply defaults
        for key, var in self.variables.items():
            env[key] = var.default
            
        # Then apply file values (not implemented yet)
        
        # Finally apply command line arguments
        for key, value in self.args.items():
            if key in self.variables:
                env[key] = self.variables[key].convert(value)

class SConsEnvironment:
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.builders: Dict[str, Any] = {}
        self.tools: List[str] = []
        self.env_dict: Dict[str, Any] = os.environ.copy()
        
    def Append(self, **kwargs):
        """Append values to construction variables"""
        for key, value in kwargs.items():
            if key not in self.variables:
                self.variables[key] = []
            if isinstance(value, (list, tuple)):
                self.variables[key].extend(value)
            else:
                self.variables[key].append(value)
                
    def Prepend(self, **kwargs):
        """Prepend values to construction variables"""
        for key, value in kwargs.items():
            if key not in self.variables:
                self.variables[key] = []
            if isinstance(value, (list, tuple)):
                self.variables[key] = list(value) + self.variables[key]
            else:
                self.variables[key].insert(0, value)

    def Replace(self, **kwargs):
        """Replace construction variables"""
        for key, value in kwargs.items():
            self.variables[key] = value

    def Clone(self):
        """Create a clone of the current environment"""
        new_env = SConsEnvironment()
        new_env.variables = self.variables.copy()
        new_env.builders = self.builders.copy()
        new_env.tools = self.tools.copy()
        new_env.env_dict = self.env_dict.copy()
        return new_env

class SConsInterpreter:
    def __init__(self, project_root: str = None):
        self.default_env = SConsEnvironment()
        self.environments: Dict[str, SConsEnvironment] = {}
        self.variables: Dict[str, Any] = {}
        self.targets: Dict[str, Any] = {}
        self.current_script_dir = ""
        self.project_root = project_root
        
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
                'Variables': Variables,
                'EnumVariable': EnumVariable,
                'BoolVariable': BoolVariable,
                'File': lambda x: Node(x),
                'Dir': lambda x: Node(x),
                'Action': Action,
                'Builder': Builder,
                'Scanner': Scanner
            }
            
            # Execute the SCons script
            exec(content, global_dict)
            
        except Exception as e:
            raise SConsError(f"Error interpreting {filepath}: {str(e)}")
    
    def Export(self, *args, **kwargs):
        """Export variables to other SConscript files"""
        if len(args) == 1 and isinstance(args[0], str):
            # Export('var1 var2 var3')
            var_names = args[0].split()
            for var in var_names:
                if var in self.variables:
                    self.variables[f"exported_{var}"] = self.variables[var]
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
                else:
                    raise SConsError(f"Variable '{var}' not found in exports")
        else:
            raise SConsError("Unsupported Import() format")
    
    def SConscript(self, scripts, *args, **kwargs):
        """Process subsidiary SConscript files"""
        if isinstance(scripts, str):
            scripts = [scripts]
        
        for script in scripts:
            script_path = os.path.join(self.current_script_dir, script)
            if not os.path.exists(script_path):
                raise SConsError(f"SConscript file not found: {script_path}")
            
            self.interpret_file(script_path)

def create_interpreter() -> SConsInterpreter:
    """Create and return a new SCons interpreter instance"""
    return SConsInterpreter()