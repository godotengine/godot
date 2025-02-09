"""Variable handling for SCons interpreter"""
from typing import Any, Dict, List, Optional, Union, Tuple
import sys
from .base import SConsError

# Get the module name for the current package
_package = __package__.split('.')[0] if __package__ else ''

class Variable:
    """Base class for SCons variables"""
    def __init__(self, key: str, help: str, default: Any):
        if isinstance(key, (list, tuple)):
            self.key = key[0]
            self.aliases = key[1:]
        else:
            self.key = key
            self.aliases = []
        self.help = help
        self.default = default
        
    def convert(self, value: Any) -> Any:
        """Convert a value to the appropriate type"""
        return value
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.key!r}, {self.help!r}, {self.default!r})"

class EnumVariable(Variable):
    """Variable with enumerated values"""
    def __init__(self, key: str, help: str, default: str, allowed_values: tuple = None, map=None):
        super().__init__(key, help, default)
        self.allowed_values = allowed_values or ()
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

class PathVariable(Variable):
    """Variable representing a file system path"""
    def __init__(self, key: str, help: str, default: str, validator=None):
        super().__init__(key, help, default)
        self.validator = validator
        
    def convert(self, value: Any) -> str:
        value = str(value)
        if self.validator:
            value = self.validator(value)
        return value

class ListVariable(Variable):
    """Variable containing a list of values"""
    def __init__(self, key: str, help: str, default: List[Any], converter=None):
        super().__init__(key, help, default)
        self.converter = converter
        
    def convert(self, value: Any) -> List[Any]:
        if isinstance(value, str):
            value = value.split()
        if not isinstance(value, (list, tuple)):
            value = [value]
        if self.converter:
            value = [self.converter(v) for v in value]
        return list(value)

class Variables:
    """Container for SCons variables"""
    def __init__(self, files=None, args=None):
        self.variables: Dict[str, Variable] = {}
        self.values: Dict[str, Any] = {}
        self.files = files or []
        self.args = args or {}
        
    def Add(self, *args, **kwargs):
        """Add a new variable"""
        print(f"Adding variable: args={args}, kwargs={kwargs}")
        print(f"Type of first arg: {type(args[0]) if args else None}")
        
        # First check if we're dealing with a variable instance
        if len(args) == 1 and hasattr(args[0], 'key') and hasattr(args[0], 'help') and hasattr(args[0], 'default'):
            # Add(Variable(...))
            var = args[0]
            self.variables[var.key] = var
            self.values[var.key] = var.default
            for alias in var.aliases:
                self.variables[alias] = var
            return var
            
        # Then check if we're dealing with a tuple/list format
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            # Add(['VAR', 'help', 'default']) or Add((['VAR', 'alias'], 'help', 'default'))
            if len(args[0]) == 3:
                key, help, default = args[0]
                var = Variable(key, help, default)
                self.variables[var.key] = var
                self.values[var.key] = default
                for alias in var.aliases:
                    self.variables[alias] = var
                return var
            elif len(args[0]) == 2:
                # Handle case where first element is a tuple of name and aliases
                key, help = args[0]
                var = Variable(key, help, None)
                self.variables[var.key] = var
                self.values[var.key] = None
                for alias in var.aliases:
                    self.variables[alias] = var
                return var
            else:
                raise SConsError(f"Invalid variable tuple length: {len(args[0])}")
                
        # Then check if we're dealing with direct arguments
        elif len(args) == 3:
            # Add('VAR', 'help', 'default')
            key, help, default = args
            var = Variable(key, help, default)
            self.variables[var.key] = var
            self.values[var.key] = default
            for alias in var.aliases:
                self.variables[alias] = var
            return var
            
        elif len(args) == 2:
            # Add('VAR', 'help')
            key, help = args
            var = Variable(key, help, None)
            self.variables[var.key] = var
            self.values[var.key] = None
            for alias in var.aliases:
                self.variables[alias] = var
            return var
            
        # Then check if we're dealing with a dictionary
        elif len(args) == 1 and isinstance(args[0], dict):
            # Add({'VAR': 'default'})
            result = []
            for key, value in args[0].items():
                var = Variable(key, f"Variable {key}", value)
                self.variables[var.key] = var
                self.values[var.key] = value
                result.append(var)
            return result[0] if len(result) == 1 else result
            
        # Then check if we're dealing with keyword arguments
        elif kwargs:
            # Add(VAR='default')
            result = []
            for key, value in kwargs.items():
                var = Variable(key, f"Variable {key}", value)
                self.variables[var.key] = var
                self.values[var.key] = value
                result.append(var)
            return result[0] if len(result) == 1 else result
            
        else:
            raise SConsError(f"Unsupported variable format: args={args}, kwargs={kwargs}")
            
    def __call__(self, *args, **kwargs):
        """Allow the Variables object to be called like a function"""
        return self
            
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to variables"""
        if name == 'Add':
            return self.Add
        elif name == 'BoolVariable':
            return BoolVariable
        elif name == 'EnumVariable':
            return EnumVariable
        elif name == 'PathVariable':
            return PathVariable
        elif name == 'ListVariable':
            return ListVariable
        raise AttributeError(f"'Variables' object has no attribute '{name}'")
            
    def Update(self, env, args=None):
        """Update environment with variable values"""
        # First apply defaults
        for key, var in self.variables.items():
            env[key] = var.default
            
        # Then apply file values (not implemented yet)
        
        # Finally apply command line arguments
        args_dict = args or self.args
        for key, value in args_dict.items():
            if key in self.variables:
                env[key] = self.variables[key].convert(value)
                
    def GenerateHelpText(self, env, sort=None) -> str:
        """Generate help text for all variables"""
        lines = []
        
        # Get all variables
        vars_list = list(self.variables.items())
        if sort:
            vars_list.sort(key=lambda x: x[0])
            
        # Generate help text
        for key, var in vars_list:
            if hasattr(var, 'help'):
                lines.append(f"{key}: {var.help}")
                if hasattr(var, 'allowed_values'):
                    lines.append(f"    allowed values: {var.allowed_values}")
                lines.append(f"    default value: {var.default}")
                lines.append("")
                
        return "\n".join(lines)
                
    def AddVariables(self, *args):
        """Add multiple variables at once"""
        for var in args:
            self.Add(var)