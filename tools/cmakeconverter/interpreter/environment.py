"""SCons Environment implementation"""
from typing import Any, Dict, List, Optional, Union, Callable
import os
import copy
from . import methods
from .base import SConsError, Node, FileNode, DirNode
from .variables import Variable

class SConsEnvironment:
    """Represents a construction environment in SCons"""
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.builders: List[Dict[str, Any]] = []
        self.tools: List[str] = []
        self.env_dict: Dict[str, str] = os.environ.copy()
        self.disabled_classes: List[str] = []
        self.core_sources: List[str] = []
        self.source_files: Dict[str, List[str]] = {}
        self._dict = {}
        self.version_info = {
            'short_name': 'godot',
            'name': 'Godot Engine',
            'major': 4,
            'minor': 4,
            'patch': 0,
            'status': 'beta',
            'build': 'custom_build',
            'module_config': '',
            'website': 'https://godotengine.org',
            'docs_branch': 'latest',
            'git_hash': '',
            'git_timestamp': 0,
        }
        self._dict = {
            'disabled_classes': self.disabled_classes,
            'version_info': self.version_info,
            'ENV': self.env_dict,
            'BUILDERS': {},
            'TOOLS': self.tools,
            'platform': '',
            'arch': 'x86_64',  # Default to x86_64
            'bits': '64',
            'CXX': os.environ.get('CXX', 'g++'),
            'CC': os.environ.get('CC', 'gcc'),
            'LINK': os.environ.get('LINK', 'g++'),
            'CFLAGS': [],
            'CXXFLAGS': [],
            'LINKFLAGS': [],
            'CCFLAGS': [],
            'CPPDEFINES': [],
            'CPPPATH': [],
            'LIBS': [],
            'LIBPATH': [],
            'extra_suffix': '',
            'target': 'editor',
            'optimize': 'speed',
            'debug_symbols': False,
            'separate_debug_symbols': False,
            'debug_paths_relative': False,
            'lto': 'none',
            'production': False,
            'threads': True,
            'deprecated': True,
            'precision': 'single',
            'minizip': True,
            'brotli': True,
            'xaudio2': False,
            'vulkan': True,
            'opengl3': True,
            'd3d12': False,
            'metal': False,
            'openxr': True,
            'use_volk': True,
            'disable_exceptions': True,
            'custom_modules': '',
            'custom_modules_recursive': True,
            'dev_mode': False,
            'tests': False,
            'fast_unsafe': False,
            'ninja': False,
            'ninja_auto_run': True,
            'ninja_file': 'build.ninja',
            'compiledb': False,
            'num_jobs': '',
            'verbose': False,
            'progress': True,
            'warnings': 'all',
            'werror': False,
            'vsproj': False,
            'vsproj_name': 'godot',
            'import_env_vars': '',
            'disable_3d': False,
            'disable_advanced_gui': False,
            'build_profile': '',
            'modules_enabled_by_default': True,
            'no_editor_splash': True,
            'system_certs_path': '',
            'use_precise_math_checks': False,
            'strict_checks': False,
            'scu_build': False,
            'scu_limit': '0',
            'engine_update_check': True,
            'steamapi': False,
            'cache_path': '',
            'cache_limit': '0',
            'module_version_string': '',
            'PROGSUFFIX': '',
            'PROGSUFFIX_WRAP': '',
            'OBJSUFFIX': '.o',
            'version_info': {},
            'suffix': '',
            'LIBSUFFIX': '.a',
            'SHLIBSUFFIX': '.so',
            'LIBSUFFIXES': [],
            'msvc': False,
            'is_msvc': False,
            'is_mingw': False,
            'is_clang': False,
            'is_gcc': True,
        }  # For direct dictionary-like access
        
    def __getattr__(self, name: str) -> Any:
        """Allow attribute access to variables"""
        if name in self._dict:
            value = self._dict[name]
            if value is None:
                if name in {'CXX', 'CC', 'LINK', 'platform', 'arch', 'bits', 'target', 'optimize', 'lto', 'precision', 'warnings',
                          'extra_suffix', 'object_prefix', 'vsproj_name', 'import_env_vars', 'build_profile', 'system_certs_path',
                          'cache_path', 'cache_limit', 'scu_limit', 'custom_modules', 'ninja_file'}:
                    return ''
                if name in {'CFLAGS', 'CXXFLAGS', 'LINKFLAGS', 'CCFLAGS', 'CPPDEFINES', 'CPPPATH', 'LIBS', 'LIBPATH'}:
                    return []
                if name in {'debug_symbols', 'separate_debug_symbols', 'debug_paths_relative', 'production', 'threads', 'deprecated',
                          'minizip', 'brotli', 'xaudio2', 'vulkan', 'opengl3', 'd3d12', 'metal', 'openxr', 'use_volk',
                          'disable_exceptions', 'custom_modules_recursive', 'dev_mode', 'tests', 'fast_unsafe', 'ninja',
                          'ninja_auto_run', 'compiledb', 'verbose', 'progress', 'werror', 'vsproj', 'disable_3d',
                          'disable_advanced_gui', 'modules_enabled_by_default', 'no_editor_splash', 'use_precise_math_checks',
                          'strict_checks', 'scu_build', 'engine_update_check', 'steamapi', 'msvc', 'is_msvc', 'is_mingw',
                          'is_clang', 'is_gcc'}:
                    return False
            return value
        if name in self.variables:
            value = self.variables[name]
            if value is None:
                if name in {'CXX', 'CC', 'LINK', 'platform', 'arch', 'bits', 'target', 'optimize', 'lto', 'precision', 'warnings',
                          'extra_suffix', 'object_prefix', 'vsproj_name', 'import_env_vars', 'build_profile', 'system_certs_path',
                          'cache_path', 'cache_limit', 'scu_limit', 'custom_modules', 'ninja_file'}:
                    return ''
                if name in {'CFLAGS', 'CXXFLAGS', 'LINKFLAGS', 'CCFLAGS', 'CPPDEFINES', 'CPPPATH', 'LIBS', 'LIBPATH'}:
                    return []
                if name in {'debug_symbols', 'separate_debug_symbols', 'debug_paths_relative', 'production', 'threads', 'deprecated',
                          'minizip', 'brotli', 'xaudio2', 'vulkan', 'opengl3', 'd3d12', 'metal', 'openxr', 'use_volk',
                          'disable_exceptions', 'custom_modules_recursive', 'dev_mode', 'tests', 'fast_unsafe', 'ninja',
                          'ninja_auto_run', 'compiledb', 'verbose', 'progress', 'werror', 'vsproj', 'disable_3d',
                          'disable_advanced_gui', 'modules_enabled_by_default', 'no_editor_splash', 'use_precise_math_checks',
                          'strict_checks', 'scu_build', 'engine_update_check', 'steamapi', 'msvc', 'is_msvc', 'is_mingw',
                          'is_clang', 'is_gcc'}:
                    return False
            return value
        # Return empty string/list for common variables if not found
        if name in {'CXX', 'CC', 'LINK', 'platform', 'arch', 'bits', 'target', 'optimize', 'lto', 'precision', 'warnings',
                  'extra_suffix', 'object_prefix', 'vsproj_name', 'import_env_vars', 'build_profile', 'system_certs_path',
                  'cache_path', 'cache_limit', 'scu_limit', 'custom_modules', 'ninja_file'}:
            return ''
        if name in {'CFLAGS', 'CXXFLAGS', 'LINKFLAGS', 'CCFLAGS', 'CPPDEFINES', 'CPPPATH', 'LIBS', 'LIBPATH'}:
            return []
        if name in {'debug_symbols', 'separate_debug_symbols', 'debug_paths_relative', 'production', 'threads', 'deprecated',
                  'minizip', 'brotli', 'xaudio2', 'vulkan', 'opengl3', 'd3d12', 'metal', 'openxr', 'use_volk',
                  'disable_exceptions', 'custom_modules_recursive', 'dev_mode', 'tests', 'fast_unsafe', 'ninja',
                  'ninja_auto_run', 'compiledb', 'verbose', 'progress', 'werror', 'vsproj', 'disable_3d',
                  'disable_advanced_gui', 'modules_enabled_by_default', 'no_editor_splash', 'use_precise_math_checks',
                  'strict_checks', 'scu_build', 'engine_update_check', 'steamapi', 'msvc', 'is_msvc', 'is_mingw',
                  'is_clang', 'is_gcc'}:
            return False
        return None
        
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to variables"""
        return self.__getattr__(key)
        
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator to check if a key exists"""
        return key in self._dict or key in self.variables
        
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like setting of variables"""
        self._dict[key] = value
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a construction variable"""
        # First try _dict, then variables, then return default
        value = self._dict.get(key)
        if value is not None:
            return value
        value = self.variables.get(key)
        if value is not None:
            return value
        return default
        
    def Append(self, **kwargs):
        """Append values to construction variables"""
        for key, value in kwargs.items():
            if key not in self.variables:
                self.variables[key] = []
            elif not isinstance(self.variables[key], list):
                self.variables[key] = [self.variables[key]]
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                self.variables[key].extend(value)
            else:
                self.variables[key].append(value)
                
    def AppendUnique(self, **kwargs):
        """Append values to construction variables, avoiding duplicates"""
        for key, value in kwargs.items():
            if key not in self.variables:
                self.variables[key] = []
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                for v in value:
                    if v not in self.variables[key]:
                        self.variables[key].append(v)
            else:
                if value not in self.variables[key]:
                    self.variables[key].append(value)
                
    def Prepend(self, **kwargs):
        """Prepend values to construction variables"""
        for key, value in kwargs.items():
            if key not in self.variables:
                self.variables[key] = []
            if value is None:
                continue
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
        new_env.variables = copy.deepcopy(self.variables)
        new_env.builders = copy.deepcopy(self.builders)
        new_env.tools = self.tools.copy()
        new_env.env_dict = self.env_dict.copy()
        new_env._dict = copy.deepcopy(self._dict)
        return new_env
        
    def File(self, path: str) -> FileNode:
        """Create a file node"""
        return FileNode(path)
        
    def Dir(self, path: str) -> DirNode:
        """Create a directory node"""
        return DirNode(path)
        
    def PrependENVPath(self, name: str, value: Union[str, List[str], None]):
        """Prepend to an environment path variable"""
        if value is None:
            return
            
        if isinstance(value, str):
            paths = [value]
        else:
            paths = value
            
        current = self.env_dict.get(name, '').split(os.pathsep)
        self.env_dict[name] = os.pathsep.join(paths + [p for p in current if p])
        
    def _get_major_minor_revision(self, version_str: str) -> tuple:
        """Get major, minor, revision numbers from version string"""
        try:
            parts = version_str.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            revision = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, revision)
        except (ValueError, IndexError):
            return (0, 0, 0)
            
    def SConsignFile(self, file: Union[str, Node, None] = None) -> None:
        """Set up the SCons signature database file"""
        # In our case, we just record the file path for CMake
        if file is None:
            self._dict['SCONSIGN_FILE'] = '.sconsign.dblite'
        elif isinstance(file, Node):
            self._dict['SCONSIGN_FILE'] = file.path
        else:
            self._dict['SCONSIGN_FILE'] = str(file)
            
    def GetOption(self, name: str) -> Any:
        """Get a command-line option"""
        # For now, return default values for known options
        defaults = {
            'clean': False,
            'help': False,
            'num_jobs': 1,
            'silent': False,
            'debug': False,
            'verbose': False,
        }
        return defaults.get(name)
        
    def SetOption(self, name: str, value: Any) -> None:
        """Set a command-line option"""
        # Store options in the environment
        if 'OPTIONS' not in self._dict:
            self._dict['OPTIONS'] = {}
        self._dict['OPTIONS'][name] = value
        
    def Dictionary(self) -> Dict[str, Any]:
        """Get a dictionary of all construction variables"""
        result = {}
        result.update(self.variables)
        result.update(self._dict)
        return result
        
    def subst(self, string: str, raw: int = 0, target=None, source=None, conv=None, executor=None) -> str:
        """Substitute construction variables in a string"""
        if string is None:
            return ''
            
        # For now, just do simple variable substitution
        result = string
        for key, value in self._dict.items():
            if value is not None:
                str_value = str(value)
                result = result.replace(f"${key}", str_value)
                result = result.replace(f"${{{key}}}", str_value)
        return result
        
    def Tool(self, tool: str, toolpath=None) -> None:
        """Load a tool"""
        # For now, just record that we want to use this tool
        self.tools.append(tool)
        
    def Decider(self, decider: Union[str, Callable]) -> None:
        """Set the up-to-date decision function"""
        # Store the decider for later use in CMake
        self._dict['DECIDER'] = decider
        
    def Builder(self, action=None, prefix='', suffix='', src_suffix='', target_suffix='', **kwargs) -> Any:
        """Create a new builder"""
        # For now, just record that we want to use this builder
        builder = {
            'action': action,
            'prefix': prefix,
            'suffix': suffix,
            'src_suffix': src_suffix,
            'target_suffix': target_suffix,
            **kwargs
        }
        self.builders.append(builder)
        return builder
        
    def Object(self, source: Union[str, List[str]], **kwargs) -> Any:
        """Create an object file from source files"""
        # For now, just record that we want to build these objects
        if isinstance(source, str):
            source = [source]
        obj = {
            'type': 'object',
            'source': source,
            **kwargs
        }
        self.builders.append(obj)
        return obj
        
    def Program(self, target: Union[str, List[str]], source: Optional[Union[str, List[str]]] = None, **kwargs) -> Any:
        """Create an executable program"""
        # For now, just record that we want to build this program
        if isinstance(target, str):
            target = [target]
        if isinstance(source, str):
            source = [source]
        prog = {
            'type': 'program',
            'target': target,
            'source': source,
            **kwargs
        }
        self.builders.append(prog)
        return prog
        
    def Glob(self, pattern: str) -> List[str]:
        """Find files matching a glob pattern"""
        import glob
        import os
        
        # If pattern is a list, process each pattern
        if isinstance(pattern, (list, tuple)):
            result = []
            for p in pattern:
                result.extend(self.Glob(p))
            return result
            
        # Handle absolute paths
        if os.path.isabs(pattern):
            return glob.glob(pattern)
            
        # Handle relative paths
        base_dir = os.getcwd()
        if hasattr(self, 'current_script_dir') and self.current_script_dir:
            base_dir = self.current_script_dir
            
        pattern = os.path.join(base_dir, pattern)
        return glob.glob(pattern)
        
    def Run(self, func: Callable) -> Callable:
        """Run a function"""
        def wrapper(target, source, env):
            if isinstance(source, list):
                source = [s.path if hasattr(s, 'path') else str(s) for s in source]
            if isinstance(target, list):
                target = [t.path if hasattr(t, 'path') else str(t) for t in target]
            return func(target, source, env)
        return wrapper
        
    def Command(self, target: Union[str, List[str]], source: Union[str, List[str]], action: Union[str, Callable], **kwargs) -> Any:
        """Run a command"""
        if isinstance(action, str):
            def action_func(target, source, env):
                os.system(action)
                return 0
            action = action_func
        return action(target, source, self)
        
    def Value(self, value: Any) -> Any:
        """Create a value node that mimics SCons' Value node behavior"""
        class ValueNode:
            def __init__(self, value):
                self.value = value if value is not None else []
                if not isinstance(self.value, (list, tuple)):
                    self.value = [str(self.value)]
            def read(self):
                """Return the value in a form suitable for iteration"""
                return self.value
            def __getitem__(self, key):
                """Support list indexing, always returning a ValueNode"""
                if isinstance(key, slice):
                    return ValueNode(self.value[key])
                try:
                    return ValueNode(self.value[key])
                except IndexError:
                    return ValueNode([])
            def __len__(self):
                """Support len() operation"""
                return len(self.value)
            def __iter__(self):
                """Support iteration"""
                return iter(self.value)
            def __str__(self):
                """String representation"""
                if len(self.value) == 1:
                    return str(self.value[0])
                return str(self.value)
            def __bool__(self):
                """Truth value testing"""
                return bool(self.value)
            def __call__(self, *args, **kwargs):
                """Support callable nodes"""
                return self
        return ValueNode(value)
        
    def NoCache(self, target: Any) -> Any:
        """Mark a target as not cacheable"""
        return target
        
    def Glob(self, pattern: str) -> List[str]:
        """Return a list of files matching the pattern"""
        print(f"DEBUG: Glob called with pattern={pattern}")
        import glob
        import os
        
        if pattern is None:
            print("DEBUG: Warning: pattern is None")
            return []
            
        print(f"DEBUG: Original pattern: {pattern}")
        if pattern.startswith('#'):
            pattern = pattern[1:]  # Remove the # prefix
            print(f"DEBUG: Pattern after removing #: {pattern}")
            
        if not pattern.startswith('/'):
            # Relative path, use current directory
            old_pattern = pattern
            pattern = os.path.join(os.getcwd(), pattern)
            print(f"DEBUG: Pattern converted from {old_pattern} to {pattern}")
            
        dirname = os.path.dirname(pattern)
        print(f"DEBUG: Directory name: {dirname}")
        if not os.path.exists(dirname):
            print(f"DEBUG: Warning: directory {dirname} does not exist")
            print(f"DEBUG: Current directory: {os.getcwd()}")
            print(f"DEBUG: Parent directory contents: {os.listdir(os.path.dirname(dirname))}")
            return []
            
        print(f"DEBUG: Directory contents: {os.listdir(dirname)}")
        result = glob.glob(pattern)
        print(f"DEBUG: Raw glob result: {result}")
        
        # Convert absolute paths to relative paths
        cwd = os.getcwd()
        print(f"DEBUG: Current working directory: {cwd}")
        result = [os.path.relpath(p, cwd) for p in result]
        print(f"DEBUG: Relative path result: {result}")
        
        if not result:
            print(f"DEBUG: Warning: no files found matching pattern {pattern}")
            return []  # Return empty list instead of None
            
        return result
        
    def Dir(self, path: str) -> Any:
        """Return a directory node"""
        print(f"DEBUG: Dir called with path={path}")
        class DirNode:
            def __init__(self, path):
                print(f"DEBUG: Creating DirNode with path={path}")
                self.path = path
                if path.startswith('#'):
                    self.path = path[1:]  # Remove the # prefix
                    print(f"DEBUG: Path after removing #: {self.path}")
            @property
            def tpath(self):
                print(f"DEBUG: Getting tpath: {self.path}")
                return self.path
            def __str__(self):
                return f"DirNode({self.path})"
            def __repr__(self):
                return self.__str__()
        result = DirNode(path)
        print(f"DEBUG: Dir result: {result}")
        return result
        
    def Object(self, path: str) -> Any:
        """Return an object node"""
        print(f"DEBUG: Object called with path={path}")
        class ObjectNode:
            def __init__(self, path):
                print(f"DEBUG: Creating ObjectNode with path={path}")
                self.path = path
                if path.startswith('#'):
                    self.path = path[1:]  # Remove the # prefix
                    print(f"DEBUG: Path after removing #: {self.path}")
            def __eq__(self, other):
                print(f"DEBUG: Comparing {self} with {other}")
                if isinstance(other, ObjectNode):
                    result = self.path == other.path
                    print(f"DEBUG: Comparison result: {result}")
                    return result
                print("DEBUG: Other object is not an ObjectNode")
                return False
            def __str__(self):
                return f"ObjectNode({self.path})"
            def __repr__(self):
                return self.__str__()
            def __hash__(self):
                result = hash(self.path)
                print(f"DEBUG: Hash for {self}: {result}")
                return result
        result = ObjectNode(path)
        print(f"DEBUG: Object result: {result}")
        return result
        
    def add_source_files(self, target: str, files: List[str], allow_gen: bool = True) -> None:
        """Add source files to a target"""
        print(f"DEBUG: add_source_files called with target={target}, files={files}, allow_gen={allow_gen}")
        print(f"DEBUG: Current source_files: {self.source_files}")
        print(f"DEBUG: Type of files: {type(files)}")
        
        if target not in self.source_files:
            print(f"DEBUG: Creating new list for target {target}")
            self.source_files[target] = []
            
        if isinstance(files, str):
            print(f"DEBUG: Converting string to list: {files}")
            files = [files]
            
        if files is not None:
            print(f"DEBUG: Extending source files: {files}")
            try:
                self.source_files[target].extend(files)
                print(f"DEBUG: Source files after extend: {self.source_files[target]}")
            except Exception as e:
                print(f"DEBUG: Error extending source files: {e}")
        else:
            print("DEBUG: Warning: files is None")
            
        print(f"DEBUG: Final source_files: {self.source_files}")
        
    def CommandNoCache(self, target: Union[str, List[str]], source: Union[str, List[str]], action: Union[str, Callable]) -> Any:
        """Run a command without caching"""
        if isinstance(action, str):
            def action_func(target, source, env):
                os.system(action)
                return 0
            action = action_func
        return action(target, source, self)
        
    def Add(self, variable: Union[str, Variable], help_text: str = '', default: Any = None, **kwargs) -> None:
        """Add a construction variable"""
        if isinstance(variable, str):
            # Simple string variable
            self.variables[variable] = default
        elif isinstance(variable, tuple):
            # Variable with name and help text
            name = variable[0]
            if isinstance(name, (list, tuple)):
                name = name[0]  # Use first name in list
            self.variables[name] = default
        else:
            # Variable object
            print(f"Adding variable: args={variable}, kwargs={kwargs}")
            if hasattr(variable, 'key'):
                self.variables[variable.key] = variable.default