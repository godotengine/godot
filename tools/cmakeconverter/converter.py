"""Convert Godot's SCons build system to CMake"""
import os
import sys
from typing import Any, Dict, List, Optional

from .interpreter import SConsInterpreter, SConsError
from .cmake_generator import GodotCMakeGenerator

class BuildSystemConverter:
    """Converts SCons build system to CMake"""
    def __init__(self):
        self.interpreter = SConsInterpreter()
        self.cmake_generator = GodotCMakeGenerator()
        self.scons_vars: Dict[str, Any] = {}
        self.scons_env: Dict[str, Any] = {}
        self.platform = ""
        self.modules: List[str] = []

    def process_scons(self, sconstruct_path: str):
        """Process the SCons build system"""
        try:
            # Create interpreter with project root
            project_root = os.path.dirname(os.path.abspath(sconstruct_path))
            converter_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Add project root and converter root to Python path
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            if converter_root not in sys.path:
                sys.path.insert(0, converter_root)
                
            self.interpreter = SConsInterpreter(project_root=project_root)

            # Parse SCons configuration
            self.interpreter.interpret_file(sconstruct_path)

            # Extract variables
            self.scons_vars = self.interpreter.variables
            
            # Get default environment
            env = self.interpreter.default_env
            self.scons_env = env.variables.copy()
            self.scons_env.update({
                'CCFLAGS': env.get('CCFLAGS', []),
                'CXXFLAGS': env.get('CXXFLAGS', []),
                'LINKFLAGS': env.get('LINKFLAGS', []),
                'CPPPATH': env.get('CPPPATH', []),
                'CPPDEFINES': env.get('CPPDEFINES', []),
                'LIBS': env.get('LIBS', []),
                'LIBPATH': env.get('LIBPATH', []),
            })

            # Get platform
            self.platform = self.scons_vars.get('platform', 'linuxbsd')

            # Get modules
            self.modules = self._detect_modules()

        except SConsError as e:
            print(f"Error processing SCons: {e}")
            raise

    def _detect_modules(self) -> List[str]:
        """Detect available and enabled Godot modules"""
        modules = []
        
        # First check enabled modules from variables
        for key, value in self.scons_vars.items():
            if key.startswith('module_') and key.endswith('_enabled'):
                if value:
                    module = key[7:-8]  # Remove 'module_' prefix and '_enabled' suffix
                    modules.append(module)
        
        # Then check for modules with SCsub files
        if not modules:  # Only if no modules were found in variables
            modules_path = os.path.join(os.path.dirname(self.interpreter.current_script_dir), "modules")
            if os.path.exists(modules_path):
                for item in os.listdir(modules_path):
                    if os.path.isdir(os.path.join(modules_path, item)):
                        # Check if it's a valid module (has a SCons file)
                        if os.path.exists(os.path.join(modules_path, item, "SCsub")):
                            modules.append(item)
        
        return modules

    def generate_cmake(self, output_path: str):
        """Generate CMake project"""
        try:
            # Process variables
            self.cmake_generator.process_variables(self.scons_vars)

            # Process environment
            self.cmake_generator.process_environment(self.scons_env)

            # Process modules
            self.cmake_generator.process_modules(self.modules)

            # Process platform
            if self.platform:
                self.cmake_generator.process_platform(self.platform)

            # Process builders
            for builder in self.interpreter.default_env.builders:
                if builder['type'] == 'program':
                    target = self.cmake_generator.project.add_target(
                        name=builder['target'][0],
                        target_type='executable'
                    )
                    target.add_sources(builder['source'])
                    if 'include_dirs' in builder:
                        target.add_include_dirs(builder['include_dirs'])
                    if 'compile_flags' in builder:
                        target.add_compile_options(builder['compile_flags'])
                    if 'link_flags' in builder:
                        target.add_link_options(builder['link_flags'])
                    if 'libraries' in builder:
                        target.add_link_libraries(builder['libraries'])
                elif builder['type'] == 'library':
                    target = self.cmake_generator.project.add_target(
                        name=builder['target'][0],
                        target_type='shared' if builder.get('shared', False) else 'static'
                    )
                    target.add_sources(builder['source'])
                    if 'include_dirs' in builder:
                        target.add_include_dirs(builder['include_dirs'])
                    if 'compile_flags' in builder:
                        target.add_compile_options(builder['compile_flags'])
                    if 'link_flags' in builder:
                        target.add_link_options(builder['link_flags'])
                    if 'libraries' in builder:
                        target.add_link_libraries(builder['libraries'])

            # Generate CMake files
            self.cmake_generator.generate(output_path)

        except Exception as e:
            print(f"Error generating CMake: {e}")
            raise

def convert_build_system(sconstruct_path: str, output_path: str):
    """Convert SCons build system to CMake"""
    converter = BuildSystemConverter()
    
    print("Processing SCons build system...")
    converter.process_scons(sconstruct_path)
    
    print("Generating CMake project...")
    converter.generate_cmake(output_path)
    
    print("Conversion complete!")