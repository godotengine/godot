"""Convert Godot's SCons build system to CMake"""
import os
import sys
from typing import Any, Dict, List, Optional

from interpreter import SConsInterpreter, SConsError
from cmake_generator import GodotCMakeGenerator

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
            # Add project root to Python path
            project_root = os.path.dirname(os.path.abspath(sconstruct_path))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Parse SCons configuration
            self.interpreter.interpret_file(sconstruct_path)

            # Extract variables
            self.scons_vars = self.interpreter.variables
            
            # Get default environment
            env = self.interpreter.default_env
            if hasattr(env, 'Dictionary'):
                self.scons_env = env.Dictionary()
            else:
                self.scons_env = {}

            # Get platform
            self.platform = self.scons_env.get('platform', '')

            # Get modules
            self.modules = self._detect_modules()

        except SConsError as e:
            print(f"Error processing SCons: {e}")
            raise

    def _detect_modules(self) -> List[str]:
        """Detect available Godot modules"""
        modules = []
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