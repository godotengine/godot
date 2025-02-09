"""CMake project generator for Godot"""
from typing import Any, Dict, List, Optional, Set
import os
import textwrap

class CMakeTarget:
    """Represents a CMake target (executable, library, etc.)"""
    def __init__(self, name: str, target_type: str):
        self.name = name
        self.type = target_type  # executable, library, etc.
        self.sources: List[str] = []
        self.include_dirs: Set[str] = set()
        self.compile_definitions: Set[str] = set()
        self.compile_options: Set[str] = set()
        self.link_libraries: Set[str] = set()
        self.link_options: Set[str] = set()
        self.dependencies: Set[str] = set()

    def add_sources(self, sources: List[str]):
        """Add source files to the target"""
        for src in sources:
            if not os.path.isabs(src):
                src = "${CMAKE_SOURCE_DIR}/" + src
            self.sources.append(src)

    def add_include_dirs(self, dirs: List[str]):
        """Add include directories"""
        self.include_dirs.update(dirs)

    def add_definitions(self, defs: List[str]):
        """Add compile definitions"""
        self.compile_definitions.update(defs)

    def add_compile_options(self, options: List[str]):
        """Add compile options"""
        self.compile_options.update(options)

    def add_link_libraries(self, libs: List[str]):
        """Add libraries to link against"""
        self.link_libraries.update(libs)

    def add_link_options(self, options: List[str]):
        """Add linker options"""
        self.link_options.update(options)

    def add_dependencies(self, deps: List[str]):
        """Add target dependencies"""
        self.dependencies.update(deps)

    def generate(self, indent: int = 0) -> str:
        """Generate CMake code for this target"""
        ind = " " * indent
        lines = []

        # Target declaration
        if self.type == "executable":
            lines.append(f"{ind}add_executable({self.name}")
        elif self.type == "shared":
            lines.append(f"{ind}add_library({self.name} SHARED")
        elif self.type == "static":
            lines.append(f"{ind}add_library({self.name} STATIC")
        else:
            lines.append(f"{ind}add_library({self.name} {self.type}")

        # Sources
        if self.sources:
            lines.append(f"{ind}    {' '.join(sorted(self.sources))}")
        lines.append(f"{ind})")

        # Include directories
        if self.include_dirs:
            lines.append(f"{ind}target_include_directories({self.name} PUBLIC")
            for inc in sorted(self.include_dirs):
                lines.append(f"{ind}    {inc}")
            lines.append(f"{ind})")

        # Compile definitions
        if self.compile_definitions:
            lines.append(f"{ind}target_compile_definitions({self.name} PUBLIC")
            for define in sorted(self.compile_definitions):
                lines.append(f"{ind}    {define}")
            lines.append(f"{ind})")

        # Compile options
        if self.compile_options:
            lines.append(f"{ind}target_compile_options({self.name} PUBLIC")
            for opt in sorted(self.compile_options):
                lines.append(f"{ind}    {opt}")
            lines.append(f"{ind})")

        # Link libraries
        if self.link_libraries:
            lines.append(f"{ind}target_link_libraries({self.name} PUBLIC")
            for lib in sorted(self.link_libraries):
                lines.append(f"{ind}    {lib}")
            lines.append(f"{ind})")

        # Link options
        if self.link_options:
            lines.append(f"{ind}target_link_options({self.name} PUBLIC")
            for opt in sorted(self.link_options):
                lines.append(f"{ind}    {opt}")
            lines.append(f"{ind})")

        # Dependencies
        if self.dependencies:
            lines.append(f"{ind}add_dependencies({self.name}")
            for dep in sorted(self.dependencies):
                lines.append(f"{ind}    {dep}")
            lines.append(f"{ind})")

        return "\n".join(lines)

class CMakeProject:
    """Represents a CMake project"""
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.minimum_version = "3.16"
        self.targets: Dict[str, CMakeTarget] = {}
        self.variables: Dict[str, Any] = {}
        self.options: Dict[str, Any] = {}
        self.custom_commands: List[str] = []
        self.custom_targets: List[str] = []
        self.includes: List[str] = []

    def add_target(self, name: str, target_type: str) -> CMakeTarget:
        """Add a new target to the project"""
        target = CMakeTarget(name, target_type)
        self.targets[name] = target
        return target

    def set_variable(self, name: str, value: Any):
        """Set a CMake variable"""
        self.variables[name] = value

    def add_option(self, name: str, description: str, default_value: Any):
        """Add a CMake option"""
        self.options[name] = {
            "description": description,
            "default": default_value
        }

    def add_custom_command(self, command: str):
        """Add a custom CMake command"""
        self.custom_commands.append(command)

    def add_custom_target(self, target: str):
        """Add a custom CMake target"""
        self.custom_targets.append(target)

    def add_include(self, path: str):
        """Add a CMake include directive"""
        self.includes.append(path)

    def generate(self) -> str:
        """Generate the complete CMake project"""
        lines = []

        # Header
        lines.append(f"cmake_minimum_required(VERSION {self.minimum_version})")
        lines.append(f"project({self.name} VERSION {self.version})")
        lines.append("")

        # Options
        if self.options:
            lines.append("# Options")
            for name, opt in self.options.items():
                lines.append(f'option({name} "{opt["description"]}" {opt["default"]})')
            lines.append("")

        # Variables
        if self.variables:
            lines.append("# Variables")
            for name, value in self.variables.items():
                if isinstance(value, (list, tuple, set)):
                    lines.append(f"set({name}")
                    for item in value:
                        lines.append(f"    {item}")
                    lines.append(")")
                else:
                    lines.append(f"set({name} {value})")
            lines.append("")

        # Includes
        if self.includes:
            lines.append("# Includes")
            for inc in self.includes:
                lines.append(f"include({inc})")
            lines.append("")

        # Custom commands
        if self.custom_commands:
            lines.append("# Custom commands")
            for cmd in self.custom_commands:
                lines.append(cmd)
            lines.append("")

        # Custom targets
        if self.custom_targets:
            lines.append("# Custom targets")
            for target in self.custom_targets:
                lines.append(target)
            lines.append("")

        # Targets
        if self.targets:
            lines.append("# Targets")
            for target in self.targets.values():
                lines.append(target.generate(indent=0))
                lines.append("")

        return "\n".join(lines)

class GodotCMakeGenerator:
    """Generates CMake project from Godot's SCons configuration"""
    def __init__(self):
        self.project = CMakeProject("godot")
        self.platform_defines: Dict[str, List[str]] = {}
        self.platform_flags: Dict[str, List[str]] = {}
        self.enabled_modules: Set[str] = set()

    def process_variables(self, variables: Dict[str, Any]):
        """Process SCons variables into CMake options/variables"""
        for name, value in variables.items():
            if isinstance(value, bool):
                # Convert boolean SCons variables to CMake options
                self.project.add_option(
                    name=name.upper(),
                    description=f"Enable {name}",
                    default_value="ON" if value else "OFF"
                )
            elif isinstance(value, (str, int, float)):
                # Convert scalar SCons variables to CMake variables
                self.project.set_variable(name.upper(), value)
            elif isinstance(value, (list, tuple)):
                # Convert list SCons variables to CMake list variables
                self.project.set_variable(name.upper(), ";".join(str(x) for x in value))

    def process_environment(self, env: Dict[str, Any]):
        """Process SCons environment into CMake configuration"""
        # Process compiler flags
        if "CCFLAGS" in env:
            self.project.set_variable("CMAKE_C_FLAGS", " ".join(env["CCFLAGS"]))
            self.project.set_variable("CMAKE_CXX_FLAGS", " ".join(env["CCFLAGS"]))

        if "CXXFLAGS" in env:
            self.project.set_variable("CMAKE_CXX_FLAGS", " ".join(env["CXXFLAGS"]))

        if "LINKFLAGS" in env:
            self.project.set_variable("CMAKE_EXE_LINKER_FLAGS", " ".join(env["LINKFLAGS"]))
            self.project.set_variable("CMAKE_SHARED_LINKER_FLAGS", " ".join(env["LINKFLAGS"]))

        # Process include paths
        if "CPPPATH" in env:
            includes = [f"${{CMAKE_SOURCE_DIR}}/{path}" for path in env["CPPPATH"]]
            self.project.set_variable("GODOT_INCLUDE_DIRS", includes)

        # Process defines
        if "CPPDEFINES" in env:
            defines = []
            for define in env["CPPDEFINES"]:
                if isinstance(define, tuple):
                    defines.append(f"{define[0]}={define[1]}")
                else:
                    defines.append(str(define))
            self.project.set_variable("GODOT_DEFINES", defines)

    def process_modules(self, modules: List[str]):
        """Process Godot modules into CMake targets"""
        for module in modules:
            target = self.project.add_target(f"godot_{module}", "static")
            target.add_compile_definitions([f"MODULE_{module.upper()}_ENABLED"])
            self.enabled_modules.add(module)

    def process_platform(self, platform: str):
        """Process platform-specific configuration"""
        if platform in self.platform_defines:
            self.project.set_variable(
                f"GODOT_{platform.upper()}_DEFINES",
                self.platform_defines[platform]
            )

        if platform in self.platform_flags:
            self.project.set_variable(
                f"GODOT_{platform.upper()}_FLAGS",
                self.platform_flags[platform]
            )

    def generate(self, output_path: str):
        """Generate the CMake project files"""
        # Main CMakeLists.txt
        cmake_content = self.project.generate()
        
        # Write the main CMakeLists.txt
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if path has a directory part
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(cmake_content)

        # Generate module CMakeLists.txt files
        for module in self.enabled_modules:
            module_path = os.path.join(os.path.dirname(output_path), "modules", module, "CMakeLists.txt")
            module_content = self._generate_module_cmake(module)
            os.makedirs(os.path.dirname(module_path), exist_ok=True)
            with open(module_path, "w") as f:
                f.write(module_content)

    def _generate_module_cmake(self, module: str) -> str:
        """Generate CMakeLists.txt for a specific module"""
        lines = []
        lines.append(f"# CMakeLists.txt for module {module}")
        lines.append("")
        lines.append(f"target_sources(godot_{module}")
        lines.append("    PRIVATE")
        lines.append("        ${MODULE_SOURCES}")  # Placeholder
        lines.append(")")
        lines.append("")
        lines.append("target_include_directories(godot_{module}")
        lines.append("    PUBLIC")
        lines.append("        ${CMAKE_CURRENT_SOURCE_DIR}")
        lines.append(")")
        return "\n".join(lines)