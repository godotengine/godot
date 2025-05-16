from typing import TYPE_CHECKING, Any, Dict, List, OrderedDict, Set

from typing_extensions import Annotated

from methods import (
    CommandNoCache,
    Run,
    add_library,
    add_module_version_string,
    add_program,
    add_shared_library,
    add_source_files,
    disable_warnings,
    force_optimization_on_debug,
    module_add_dependencies,
    module_check_dependencies,
    use_windows_spawn_fix,
)

if TYPE_CHECKING:
    from SCons.Script.SConscript import SConsEnvironment
from abc import ABC

PositiveInt = Annotated[int, lambda x: x > 0]
NonNegativeInt = Annotated[int, lambda x: x >= 0]


class GodotSConsEnvironment(SConsEnvironment, ABC):
    """
    This class exists to allow typechecking for SConsEnvironment augmented with additional
    methods and properties in the SConstruct file.

    Every method and property in this class should be abstract as they are never run
    and only exist for the 🍶 of typechecking.
    """

    @property
    def disabled_modules(self) -> Set[Any]:
        # defined in SConstruct file
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return set()

    @property
    def module_version_string(self) -> str:
        # defined in SConstruct file
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return ""

    @module_version_string.setter
    def module_version_string(self, value: str) -> None:
        # allows env.module_version_string = x without IDE warning
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")

    @property
    def msvc(self) -> bool:
        # defined in SConstruct file
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return False

    def Object(self, path_arg: Any) -> Any:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return path_arg

    @property
    def module_dependencies(self) -> Dict[Any, List[List[Any]]]:
        # defined in SConstruct file
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return {}

    @property
    def module_list(self) -> OrderedDict[Any, Any]:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return OrderedDict()

    def add_module_version_string(self, s: str) -> None:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        add_module_version_string(self, s)

    def add_source_files_orig(self, sources, files, allow_gen: bool = False) -> None:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        add_source_files(self, sources, files, allow_gen=allow_gen)

    def use_windows_spawn_fix(self, platform=None) -> Any:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        use_windows_spawn_fix(self, platform)

    def add_shared_library(self, name: str, sources, **args):
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        add_shared_library(self, name, sources, **args)

    def add_library(self, name, sources, **args):
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        add_library(self, name, sources, **args)

    def add_program(self, name, sources, **args):
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        add_program(self, name, sources, **args)

    def CommandNoCache(self, target, sources, command, **args):
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        CommandNoCache(self, target, sources, command, **args)

    def Run(self, function):
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        Run(self, function)

    def disable_warnings(self) -> None:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        disable_warnings(self)

    def force_optimization_on_debug(self) -> None:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        force_optimization_on_debug(self)

    def module_add_dependencies(self, module, dependencies, optional: bool = False) -> None:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        module_add_dependencies(self, module, dependencies, optional=optional)

    def module_check_dependencies(self, module) -> bool:
        if not TYPE_CHECKING:
            raise NotImplementedError("This should never be run☠.")
        return module_check_dependencies(self, module)
