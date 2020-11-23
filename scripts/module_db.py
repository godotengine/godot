#!/usr/bin/python3

import utils

import argparse
import runpy
import glob
import os
import sys
import json
import functools
from typing import Dict, Set
from types import SimpleNamespace
from collections import defaultdict


class Environment:
    def __init__(self):
        self.platform: str = ''
        self.cpu_family: str = ''
        self.tools_enabled: bool = False


class Module:
    def __init__(self, name: str = '', path: str = '', build: bool = True):
        self.name = name
        self.path = path
        self.build = build
        self.module_dependencies: dict = {}
        self.doc_path: str = ''
        self.doc_classes: [str] = []
        self.build_info: [str] = []

    def _encode(self) -> dict():
        return self.__dict__

    def _decode(self, data: dict):
        for key, value in data.items():
            self.__setattr__(key, value)


# uses the graph/topo sort from:
# https://www.geeksforgeeks.org/python-program-for-topological-sorting/
# albeit highly modified
class ModuleDepGraph:
    def __init__(self, modules: [Module]):
        self.graph = dict()
        self.vertices = [m.name for m in modules]

        # construct the edges
        for m in modules:
            self.graph[m.name] = []
            for dep_name in m.module_dependencies.keys():
                self.graph[m.name].append(dep_name)

    # A recursive function used by dependency_sort
    def topological_sort_util(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if i in visited and visited[i] == False:
                self.topological_sort_util(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to performs a topological sort, and then reverses it to obtain the dependency sort.
    def dependency_sort(self) -> []:
        # Mark all the vertices as not visited
        visited = dict()
        for v in self.vertices:
            visited[v] = False

        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for v in self.vertices:
            if visited[v] == False:
                self.topological_sort_util(v, visited, stack)

        # reverse the topological sort
        stack.reverse()
        return stack


class ModuleDb:
    def __init__(self):
        self.modules: Dict[str, Module] = dict()
        self.modules_enabled: [Module] = []
        self.modules_disabled: [Module] = []
        return

    def _encode(self) -> dict():
        d = dict()
        d['modules_enabled'] = self.get_modules_enabled_names()
        d['modules_disabled'] = self.get_modules_disabled_names()
        d['modules'] = dict()
        for module in self.get_modules():
            d['modules'][module.name] = module._encode()

        return d

    def _decode(self, data: dict):
        for module_dict in data['modules'].values():
            module = Module()
            module._decode(module_dict)
            self.modules[module.name] = module

        for mname in data['modules_enabled']:
            self.modules_enabled.append(self.get_module(mname))

        for mname in data['modules_disabled']:
            self.modules_disabled.append(self.get_module(mname))
        return

    def get_modules(self) -> iter:
        return self.modules.values()

    def get_modules_dependency_sorted(self) -> iter:
        graph = ModuleDepGraph(self.modules.values())
        dep_sorted_names = graph.dependency_sort()

        dep_modules = [self.modules[x] for x in dep_sorted_names]
        return dep_modules

    def get_module(self, module_name: str) -> Module:
        return self.modules[module_name]

    def has_module(self, module_name: str) -> bool:
        return module_name in self.modules

    def get_doc_paths(self) -> [str]:
        dpset = set()
        for module in self.modules.values():
            if not module.doc_path or len(module.doc_classes) == 0:
                continue
            dpset.add(os.path.join(module.path, module.doc_path))

        dplist: [str] = list(dpset)
        dplist.sort()

        return dplist

    def get_modules_enabled_names(self) -> [str]:
        em: [str] = [module.name for module in self.get_modules()
                     if module.build]
        em.sort()
        return em

    def get_modules_disabled_names(self) -> [str]:
        dm: [str] = [module.name for module in self.get_modules()
                     if not module.build]
        dm.sort()
        return dm

    def get_module_build_paths(self) -> [str]:
        bp: [str] = [
            module.path for module in self.get_modules_dependency_sorted() if module.build]
        return bp


def load_db(module_db_file: str) -> ModuleDb:
    if not os.path.isfile(module_db_file):
        return None

    mdb: ModuleDb = ModuleDb()
    with open(module_db_file, 'r') as f:
        data = json.load(f)
        mdb._decode(data)

    return mdb


def write_db(out_file: str, mdb: ModuleDb):
    with open(out_file, 'w') as f:
        json.dump(mdb._encode(), f, indent='\t')
    return

################################################################################
# PRIVATE FUNCTIONS FOR THE TOOL
################################################################################

# Load the config.py for the module and obtain the information we seek for the module


def __parse_module_config(config_path: str, env: Environment) -> Module:
    module_path = os.path.dirname(config_path)
    module_name = module_path.split(os.sep)[-1]

    module: Module = Module(name=module_name, path=module_path, build=True)

    config = runpy.run_path(config_path)

    if 'can_build' in config:
        can_build = config['can_build'](env)
        module.build &= can_build
        if not can_build:
            module.build_info.append('The can_build() check failed.')

    if 'module_dependencies' in config:
        module.module_dependencies = config['module_dependencies']()

    if 'get_doc_path' in config:
        module.doc_path = config['get_doc_path']()

    if 'get_doc_classes' in config:
        module.doc_classes = config['get_doc_classes']()

    build_file: str = os.path.join(module_path, 'meson.build')
    if not os.path.isfile(build_file):
        module.build = False
        module.build_info.append(
            'The meson.build file for this module is missing.')

    return module


# This function verifies all module dependencies (after they have been added to the db)
# and marks the 'build' var if the dependencies are not met.
def __check_module_dependencies(mdb: ModuleDb):
    deps_checked: Set[str] = set()

    # Function used recursively to figure out if a module has all of its dependencies
    def check_deps(module: Module) -> bool:
        # Another module might have checked us
        if module.name in deps_checked:
            return module.build

        # We are visiting it now
        deps_checked.add(module.name)

        # Go through our dependencies
        for dep_name, dep_options in module.module_dependencies.items():
            required = dep_options['required'] if 'required' in dep_options else True

            # If a module with the name doesn't exist, we have a problem
            if not mdb.has_module(dep_name):
                module.build = False
                module.build_info.append(
                    'Dependent module ' + dep_name + ' not found.')
                return False

            # Check if the module we refer to has its dependencies met
            elif not check_deps(mdb.get_module(dep_name)) and required:
                module.build = False
                module.build_info.append(
                    'Cannot build module due do dependency \"' + dep_name + '\" not building.')

        return module.build

    for module in mdb.get_modules():
        check_deps(module)


# Create the module database file
def __create_db_file(args):
    output = args.output
    module_search_paths: [str] = args.module_search_path
    modules_disabled: [str] = args.module_disabled

    configs = []
    for msp in module_search_paths:
        configs += glob.glob(os.path.join(msp, '**', 'config.py'))

    mdb: ModuleDb = ModuleDb()

    for config in configs:
        # make a new env for every config, incase someone mutates it
        env: Environment = Environment()
        env.platform = args.platform
        env.cpu_family = args.cpu_family
        env.tools_enabled = args.tools_enabled

        module: Module = __parse_module_config(config, env)
        mdb.modules[module.name] = module

    for module in mdb.get_modules():
        if module.name in modules_disabled:
            module.build = False
            module.build_info.append(
                'Module was disabled from the command line.')

    __check_module_dependencies(mdb)

    # BRUTE FORCE DISABLE ALL
    if args.disable_all:
        for module in mdb.get_modules():
            module.build = False
            module.build_info.append('Disabled due to disable_all argument')

    for module in mdb.get_modules():
        if module.build:
            mdb.modules_enabled.append(module.name)
        else:
            mdb.modules_disabled.append(module.name)

    write_db(output, mdb)


def __print_modules_enabled(module_db_file: str):
    mdb: ModuleDb = load_db(module_db_file)
    print(','.join(mdb.get_modules_enabled_names()))


def __print_modules_disabled(module_db_file: str):
    mdb: ModuleDb = load_db(module_db_file)
    print(','.join(mdb.get_modules_disabled_names()))


def __print_module_build_paths(module_db_file: str):
    mdb: ModuleDb = load_db(module_db_file)
    print(','.join(mdb.get_module_build_paths()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modules tools.')

    subparsers = parser.add_subparsers(help='sub-command help', dest='command')

    # create the database creator parser
    create_db_parser = subparsers.add_parser(
        'create_db', help='create_db help')
    create_db_parser.add_argument(
        'output', type=str, help='The output modules db.')
    create_db_parser.add_argument(
        '--platform', type=str, required=True, help='The current platform being built.'
    )
    create_db_parser.add_argument(
        '--cpu_family', type=str, required=True, help='The current cpu_family being built.'
    )
    create_db_parser.add_argument(
        '--tools_enabled', dest='tools_enabled', action='store_true', help='Whether or not tooling is enabled for this build.'
    )
    create_db_parser.add_argument(
        '--module_search_path', type=str, nargs='+', help="The module search paths to check for config.py's\n"
        "This can be specified multiple times for each path, but must be specified at least once."
    )
    create_db_parser.add_argument(
        '--module_disabled', type=str, nargs='*', action='append', help="The modules to be disabled for building.\n"
        "This can be specified multiple times for each module to be disabled."
    )
    create_db_parser.add_argument(
        '--disable_all_modules', dest='disable_all', action='store_true', help="Override to disable all modules."
    )
    create_db_parser.set_defaults(disable_all=False)

    # modules enabled printer
    modules_enabled_parser = subparsers.add_parser(
        'get_enabled_modules', help='get_enabled_modules help')
    modules_enabled_parser.add_argument(
        'database_file', type=str, help='The input modules_db file')

    # modules disabled printer
    modules_disabled_parser = subparsers.add_parser(
        'get_disabled_modules', help='get_disabled_modules help')
    modules_disabled_parser.add_argument(
        'database_file', type=str, help='The input modules_db file')

    # module buld_files printer
    modules_bp_parser = subparsers.add_parser(
        'get_module_build_paths', help='get_module_build_paths help')
    modules_bp_parser.add_argument(
        'database_file', type=str, help='The input modules_db file')

    args = parser.parse_args()

    # Go through each module and check its configuration
    if args.command == 'create_db':
        args.module_disabled = [x[0] for x in args.module_disabled]

        __create_db_file(args)

    # Get the list of enabled modules
    elif args.command == 'get_enabled_modules':
        __print_modules_enabled(args.database_file)

    # Get this list of disabled modules
    elif args.command == 'get_disabled_modules':
        __print_modules_disabled(args.database_file)

    elif args.command == 'get_module_build_paths':
        __print_module_build_paths(args.database_file)
