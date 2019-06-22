#!/usr/bin/env python

# Copyright 2017 The Glslang Authors. All rights reserved.
# Copyright (c) 2018 Valve Corporation
# Copyright (c) 2018 LunarG, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script was heavily leveraged from KhronosGroup/glslang
# update_glslang_sources.py.
"""update_deps.py

Get and build dependent repositories using known-good commits.

Purpose
-------

This program is intended to assist a developer of this repository
(the "home" repository) by gathering and building the repositories that
this home repository depend on.  It also checks out each dependent
repository at a "known-good" commit in order to provide stability in
the dependent repositories.

Python Compatibility
--------------------

This program can be used with Python 2.7 and Python 3.

Known-Good JSON Database
------------------------

This program expects to find a file named "known-good.json" in the
same directory as the program file.  This JSON file is tailored for
the needs of the home repository by including its dependent repositories.

Program Options
---------------

See the help text (update_deps.py --help) for a complete list of options.

Program Operation
-----------------

The program uses the user's current directory at the time of program
invocation as the location for fetching and building the dependent
repositories.  The user can override this by using the "--dir" option.

For example, a directory named "build" in the repository's root directory
is a good place to put the dependent repositories because that directory
is not tracked by Git. (See the .gitignore file.)  The "external" directory
may also be a suitable location.
A user can issue:

$ cd My-Repo
$ mkdir build
$ cd build
$ ../scripts/update_deps.py

or, to do the same thing, but using the --dir option:

$ cd My-Repo
$ mkdir build
$ scripts/update_deps.py --dir=build

With these commands, the "build" directory is considered the "top"
directory where the program clones the dependent repositories.  The
JSON file configures the build and install working directories to be
within this "top" directory.

Note that the "dir" option can also specify an absolute path:

$ cd My-Repo
$ scripts/update_deps.py --dir=/tmp/deps

The "top" dir is then /tmp/deps (Linux filesystem example) and is
where this program will clone and build the dependent repositories.

Helper CMake Config File
------------------------

When the program finishes building the dependencies, it writes a file
named "helper.cmake" to the "top" directory that contains CMake commands
for setting CMake variables for locating the dependent repositories.
This helper file can be used to set up the CMake build files for this
"home" repository.

A complete sequence might look like:

$ git clone git@github.com:My-Group/My-Repo.git
$ cd My-Repo
$ mkdir build
$ cd build
$ ../scripts/update_deps.py
$ cmake -C helper.cmake ..
$ cmake --build .

JSON File Schema
----------------

There's no formal schema for the "known-good" JSON file, but here is
a description of its elements.  All elements are required except those
marked as optional.  Please see the "known_good.json" file for
examples of all of these elements.

- name

The name of the dependent repository.  This field can be referenced
by the "deps.repo_name" structure to record a dependency.

- url

Specifies the URL of the repository.
Example: https://github.com/KhronosGroup/Vulkan-Loader.git

- sub_dir

The directory where the program clones the repository, relative to
the "top" directory.

- build_dir

The directory used to build the repository, relative to the "top"
directory.

- install_dir

The directory used to store the installed build artifacts, relative
to the "top" directory.

- commit

The commit used to checkout the repository.  This can be a SHA-1
object name or a refname used with the remote name "origin".
For example, this field can be set to "origin/sdk-1.1.77" to
select the end of the sdk-1.1.77 branch.

- deps (optional)

An array of pairs consisting of a CMake variable name and a
repository name to specify a dependent repo and a "link" to
that repo's install artifacts.  For example:

"deps" : [
    {
        "var_name" : "VULKAN_HEADERS_INSTALL_DIR",
        "repo_name" : "Vulkan-Headers"
    }
]

which represents that this repository depends on the Vulkan-Headers
repository and uses the VULKAN_HEADERS_INSTALL_DIR CMake variable to
specify the location where it expects to find the Vulkan-Headers install
directory.
Note that the "repo_name" element must match the "name" element of some
other repository in the JSON file.

- prebuild (optional)
- prebuild_linux (optional)  (For Linux and MacOS)
- prebuild_windows (optional)

A list of commands to execute before building a dependent repository.
This is useful for repositories that require the execution of some
sort of "update" script or need to clone an auxillary repository like
googletest.

The commands listed in "prebuild" are executed first, and then the
commands for the specific platform are executed.

- custom_build (optional)

A list of commands to execute as a custom build instead of using
the built in CMake way of building. Requires "build_step" to be
set to "custom"

You can insert the following keywords into the commands listed in
"custom_build" if they require runtime information (like whether the
build config is "Debug" or "Release").

Keywords:
{0} reference to a dictionary of repos and their attributes
{1} reference to the command line arguments set before start
{2} reference to the CONFIG_MAP value of config.

Example:
{2} returns the CONFIG_MAP value of config e.g. debug -> Debug
{1}.config returns the config variable set when you ran update_dep.py
{0}[Vulkan-Headers][repo_root] returns the repo_root variable from
                                   the Vulkan-Headers GoodRepo object.

- cmake_options (optional)

A list of options to pass to CMake during the generation phase.

- ci_only (optional)

A list of environment variables where one must be set to "true"
(case-insensitive) in order for this repo to be fetched and built.
This list can be used to specify repos that should be built only in CI.
Typically, this list might contain "TRAVIS" and/or "APPVEYOR" because
each of these CI systems sets an environment variable with its own
name to "true".  Note that this could also be (ab)used to control
the processing of the repo with any environment variable.  The default
is an empty list, which means that the repo is always processed.

- build_step (optional)

Specifies if the dependent repository should be built or not. This can
have a value of 'build', 'custom',  or 'skip'. The dependent repositories are
built by default.

- build_platforms (optional)

A list of platforms the repository will be built on.
Legal options include:
"windows"
"linux"
"darwin"

Builds on all platforms by default.

Note
----

The "sub_dir", "build_dir", and "install_dir" elements are all relative
to the effective "top" directory.  Specifying absolute paths is not
supported.  However, the "top" directory specified with the "--dir"
option can be a relative or absolute path.

"""

from __future__ import print_function

import argparse
import json
import distutils.dir_util
import os.path
import subprocess
import sys
import platform
import multiprocessing
import shlex
import shutil

KNOWN_GOOD_FILE_NAME = 'known_good.json'

CONFIG_MAP = {
    'debug': 'Debug',
    'release': 'Release',
    'relwithdebinfo': 'RelWithDebInfo',
    'minsizerel': 'MinSizeRel'
}

VERBOSE = False

DEVNULL = open(os.devnull, 'wb')


def command_output(cmd, directory, fail_ok=False):
    """Runs a command in a directory and returns its standard output stream.

    Captures the standard error stream and prints it if error.

    Raises a RuntimeError if the command fails to launch or otherwise fails.
    """
    if VERBOSE:
        print('In {d}: {cmd}'.format(d=directory, cmd=cmd))
    p = subprocess.Popen(
        cmd, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    if p.returncode != 0:
        print('*** Error ***\nstderr contents:\n{}'.format(stderr))
        if not fail_ok:
            raise RuntimeError('Failed to run {} in {}'.format(cmd, directory))
    if VERBOSE:
        print(stdout)
    return stdout

class GoodRepo(object):
    """Represents a repository at a known-good commit."""

    def __init__(self, json, args):
        """Initializes this good repo object.

        Args:
        'json':  A fully populated JSON object describing the repo.
        'args':  Results from ArgumentParser
        """
        self._json = json
        self._args = args
        # Required JSON elements
        self.name = json['name']
        self.url = json['url']
        self.sub_dir = json['sub_dir']
        self.commit = json['commit']
        # Optional JSON elements
        self.build_dir = None
        self.install_dir = None
        if json.get('build_dir'):
            self.build_dir = os.path.normpath(json['build_dir'])
        if json.get('install_dir'):
            self.install_dir = os.path.normpath(json['install_dir'])
        self.deps = json['deps'] if ('deps' in json) else []
        self.prebuild = json['prebuild'] if ('prebuild' in json) else []
        self.prebuild_linux = json['prebuild_linux'] if (
            'prebuild_linux' in json) else []
        self.prebuild_windows = json['prebuild_windows'] if (
            'prebuild_windows' in json) else []
        self.custom_build = json['custom_build'] if ('custom_build' in json) else []
        self.cmake_options = json['cmake_options'] if (
            'cmake_options' in json) else []
        self.ci_only = json['ci_only'] if ('ci_only' in json) else []
        self.build_step = json['build_step'] if ('build_step' in json) else 'build'
        self.build_platforms = json['build_platforms'] if ('build_platforms' in json) else []
        # Absolute paths for a repo's directories
        dir_top = os.path.abspath(args.dir)
        self.repo_dir = os.path.join(dir_top, self.sub_dir)
        if self.build_dir:
            self.build_dir = os.path.join(dir_top, self.build_dir)
        if self.install_dir:
            self.install_dir = os.path.join(dir_top, self.install_dir)
	    # Check if platform is one to build on
        self.on_build_platform = False
        if self.build_platforms == [] or platform.system().lower() in self.build_platforms:
            self.on_build_platform = True

    def Clone(self):
        distutils.dir_util.mkpath(self.repo_dir)
        command_output(['git', 'clone', self.url, '.'], self.repo_dir)

    def Fetch(self):
        command_output(['git', 'fetch', 'origin'], self.repo_dir)

    def Checkout(self):
        print('Checking out {n} in {d}'.format(n=self.name, d=self.repo_dir))
        if self._args.do_clean_repo:
            shutil.rmtree(self.repo_dir, ignore_errors=True)
        if not os.path.exists(os.path.join(self.repo_dir, '.git')):
            self.Clone()
        self.Fetch()
        if len(self._args.ref):
            command_output(['git', 'checkout', self._args.ref], self.repo_dir)
        else:
            command_output(['git', 'checkout', self.commit], self.repo_dir)
        print(command_output(['git', 'status'], self.repo_dir))

    def CustomPreProcess(self, cmd_str, repo_dict):
        return cmd_str.format(repo_dict, self._args, CONFIG_MAP[self._args.config])

    def PreBuild(self):
        """Execute any prebuild steps from the repo root"""
        for p in self.prebuild:
            command_output(shlex.split(p), self.repo_dir)
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            for p in self.prebuild_linux:
                command_output(shlex.split(p), self.repo_dir)
        if platform.system() == 'Windows':
            for p in self.prebuild_windows:
                command_output(shlex.split(p), self.repo_dir)

    def CustomBuild(self, repo_dict):
        """Execute any custom_build steps from the repo root"""
        for p in self.custom_build:
            cmd = self.CustomPreProcess(p, repo_dict)
            command_output(shlex.split(cmd), self.repo_dir)

    def CMakeConfig(self, repos):
        """Build CMake command for the configuration phase and execute it"""
        if self._args.do_clean_build:
            shutil.rmtree(self.build_dir)
        if self._args.do_clean_install:
            shutil.rmtree(self.install_dir)

        # Create and change to build directory
        distutils.dir_util.mkpath(self.build_dir)
        os.chdir(self.build_dir)

        cmake_cmd = [
            'cmake', self.repo_dir,
            '-DCMAKE_INSTALL_PREFIX=' + self.install_dir
        ]

        # For each repo this repo depends on, generate a CMake variable
        # definitions for "...INSTALL_DIR" that points to that dependent
        # repo's install dir.
        for d in self.deps:
            dep_commit = [r for r in repos if r.name == d['repo_name']]
            if len(dep_commit):
                cmake_cmd.append('-D{var_name}={install_dir}'.format(
                    var_name=d['var_name'],
                    install_dir=dep_commit[0].install_dir))

        # Add any CMake options
        for option in self.cmake_options:
            cmake_cmd.append(option)

        # Set build config for single-configuration generators
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            cmake_cmd.append('-DCMAKE_BUILD_TYPE={config}'.format(
                config=CONFIG_MAP[self._args.config]))

        # Use the CMake -A option to select the platform architecture
        # without needing a Visual Studio generator.
        if platform.system() == 'Windows':
            if self._args.arch == '64' or self._args.arch == 'x64' or self._args.arch == 'win64':
                cmake_cmd.append('-A')
                cmake_cmd.append('x64')

        # Apply a generator, if one is specified.  This can be used to supply
        # a specific generator for the dependent repositories to match
        # that of the main repository.
        if self._args.generator is not None:
            cmake_cmd.extend(['-G', self._args.generator])

        if VERBOSE:
            print("CMake command: " + " ".join(cmake_cmd))

        ret_code = subprocess.call(cmake_cmd)
        if ret_code != 0:
            sys.exit(ret_code)

    def CMakeBuild(self):
        """Build CMake command for the build phase and execute it"""
        cmake_cmd = ['cmake', '--build', self.build_dir, '--target', 'install']
        if self._args.do_clean:
            cmake_cmd.append('--clean-first')

        if platform.system() == 'Windows':
            cmake_cmd.append('--config')
            cmake_cmd.append(CONFIG_MAP[self._args.config])

        # Speed up the build.
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            cmake_cmd.append('--')
            num_make_jobs = multiprocessing.cpu_count()
            env_make_jobs = os.environ.get('MAKE_JOBS', None)
            if env_make_jobs is not None:
                try:
                    num_make_jobs = min(num_make_jobs, int(env_make_jobs))
                except ValueError:
                    print('warning: environment variable MAKE_JOBS has non-numeric value "{}".  '
                          'Using {} (CPU count) instead.'.format(env_make_jobs, num_make_jobs))
            cmake_cmd.append('-j{}'.format(num_make_jobs))
        if platform.system() == 'Windows':
            cmake_cmd.append('--')
            cmake_cmd.append('/maxcpucount')

        if VERBOSE:
            print("CMake command: " + " ".join(cmake_cmd))

        ret_code = subprocess.call(cmake_cmd)
        if ret_code != 0:
            sys.exit(ret_code)

    def Build(self, repos, repo_dict):
        """Build the dependent repo"""
        print('Building {n} in {d}'.format(n=self.name, d=self.repo_dir))
        print('Build dir = {b}'.format(b=self.build_dir))
        print('Install dir = {i}\n'.format(i=self.install_dir))

        # Run any prebuild commands
        self.PreBuild()

        if self.build_step == 'custom':
            self.CustomBuild(repo_dict)
            return

        # Build and execute CMake command for creating build files
        self.CMakeConfig(repos)

        # Build and execute CMake command for the build
        self.CMakeBuild()


def GetGoodRepos(args):
    """Returns the latest list of GoodRepo objects.

    The known-good file is expected to be in the same
    directory as this script unless overridden by the 'known_good_dir'
    parameter.
    """
    if args.known_good_dir:
        known_good_file = os.path.join( os.path.abspath(args.known_good_dir),
            KNOWN_GOOD_FILE_NAME)
    else:
        known_good_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), KNOWN_GOOD_FILE_NAME)
    with open(known_good_file) as known_good:
        return [
            GoodRepo(repo, args)
            for repo in json.loads(known_good.read())['repos']
        ]


def GetInstallNames(args):
    """Returns the install names list.

    The known-good file is expected to be in the same
    directory as this script unless overridden by the 'known_good_dir'
    parameter.
    """
    if args.known_good_dir:
        known_good_file = os.path.join(os.path.abspath(args.known_good_dir),
            KNOWN_GOOD_FILE_NAME)
    else:
        known_good_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), KNOWN_GOOD_FILE_NAME)
    with open(known_good_file) as known_good:
        install_info = json.loads(known_good.read())
        if install_info.get('install_names'):
            return install_info['install_names']
        else:
            return None


def CreateHelper(args, repos, filename):
    """Create a CMake config helper file.

    The helper file is intended to be used with 'cmake -C <file>'
    to build this home repo using the dependencies built by this script.

    The install_names dictionary represents the CMake variables used by the
    home repo to locate the install dirs of the dependent repos.
    This information is baked into the CMake files of the home repo and so
    this dictionary is kept with the repo via the json file.
    """
    def escape(path):
        return path.replace('\\', '\\\\')
    install_names = GetInstallNames(args)
    with open(filename, 'w') as helper_file:
        for repo in repos:
            if install_names and repo.name in install_names and repo.on_build_platform:
                helper_file.write('set({var} "{dir}" CACHE STRING "" FORCE)\n'
                                  .format(
                                      var=install_names[repo.name],
                                      dir=escape(repo.install_dir)))


def main():
    parser = argparse.ArgumentParser(
        description='Get and build dependent repos at known-good commits')
    parser.add_argument(
        '--known_good_dir',
        dest='known_good_dir',
        help="Specify directory for known_good.json file.")
    parser.add_argument(
        '--dir',
        dest='dir',
        default='.',
        help="Set target directory for repository roots. Default is \'.\'.")
    parser.add_argument(
        '--ref',
        dest='ref',
        default='',
        help="Override 'commit' with git reference. E.g., 'origin/master'")
    parser.add_argument(
        '--no-build',
        dest='do_build',
        action='store_false',
        help=
        "Clone/update repositories and generate build files without performing compilation",
        default=True)
    parser.add_argument(
        '--clean',
        dest='do_clean',
        action='store_true',
        help="Clean files generated by compiler and linker before building",
        default=False)
    parser.add_argument(
        '--clean-repo',
        dest='do_clean_repo',
        action='store_true',
        help="Delete repository directory before building",
        default=False)
    parser.add_argument(
        '--clean-build',
        dest='do_clean_build',
        action='store_true',
        help="Delete build directory before building",
        default=False)
    parser.add_argument(
        '--clean-install',
        dest='do_clean_install',
        action='store_true',
        help="Delete install directory before building",
        default=False)
    parser.add_argument(
        '--arch',
        dest='arch',
        choices=['32', '64', 'x86', 'x64', 'win32', 'win64'],
        type=str.lower,
        help="Set build files architecture (Windows)",
        default='64')
    parser.add_argument(
        '--config',
        dest='config',
        choices=['debug', 'release', 'relwithdebinfo', 'minsizerel'],
        type=str.lower,
        help="Set build files configuration",
        default='debug')
    parser.add_argument(
        '--generator',
        dest='generator',
        help="Set the CMake generator",
        default=None)

    args = parser.parse_args()
    save_cwd = os.getcwd()

    # Create working "top" directory if needed
    distutils.dir_util.mkpath(args.dir)
    abs_top_dir = os.path.abspath(args.dir)

    repos = GetGoodRepos(args)
    repo_dict = {}

    print('Starting builds in {d}'.format(d=abs_top_dir))
    for repo in repos:
        # If the repo has a platform whitelist, skip the repo
        # unless we are building on a whitelisted platform.
        if not repo.on_build_platform:
            continue

        field_list = ('url',
                      'sub_dir',
                      'commit',
                      'build_dir',
                      'install_dir',
                      'deps',
                      'prebuild',
                      'prebuild_linux',
                      'prebuild_windows',
                      'custom_build',
                      'cmake_options',
                      'ci_only',
                      'build_step',
                      'build_platforms',
                      'repo_dir',
                      'on_build_platform')
        repo_dict[repo.name] = {field: getattr(repo, field) for field in field_list}

        # If the repo has a CI whitelist, skip the repo unless
        # one of the CI's environment variable is set to true.
        if len(repo.ci_only):
            do_build = False
            for env in repo.ci_only:
                if not env in os.environ:
                    continue
                if os.environ[env].lower() == 'true':
                    do_build = True
                    break
            if not do_build:
                continue

        # Clone/update the repository
        repo.Checkout()

        # Build the repository
        if args.do_build and repo.build_step != 'skip':
            repo.Build(repos, repo_dict)

    # Need to restore original cwd in order for CreateHelper to find json file
    os.chdir(save_cwd)
    CreateHelper(args, repos, os.path.join(abs_top_dir, 'helper.cmake'))

    sys.exit(0)


if __name__ == '__main__':
    main()
