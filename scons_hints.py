"""
Adds intellisense hints for SCons scripts.
Implemented via `from scons_hints import *`.

This is NOT a 1-1 representation of what the defines will represent in an
SCons build, as proxies are almost always utilized instead. Rather, this is
a means of tracing back what those proxies are calling to in the first place.
"""

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from SCons.Action import Action
#     from SCons.Builder import Builder
#     from SCons.Defaults import Chmod, Copy, CScan, DefaultEnvironment, Delete, DirScanner, Mkdir, Move, Touch
#     from SCons.Platform import Platform
#     from SCons.Platform.virtualenv import Virtualenv
#     from SCons.Scanner import FindPathDirs, ScannerBase
#     from SCons.Script import ARGLIST, ARGUMENTS, BUILD_TARGETS, COMMAND_LINE_TARGETS, DEFAULT_TARGETS
#     from SCons.Subst import SetAllowableExceptions as AllowSubstExceptions
#     from SCons.Tool import CScanner, DScanner, ProgramScanner, SourceFileScanner, Tool
#     from SCons.Util import AddMethod, WhereIs
#     from SCons.Variables import BoolVariable, EnumVariable, ListVariable, PathVariable, PackageVariable, Variables
#     from SCons.Script.Main import (
#         AddOption,
#         BuildTask,
#         CleanTask,
#         DebugOptions,
#         GetBuildFailures,
#         GetOption,
#         PrintHelp,
#         Progress,
#         QuestionTask,
#         SetOption,
#         ValidateOptions,
#     )
#     from SCons.Script.SConscript import (
#         SConsEnvironment,
#         SConsEnvironment as Environment,  # Hack to have Environment() properly recognized as SConsEnvironment.
#         Configure,
#         Return,
#     )
#     from SCons.Environment import (
#         # Environment,  # See above.
#         Base,
#     )

#     Default = SConsEnvironment.Default
#     EnsurePythonVersion = SConsEnvironment.EnsurePythonVersion
#     EnsureSConsVersion = SConsEnvironment.EnsureSConsVersion
#     Exit = SConsEnvironment.Exit
#     Export = SConsEnvironment.Export
#     GetLaunchDir = SConsEnvironment.GetLaunchDir
#     Help = SConsEnvironment.Help
#     Import = SConsEnvironment.Import
#     SConscript = SConsEnvironment.SConscript
#     SConscriptChdir = SConsEnvironment.SConscriptChdir

#     AddPostAction = Base.AddPostAction
#     AddPreAction = Base.AddPreAction
#     Alias = Base.Alias
#     AlwaysBuild = Base.AlwaysBuild
#     CacheDir = Base.CacheDir
#     Clean = Base.Clean
#     Command = Base.Command
#     Decider = Base.Decider
#     Depends = Base.Depends
#     Dir = Base.Dir
#     Entry = Base.Entry
#     Execute = Base.Execute
#     File = Base.File
#     FindFile = Base.FindFile
#     FindInstalledFiles = Base.FindInstalledFiles
#     FindSourceFiles = Base.FindSourceFiles
#     Flatten = Base.Flatten
#     GetBuildPath = Base.GetBuildPath
#     Glob = Base.Glob
#     Ignore = Base.Ignore
#     Install = Base.Install
#     InstallAs = Base.InstallAs
#     InstallVersionedLib = Base.InstallVersionedLib
#     Literal = Base.Literal
#     Local = Base.Local
#     NoCache = Base.NoCache
#     NoClean = Base.NoClean
#     ParseDepends = Base.ParseDepends
#     Precious = Base.Precious
#     PyPackageDir = Base.PyPackageDir
#     Repository = Base.Repository
#     Requires = Base.Requires
#     SConsignFile = Base.SConsignFile
#     SideEffect = Base.SideEffect
#     Split = Base.Split
#     Tag = Base.Tag
#     Value = Base.Value
#     VariantDir = Base.VariantDir

#     env: SConsEnvironment
#     env_modules: SConsEnvironment
