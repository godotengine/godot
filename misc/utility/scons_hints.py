"""
Adds type hints to SCons scripts. Implemented via
`from misc.utility.scons_hints import *`.

This is NOT a 1-1 representation of what the defines will represent in an
SCons build, as proxies are almost always utilized instead. Rather, this is
a means of tracing back what those proxies are calling to in the first place.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # ruff: noqa: F401
    from SCons.Action import Action
    from SCons.Builder import Builder
    from SCons.Defaults import Chmod, Copy, CScan, DefaultEnvironment, Delete, DirScanner, Mkdir, Move, Touch
    from SCons.Environment import Base
    from SCons.Platform import Platform
    from SCons.Platform.virtualenv import Virtualenv
    from SCons.Scanner import FindPathDirs, ScannerBase
    from SCons.Script import ARGLIST, ARGUMENTS, BUILD_TARGETS, COMMAND_LINE_TARGETS, DEFAULT_TARGETS
    from SCons.Script.Main import (
        AddOption,
        BuildTask,
        CleanTask,
        DebugOptions,
        GetBuildFailures,
        GetOption,
        PrintHelp,
        Progress,
        QuestionTask,
        SetOption,
        ValidateOptions,
    )
    from SCons.Script.SConscript import Configure, Return, SConsEnvironment, call_stack
    from SCons.Script.SConscript import SConsEnvironment as Environment
    from SCons.Subst import SetAllowableExceptions as AllowSubstExceptions
    from SCons.Tool import CScanner, DScanner, ProgramScanner, SourceFileScanner, Tool
    from SCons.Util import AddMethod, WhereIs
    from SCons.Variables import BoolVariable, EnumVariable, ListVariable, PackageVariable, PathVariable, Variables

    # Global functions
    GetSConsVersion = SConsEnvironment.GetSConsVersion
    EnsurePythonVersion = SConsEnvironment.EnsurePythonVersion
    EnsureSConsVersion = SConsEnvironment.EnsureSConsVersion
    Exit = SConsEnvironment.Exit
    GetLaunchDir = SConsEnvironment.GetLaunchDir
    SConscriptChdir = SConsEnvironment.SConscriptChdir

    # SConsEnvironment functions
    Default = SConsEnvironment(DefaultEnvironment()).Default
    Export = SConsEnvironment(DefaultEnvironment()).Export
    Help = SConsEnvironment(DefaultEnvironment()).Help
    Import = SConsEnvironment(DefaultEnvironment()).Import
    SConscript = SConsEnvironment(DefaultEnvironment()).SConscript

    # Environment functions
    AddPostAction = DefaultEnvironment().AddPostAction
    AddPreAction = DefaultEnvironment().AddPreAction
    Alias = DefaultEnvironment().Alias
    AlwaysBuild = DefaultEnvironment().AlwaysBuild
    CacheDir = DefaultEnvironment().CacheDir
    Clean = DefaultEnvironment().Clean
    Command = DefaultEnvironment().Command
    Decider = DefaultEnvironment().Decider
    Depends = DefaultEnvironment().Depends
    Dir = DefaultEnvironment().Dir
    Entry = DefaultEnvironment().Entry
    Execute = DefaultEnvironment().Execute
    File = DefaultEnvironment().File
    FindFile = DefaultEnvironment().FindFile
    FindInstalledFiles = DefaultEnvironment().FindInstalledFiles
    FindSourceFiles = DefaultEnvironment().FindSourceFiles
    Flatten = DefaultEnvironment().Flatten
    GetBuildPath = DefaultEnvironment().GetBuildPath
    Glob = DefaultEnvironment().Glob
    Ignore = DefaultEnvironment().Ignore
    Install = DefaultEnvironment().Install
    InstallAs = DefaultEnvironment().InstallAs
    InstallVersionedLib = DefaultEnvironment().InstallVersionedLib
    Literal = DefaultEnvironment().Literal
    Local = DefaultEnvironment().Local
    NoCache = DefaultEnvironment().NoCache
    NoClean = DefaultEnvironment().NoClean
    ParseDepends = DefaultEnvironment().ParseDepends
    Precious = DefaultEnvironment().Precious
    PyPackageDir = DefaultEnvironment().PyPackageDir
    Repository = DefaultEnvironment().Repository
    Requires = DefaultEnvironment().Requires
    SConsignFile = DefaultEnvironment().SConsignFile
    SideEffect = DefaultEnvironment().SideEffect
    Split = DefaultEnvironment().Split
    Tag = DefaultEnvironment().Tag
    Value = DefaultEnvironment().Value
    VariantDir = DefaultEnvironment().VariantDir

    env: SConsEnvironment
    env_modules: SConsEnvironment
