"""
Adds type hints to SCons scripts. Implemented via
`from misc.utility.scons_hints import *`.

This is NOT a 1-1 representation of what the defines will represent in an
SCons build, as proxies are almost always utilized instead. Rather, this is
a means of tracing back what those proxies are calling to in the first place.
"""

from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    __all__ = ()
else:
    __all__ = (
        # SCons.Action.
        "Action",
        # SCons.Builder.
        "Builder",
        # SCons.Defaults.
        "Chmod",
        "Copy",
        "CScan",
        "DefaultEnvironment",
        "Delete",
        "DirScanner",
        "Mkdir",
        "Move",
        "Touch",
        # SCons.Environment.
        "Environment",
        # SCons.Platform.
        "Platform",
        # SCons.Platform.virtualenv.
        "Virtualenv",
        # SCons.Scanner.
        "FindPathDirs",
        "ScannerBase",
        # SCons.Script.
        "ARGLIST",
        "ARGUMENTS",
        "BUILD_TARGETS",
        "COMMAND_LINE_TARGETS",
        "DEFAULT_TARGETS",
        # SCons.Script.Main.
        "AddOption",
        "BuildTask",
        "CleanTask",
        "DebugOptions",
        "GetBuildFailures",
        "GetOption",
        "PrintHelp",
        "Progress",
        "QuestionTask",
        "SetOption",
        "ValidateOptions",
        # SCons.Script.SConscript.
        "Configure",
        "Return",
        "SConsEnvironment",
        "call_stack",
        # SCons.Subst.
        "AllowSubstExceptions",
        # SCons.Tool.
        "CScanner",
        "DScanner",
        "ProgramScanner",
        "SourceFileScanner",
        "Tool",
        # SCons.Util.
        "AddMethod",
        "WhereIs",
        # SCons.Variables.
        "BoolVariable",
        "EnumVariable",
        "ListVariable",
        "PackageVariable",
        "PathVariable",
        "Variables",
        # Global functions.
        "GetSConsVersion",
        "EnsurePythonVersion",
        "EnsureSConsVersion",
        "Exit",
        "GetLaunchDir",
        "SConscriptChdir",
        # SConsEnvironment functions.
        "Default",
        "Export",
        "Help",
        "Import",
        "SConscript",
        # Environment functions.
        "AddPostAction",
        "AddPreAction",
        "Alias",
        "AlwaysBuild",
        "CacheDir",
        "Clean",
        "Command",
        "Decider",
        "Depends",
        "Dir",
        "Entry",
        "Execute",
        "File",
        "FindFile",
        "FindInstalledFiles",
        "FindSourceFiles",
        "Flatten",
        "GetBuildPath",
        "Glob",
        "Ignore",
        "Install",
        "InstallAs",
        "InstallVersionedLib",
        "Literal",
        "Local",
        "NoCache",
        "NoClean",
        "ParseDepends",
        "Precious",
        "PyPackageDir",
        "Repository",
        "Requires",
        "SConsignFile",
        "SideEffect",
        "Split",
        "Tag",
        "Value",
        "VariantDir",
        # Global default builders.
        "CFile",
        "CXXFile",
        "DVI",
        "Jar",
        "Java",
        "JavaH",
        "Library",
        "LoadableModule",
        "M4",
        "MSVSProject",
        "Object",
        "PCH",
        "PDF",
        "PostScript",
        "Program",
        "RES",
        "RMIC",
        "SharedLibrary",
        "SharedObject",
        "StaticLibrary",
        "StaticObject",
        "Substfile",
        "Tar",
        "Textfile",
        "TypeLibrary",
        "Zip",
        "Package",
        # Our exported environments.
        "env",
        "env_modules",
    )

    from SCons.Action import Action
    from SCons.Builder import Builder
    from SCons.Defaults import Chmod, Copy, CScan, DefaultEnvironment, Delete, DirScanner, Mkdir, Move, Touch
    from SCons.Environment import Environment
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
    from SCons.Subst import SetAllowableExceptions as AllowSubstExceptions
    from SCons.Tool import CScanner, DScanner, ProgramScanner, SourceFileScanner, Tool
    from SCons.Util import AddMethod, WhereIs
    from SCons.Variables import BoolVariable, EnumVariable, ListVariable, PackageVariable, PathVariable, Variables

    # Global functions.
    GetSConsVersion = SConsEnvironment.GetSConsVersion
    EnsurePythonVersion = SConsEnvironment.EnsurePythonVersion
    EnsureSConsVersion = SConsEnvironment.EnsureSConsVersion
    Exit = SConsEnvironment.Exit
    GetLaunchDir = SConsEnvironment.GetLaunchDir
    SConscriptChdir = SConsEnvironment.SConscriptChdir

    # SConsEnvironment functions.
    Default = SConsEnvironment().Default
    Export = SConsEnvironment().Export
    Help = SConsEnvironment().Help
    Import = SConsEnvironment().Import
    SConscript = SConsEnvironment().SConscript

    # Environment functions.
    AddPostAction = Environment().AddPostAction
    AddPreAction = Environment().AddPreAction
    Alias = Environment().Alias
    AlwaysBuild = Environment().AlwaysBuild
    CacheDir = Environment().CacheDir
    Clean = Environment().Clean
    Command = Environment().Command
    Decider = Environment().Decider
    Depends = Environment().Depends
    Dir = Environment().Dir
    Entry = Environment().Entry
    Execute = Environment().Execute
    File = Environment().File
    FindFile = Environment().FindFile
    FindInstalledFiles = Environment().FindInstalledFiles
    FindSourceFiles = Environment().FindSourceFiles
    Flatten = Environment().Flatten
    GetBuildPath = Environment().GetBuildPath
    Glob = Environment().Glob
    Ignore = Environment().Ignore
    Install = Environment().Install
    InstallAs = Environment().InstallAs
    InstallVersionedLib = Environment().InstallVersionedLib
    Literal = Environment().Literal
    Local = Environment().Local
    NoCache = Environment().NoCache
    NoClean = Environment().NoClean
    ParseDepends = Environment().ParseDepends
    Precious = Environment().Precious
    PyPackageDir = Environment().PyPackageDir
    Repository = Environment().Repository
    Requires = Environment().Requires
    SConsignFile = Environment().SConsignFile
    SideEffect = Environment().SideEffect
    Split = Environment().Split
    Tag = Environment().Tag
    Value = Environment().Value
    VariantDir = Environment().VariantDir

    # Global default builders.
    CFile = Environment().CFile
    CXXFile = Environment().CXXFile
    DVI = Environment().DVI
    Jar = Environment().Jar
    Java = Environment().Java
    JavaH = Environment().JavaH
    Library = Environment().Library
    LoadableModule = Environment().LoadableModule
    M4 = Environment().M4
    MSVSProject = Environment().MSVSProject
    Object = Environment().Object
    PCH = Environment().PCH
    PDF = Environment().PDF
    PostScript = Environment().PostScript
    Program = Environment().Program
    RES = Environment().RES
    RMIC = Environment().RMIC
    SharedLibrary = Environment().SharedLibrary
    SharedObject = Environment().SharedObject
    StaticLibrary = Environment().StaticLibrary
    StaticObject = Environment().StaticObject
    Substfile = Environment().Substfile
    Tar = Environment().Tar
    Textfile = Environment().Textfile
    TypeLibrary = Environment().TypeLibrary
    Zip = Environment().Zip
    Package = Environment().Package

    # Our exported environments.
    env = SConsEnvironment()
    env_modules = SConsEnvironment()
