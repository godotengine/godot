import SCons.Action
import SCons.Builder
import SCons.Defaults
import SCons.Environment
import SCons.Node.FS
import SCons.Platform
import SCons.Platform.virtualenv
import SCons.Scanner
import SCons.SConf
import SCons.Script
import SCons.Script.Main
import SCons.Script.SConscript as _SConscript
import SCons.Subst
import SCons.Tool
import SCons.Util
import SCons.Variables

Action = SCons.Action.Action

Builder = SCons.Builder.Builder

Chmod = SCons.Defaults.Chmod
Copy = SCons.Defaults.Copy
CScan = SCons.Defaults.CScan
DefaultEnvironment = SCons.Defaults.DefaultEnvironment
Delete = SCons.Defaults.Delete
DirScanner = SCons.Defaults.DirScanner
Mkdir = SCons.Defaults.Mkdir
Move = SCons.Defaults.Move
Touch = SCons.Defaults.Touch

SConsEnvironment = _SConscript.SConsEnvironment
Environment = _SConscript.SConsEnvironment
Configure = _SConscript.Configure
Return = _SConscript.Return

Base = SCons.Environment.Environment

Platform = SCons.Platform.Platform

Virtualenv = SCons.Platform.virtualenv.Virtualenv

FindPathDirs = SCons.Scanner.FindPathDirs
ScannerBase = SCons.Scanner.ScannerBase

ARGLIST = SCons.Script.ARGLIST
ARGUMENTS = SCons.Script.ARGUMENTS
BUILD_TARGETS = SCons.Script.BUILD_TARGETS
COMMAND_LINE_TARGETS = SCons.Script.COMMAND_LINE_TARGETS
DEFAULT_TARGETS = SCons.Script.DEFAULT_TARGETS

AddOption = SCons.Script.Main.AddOption
BuildTask = SCons.Script.Main.BuildTask
CleanTask = SCons.Script.Main.CleanTask
DebugOptions = SCons.Script.Main.DebugOptions
GetBuildFailures = SCons.Script.Main.GetBuildFailures
GetOption = SCons.Script.Main.GetOption
PrintHelp = SCons.Script.Main.PrintHelp
Progress = SCons.Script.Main.Progress
QuestionTask = SCons.Script.Main.QuestionTask
SetOption = SCons.Script.Main.SetOption
ValidateOptions = SCons.Script.Main.ValidateOptions

AllowSubstExceptions = SCons.Subst.SetAllowableExceptions

CScanner = SCons.Tool.CScanner
DScanner = SCons.Tool.DScanner
ProgramScanner = SCons.Tool.ProgramScanner
SourceFileScanner = SCons.Tool.SourceFileScanner
Tool = SCons.Tool.Tool

AddMethod = SCons.Util.AddMethod
WhereIs = SCons.Util.WhereIs

BoolVariable = SCons.Variables.BoolVariable
EnumVariable = SCons.Variables.EnumVariable
ListVariable = SCons.Variables.ListVariable
PathVariable = SCons.Variables.PathVariable
PackageVariable = SCons.Variables.PackageVariable
Variables = SCons.Variables.Variables

Default = SConsEnvironment.Default
EnsurePythonVersion = SConsEnvironment.EnsurePythonVersion
EnsureSConsVersion = SConsEnvironment.EnsureSConsVersion
Exit = SConsEnvironment.Exit
Export = SConsEnvironment.Export
GetLaunchDir = SConsEnvironment.GetLaunchDir
Help = SConsEnvironment.Help
Import = SConsEnvironment.Import
SConscript = SConsEnvironment.SConscript
SConscriptChdir = SConsEnvironment.SConscriptChdir

AddPostAction = Base.AddPostAction
AddPreAction = Base.AddPreAction
Alias = Base.Alias
AlwaysBuild = Base.AlwaysBuild
CacheDir = Base.CacheDir
Clean = Base.Clean
Command = Base.Command
Decider = Base.Decider
Depends = Base.Depends
Dir = Base.Dir
Entry = Base.Entry
Execute = Base.Execute
File = Base.File
FindFile = Base.FindFile
FindInstalledFiles = Base.FindInstalledFiles
FindSourceFiles = Base.FindSourceFiles
Flatten = Base.Flatten
GetBuildPath = Base.GetBuildPath
Glob = Base.Glob
Ignore = Base.Ignore
Install = Base.Install
InstallAs = Base.InstallAs
InstallVersionedLib = Base.InstallVersionedLib
Literal = Base.Literal
Local = Base.Local
NoCache = Base.NoCache
NoClean = Base.NoClean
ParseDepends = Base.ParseDepends
Precious = Base.Precious
PyPackageDir = Base.PyPackageDir
Repository = Base.Repository
Requires = Base.Requires
SConsignFile = Base.SConsignFile
SideEffect = Base.SideEffect
Split = Base.Split
Tag = Base.Tag
Value = Base.Value
VariantDir = Base.VariantDir

env: _SConscript.SConsEnvironment
env_modules: _SConscript.SConsEnvironment
