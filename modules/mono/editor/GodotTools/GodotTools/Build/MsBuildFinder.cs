using System;
using System.Collections.Generic;
using System.IO;
using Godot;
using GodotTools.Ides.Rider;
using GodotTools.Internals;
using Directory = System.IO.Directory;
using Environment = System.Environment;
using File = System.IO.File;
using Path = System.IO.Path;
using OS = GodotTools.Utils.OS;

namespace GodotTools.Build
{
    public static class MsBuildFinder
    {
        private static string _msbuildToolsPath = string.Empty;
        private static string _msbuildUnixPath = string.Empty;

        public static (string, BuildTool) FindMsBuild()
        {
            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var buildTool = (BuildTool)editorSettings.GetSetting("mono/builds/build_tool");

            if (OS.IsWindows)
            {
                switch (buildTool)
                {
                    case BuildTool.DotnetCli:
                    {
                        string dotnetCliPath = OS.PathWhich("dotnet");
                        if (!string.IsNullOrEmpty(dotnetCliPath))
                            return (dotnetCliPath, BuildTool.DotnetCli);
                        GD.PushError($"Cannot find executable for '{BuildManager.PropNameDotnetCli}'. Fallback to MSBuild from Visual Studio.");
                        goto case BuildTool.MsBuildVs;
                    }
                    case BuildTool.MsBuildVs:
                    {
                        if (string.IsNullOrEmpty(_msbuildToolsPath) || !File.Exists(_msbuildToolsPath))
                        {
                            // Try to search it again if it wasn't found last time or if it was removed from its location
                            _msbuildToolsPath = FindMsBuildToolsPathOnWindows();

                            if (string.IsNullOrEmpty(_msbuildToolsPath))
                                throw new FileNotFoundException($"Cannot find executable for '{BuildManager.PropNameMSBuildVs}'.");
                        }

                        if (!_msbuildToolsPath.EndsWith("\\"))
                            _msbuildToolsPath += "\\";

                        return (Path.Combine(_msbuildToolsPath, "MSBuild.exe"), BuildTool.MsBuildVs);
                    }
                    case BuildTool.MsBuildMono:
                    {
                        string msbuildPath = Path.Combine(Internal.MonoWindowsInstallRoot, "bin", "msbuild.bat");

                        if (!File.Exists(msbuildPath))
                            throw new FileNotFoundException($"Cannot find executable for '{BuildManager.PropNameMSBuildMono}'. Tried with path: {msbuildPath}");

                        return (msbuildPath, BuildTool.MsBuildMono);
                    }
                    case BuildTool.JetBrainsMsBuild:
                    {
                        string editorPath = (string)editorSettings.GetSetting(RiderPathManager.EditorPathSettingName);

                        if (!File.Exists(editorPath))
                            throw new FileNotFoundException($"Cannot find Rider executable. Tried with path: {editorPath}");

                        var riderDir = new FileInfo(editorPath).Directory?.Parent;

                        string msbuildPath = Path.Combine(riderDir.FullName, @"tools\MSBuild\Current\Bin\MSBuild.exe");

                        if (!File.Exists(msbuildPath))
                            throw new FileNotFoundException($"Cannot find executable for '{BuildManager.PropNameMSBuildJetBrains}'. Tried with path: {msbuildPath}");

                        return (msbuildPath, BuildTool.JetBrainsMsBuild);
                    }
                    default:
                        throw new IndexOutOfRangeException("Invalid build tool in editor settings");
                }
            }

            if (OS.IsUnixLike)
            {
                switch (buildTool)
                {
                    case BuildTool.DotnetCli:
                    {
                        string dotnetCliPath = FindBuildEngineOnUnix("dotnet");
                        if (!string.IsNullOrEmpty(dotnetCliPath))
                            return (dotnetCliPath, BuildTool.DotnetCli);
                        GD.PushError($"Cannot find executable for '{BuildManager.PropNameDotnetCli}'. Fallback to MSBuild from Mono.");
                        goto case BuildTool.MsBuildMono;
                    }
                    case BuildTool.MsBuildMono:
                    {
                        if (string.IsNullOrEmpty(_msbuildUnixPath) || !File.Exists(_msbuildUnixPath))
                        {
                            // Try to search it again if it wasn't found last time or if it was removed from its location
                            _msbuildUnixPath = FindBuildEngineOnUnix("msbuild");
                        }

                        if (string.IsNullOrEmpty(_msbuildUnixPath))
                            throw new FileNotFoundException($"Cannot find binary for '{BuildManager.PropNameMSBuildMono}'");

                        return (_msbuildUnixPath, BuildTool.MsBuildMono);
                    }
                    default:
                        throw new IndexOutOfRangeException("Invalid build tool in editor settings");
                }
            }

            throw new PlatformNotSupportedException();
        }

        private static IEnumerable<string> MsBuildHintDirs
        {
            get
            {
                var result = new List<string>();

                if (OS.IsOSX)
                {
                    result.Add("/Library/Frameworks/Mono.framework/Versions/Current/bin/");
                    result.Add("/opt/local/bin/");
                    result.Add("/usr/local/var/homebrew/linked/mono/bin/");
                    result.Add("/usr/local/bin/");
                    result.Add("/usr/local/bin/dotnet/");
                    result.Add("/usr/local/share/dotnet/");
                }

                result.Add("/opt/novell/mono/bin/");

                return result;
            }
        }

        private static string FindBuildEngineOnUnix(string name)
        {
            string ret = OS.PathWhich(name);

            if (!string.IsNullOrEmpty(ret))
                return ret;

            string retFallback = OS.PathWhich($"{name}.exe");

            if (!string.IsNullOrEmpty(retFallback))
                return retFallback;

            foreach (string hintDir in MsBuildHintDirs)
            {
                string hintPath = Path.Combine(hintDir, name);

                if (File.Exists(hintPath))
                    return hintPath;
            }

            return string.Empty;
        }

        private static string FindMsBuildToolsPathOnWindows()
        {
            if (!OS.IsWindows)
                throw new PlatformNotSupportedException();

            // Try to find 15.0 with vswhere

            string[] envNames = Internal.GodotIs32Bits() ?
                envNames = new[] { "ProgramFiles", "ProgramW6432" } :
                envNames = new[] { "ProgramFiles(x86)", "ProgramFiles" };

            string vsWherePath = null;
            foreach (var envName in envNames)
            {
                vsWherePath = Environment.GetEnvironmentVariable(envName);
                if (!string.IsNullOrEmpty(vsWherePath))
                {
                    vsWherePath += "\\Microsoft Visual Studio\\Installer\\vswhere.exe";
                    if (File.Exists(vsWherePath))
                        break;
                }

                vsWherePath = null;
            }

            var vsWhereArgs = new[] {"-latest", "-products", "*", "-requires", "Microsoft.Component.MSBuild"};

            var outputArray = new Godot.Collections.Array<string>();
            int exitCode = Godot.OS.Execute(vsWherePath, vsWhereArgs,
                blocking: true, output: (Godot.Collections.Array)outputArray);

            if (exitCode != 0)
                return string.Empty;

            if (outputArray.Count == 0)
                return string.Empty;

            var lines = outputArray[0].Split('\n');

            foreach (string line in lines)
            {
                int sepIdx = line.IndexOf(':');

                if (sepIdx <= 0)
                    continue;

                string key = line.Substring(0, sepIdx); // No need to trim

                if (key != "installationPath")
                    continue;

                string value = line.Substring(sepIdx + 1).StripEdges();

                if (string.IsNullOrEmpty(value))
                    throw new FormatException("installationPath value is empty");

                if (!value.EndsWith("\\"))
                    value += "\\";

                // Since VS2019, the directory is simply named "Current"
                string msbuildDir = Path.Combine(value, "MSBuild\\Current\\Bin");

                if (Directory.Exists(msbuildDir))
                    return msbuildDir;

                // Directory name "15.0" is used in VS 2017
                return Path.Combine(value, "MSBuild\\15.0\\Bin");
            }

            return string.Empty;
        }
    }
}
