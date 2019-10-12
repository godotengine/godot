using System;
using System.Collections.Generic;
using System.IO;
using Godot;
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

        public static string FindMsBuild()
        {
            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var buildTool = (BuildManager.BuildTool) editorSettings.GetSetting("mono/builds/build_tool");

            if (OS.IsWindows())
            {
                switch (buildTool)
                {
                    case BuildManager.BuildTool.MsBuildVs:
                    {
                        if (_msbuildToolsPath.Empty() || !File.Exists(_msbuildToolsPath))
                        {
                            // Try to search it again if it wasn't found last time or if it was removed from its location
                            _msbuildToolsPath = FindMsBuildToolsPathOnWindows();

                            if (_msbuildToolsPath.Empty())
                            {
                                throw new FileNotFoundException($"Cannot find executable for '{BuildManager.PropNameMsbuildVs}'.");
                            }
                        }

                        if (!_msbuildToolsPath.EndsWith("\\"))
                            _msbuildToolsPath += "\\";

                        return Path.Combine(_msbuildToolsPath, "MSBuild.exe");
                    }
                    case BuildManager.BuildTool.MsBuildMono:
                    {
                        string msbuildPath = Path.Combine(Internal.MonoWindowsInstallRoot, "bin", "msbuild.bat");

                        if (!File.Exists(msbuildPath))
                        {
                            throw new FileNotFoundException($"Cannot find executable for '{BuildManager.PropNameMsbuildMono}'. Tried with path: {msbuildPath}");
                        }

                        return msbuildPath;
                    }
                    default:
                        throw new IndexOutOfRangeException("Invalid build tool in editor settings");
                }
            }

            if (OS.IsUnix())
            {
                if (buildTool == BuildManager.BuildTool.MsBuildMono)
                {
                    if (_msbuildUnixPath.Empty() || !File.Exists(_msbuildUnixPath))
                    {
                        // Try to search it again if it wasn't found last time or if it was removed from its location
                        _msbuildUnixPath = FindBuildEngineOnUnix("msbuild");
                    }

                    if (_msbuildUnixPath.Empty())
                    {
                        throw new FileNotFoundException($"Cannot find binary for '{BuildManager.PropNameMsbuildMono}'");
                    }

                    return _msbuildUnixPath;
                }
                else
                {
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

                if (OS.IsOSX())
                {
                    result.Add("/Library/Frameworks/Mono.framework/Versions/Current/bin/");
                    result.Add("/usr/local/var/homebrew/linked/mono/bin/");
                }

                result.Add("/opt/novell/mono/bin/");

                return result;
            }
        }

        private static string FindBuildEngineOnUnix(string name)
        {
            string ret = OS.PathWhich(name);

            if (!ret.Empty())
                return ret;

            string retFallback = OS.PathWhich($"{name}.exe");

            if (!retFallback.Empty())
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
            if (!OS.IsWindows())
                throw new PlatformNotSupportedException();

            // Try to find 15.0 with vswhere

            string vsWherePath = Environment.GetEnvironmentVariable(Internal.GodotIs32Bits() ? "ProgramFiles" : "ProgramFiles(x86)");
            vsWherePath += "\\Microsoft Visual Studio\\Installer\\vswhere.exe";

            var vsWhereArgs = new[] {"-latest", "-products", "*", "-requires", "Microsoft.Component.MSBuild"};

            var outputArray = new Godot.Collections.Array<string>();
            int exitCode = Godot.OS.Execute(vsWherePath, vsWhereArgs,
                blocking: true, output: (Godot.Collections.Array) outputArray);

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

                if (value.Empty())
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
