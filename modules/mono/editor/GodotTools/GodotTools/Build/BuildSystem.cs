using GodotTools.Core;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using GodotTools.BuildLogger;
using GodotTools.Internals;
using GodotTools.Utils;
using Directory = System.IO.Directory;

namespace GodotTools.Build
{
    public static class BuildSystem
    {
        private static string GetMsBuildPath()
        {
            string msbuildPath = MsBuildFinder.FindMsBuild();

            if (msbuildPath == null)
                throw new FileNotFoundException("Cannot find the MSBuild executable.");

            return msbuildPath;
        }

        private static string MonoWindowsBinDir
        {
            get
            {
                string monoWinBinDir = Path.Combine(Internal.MonoWindowsInstallRoot, "bin");

                if (!Directory.Exists(monoWinBinDir))
                    throw new FileNotFoundException("Cannot find the Windows Mono install bin directory.");

                return monoWinBinDir;
            }
        }

        private static Godot.EditorSettings EditorSettings =>
            GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();

        private static bool UsingMonoMsBuildOnWindows
        {
            get
            {
                if (OS.IsWindows())
                {
                    return (BuildManager.BuildTool) EditorSettings.GetSetting("mono/builds/build_tool")
                           == BuildManager.BuildTool.MsBuildMono;
                }

                return false;
            }
        }

        private static bool PrintBuildOutput =>
            (bool) EditorSettings.GetSetting("mono/builds/print_build_output");

        private static Process LaunchBuild(string solution, string config, string loggerOutputDir, IEnumerable<string> customProperties = null)
        {
            var customPropertiesList = new List<string>();

            if (customProperties != null)
                customPropertiesList.AddRange(customProperties);

            string compilerArgs = BuildArguments(solution, config, loggerOutputDir, customPropertiesList);

            var startInfo = new ProcessStartInfo(GetMsBuildPath(), compilerArgs);

            bool redirectOutput = !IsDebugMsBuildRequested() && !PrintBuildOutput;

            if (!redirectOutput || Godot.OS.IsStdoutVerbose())
                Console.WriteLine($"Running: \"{startInfo.FileName}\" {startInfo.Arguments}");

            startInfo.RedirectStandardOutput = redirectOutput;
            startInfo.RedirectStandardError = redirectOutput;
            startInfo.UseShellExecute = false;

            if (UsingMonoMsBuildOnWindows)
            {
                // These environment variables are required for Mono's MSBuild to find the compilers.
                // We use the batch files in Mono's bin directory to make sure the compilers are executed with mono.
                string monoWinBinDir = MonoWindowsBinDir;
                startInfo.EnvironmentVariables.Add("CscToolExe", Path.Combine(monoWinBinDir, "csc.bat"));
                startInfo.EnvironmentVariables.Add("VbcToolExe", Path.Combine(monoWinBinDir, "vbc.bat"));
                startInfo.EnvironmentVariables.Add("FscToolExe", Path.Combine(monoWinBinDir, "fsharpc.bat"));
            }

            // Needed when running from Developer Command Prompt for VS
            RemovePlatformVariable(startInfo.EnvironmentVariables);

            var process = new Process {StartInfo = startInfo};

            process.Start();

            if (redirectOutput)
            {
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            }

            return process;
        }

        public static int Build(BuildInfo buildInfo)
        {
            return Build(buildInfo.Solution, buildInfo.Configuration,
                buildInfo.LogsDirPath, buildInfo.CustomProperties);
        }

        public static async Task<int> BuildAsync(BuildInfo buildInfo)
        {
            return await BuildAsync(buildInfo.Solution, buildInfo.Configuration,
                buildInfo.LogsDirPath, buildInfo.CustomProperties);
        }

        public static int Build(string solution, string config, string loggerOutputDir, IEnumerable<string> customProperties = null)
        {
            using (var process = LaunchBuild(solution, config, loggerOutputDir, customProperties))
            {
                process.WaitForExit();

                return process.ExitCode;
            }
        }

        public static async Task<int> BuildAsync(string solution, string config, string loggerOutputDir, IEnumerable<string> customProperties = null)
        {
            using (var process = LaunchBuild(solution, config, loggerOutputDir, customProperties))
            {
                await process.WaitForExitAsync();

                return process.ExitCode;
            }
        }

        private static string BuildArguments(string solution, string config, string loggerOutputDir, List<string> customProperties)
        {
            string arguments = $@"""{solution}"" /v:normal /t:Build ""/p:{"Configuration=" + config}"" " +
                               $@"""/l:{typeof(GodotBuildLogger).FullName},{GodotBuildLogger.AssemblyPath};{loggerOutputDir}""";

            foreach (string customProperty in customProperties)
            {
                arguments += " /p:" + customProperty;
            }

            return arguments;
        }

        private static void RemovePlatformVariable(StringDictionary environmentVariables)
        {
            // EnvironmentVariables is case sensitive? Seriously?

            var platformEnvironmentVariables = new List<string>();

            foreach (string env in environmentVariables.Keys)
            {
                if (env.ToUpper() == "PLATFORM")
                    platformEnvironmentVariables.Add(env);
            }

            foreach (string env in platformEnvironmentVariables)
                environmentVariables.Remove(env);
        }

        private static bool IsDebugMsBuildRequested()
        {
            return Environment.GetEnvironmentVariable("GODOT_DEBUG_MSBUILD")?.Trim() == "1";
        }
    }
}
