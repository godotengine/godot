using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using GodotTools.BuildLogger;
using GodotTools.Utils;

namespace GodotTools.Build
{
    public static class BuildSystem
    {
        private static Process LaunchBuild(BuildInfo buildInfo, Action<string> stdOutHandler,
            Action<string> stdErrHandler)
        {
            string dotnetPath = DotNetFinder.FindDotNetExe();

            if (dotnetPath == null)
                throw new FileNotFoundException("Cannot find the dotnet executable.");

            var startInfo = new ProcessStartInfo(dotnetPath);

            BuildArguments(buildInfo, startInfo.ArgumentList);

            string launchMessage = startInfo.GetCommandLineDisplay(new StringBuilder("Running: ")).ToString();
            stdOutHandler?.Invoke(launchMessage);
            if (Godot.OS.IsStdOutVerbose())
                Console.WriteLine(launchMessage);

            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.UseShellExecute = false;
            startInfo.CreateNoWindow = true;

            // Needed when running from Developer Command Prompt for VS
            RemovePlatformVariable(startInfo.EnvironmentVariables);

            var process = new Process { StartInfo = startInfo };

            if (stdOutHandler != null)
                process.OutputDataReceived += (_, e) => stdOutHandler.Invoke(e.Data);
            if (stdErrHandler != null)
                process.ErrorDataReceived += (_, e) => stdErrHandler.Invoke(e.Data);

            process.Start();

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            return process;
        }

        public static int Build(BuildInfo buildInfo, Action<string> stdOutHandler, Action<string> stdErrHandler)
        {
            using (var process = LaunchBuild(buildInfo, stdOutHandler, stdErrHandler))
            {
                process.WaitForExit();

                return process.ExitCode;
            }
        }

        public static async Task<int> BuildAsync(BuildInfo buildInfo, Action<string> stdOutHandler,
            Action<string> stdErrHandler)
        {
            using (var process = LaunchBuild(buildInfo, stdOutHandler, stdErrHandler))
            {
                await process.WaitForExitAsync();

                return process.ExitCode;
            }
        }

        private static Process LaunchPublish(BuildInfo buildInfo, Action<string> stdOutHandler,
            Action<string> stdErrHandler)
        {
            string dotnetPath = DotNetFinder.FindDotNetExe();

            if (dotnetPath == null)
                throw new FileNotFoundException("Cannot find the dotnet executable.");

            var startInfo = new ProcessStartInfo(dotnetPath);

            BuildPublishArguments(buildInfo, startInfo.ArgumentList);

            string launchMessage = startInfo.GetCommandLineDisplay(new StringBuilder("Running: ")).ToString();
            stdOutHandler?.Invoke(launchMessage);
            if (Godot.OS.IsStdOutVerbose())
                Console.WriteLine(launchMessage);

            startInfo.RedirectStandardOutput = true;
            startInfo.RedirectStandardError = true;
            startInfo.UseShellExecute = false;

            // Needed when running from Developer Command Prompt for VS
            RemovePlatformVariable(startInfo.EnvironmentVariables);

            var process = new Process { StartInfo = startInfo };

            if (stdOutHandler != null)
                process.OutputDataReceived += (_, e) => stdOutHandler.Invoke(e.Data);
            if (stdErrHandler != null)
                process.ErrorDataReceived += (_, e) => stdErrHandler.Invoke(e.Data);

            process.Start();

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            return process;
        }

        public static int Publish(BuildInfo buildInfo, Action<string> stdOutHandler, Action<string> stdErrHandler)
        {
            using (var process = LaunchPublish(buildInfo, stdOutHandler, stdErrHandler))
            {
                process.WaitForExit();

                return process.ExitCode;
            }
        }

        private static void BuildArguments(BuildInfo buildInfo, Collection<string> arguments)
        {
            // `dotnet clean` / `dotnet build` commands
            arguments.Add(buildInfo.OnlyClean ? "clean" : "build");

            // Solution
            arguments.Add(buildInfo.Solution);

            // `dotnet clean` doesn't recognize these options
            if (!buildInfo.OnlyClean)
            {
                // Restore
                // `dotnet build` restores by default, unless requested not to
                if (!buildInfo.Restore)
                    arguments.Add("--no-restore");

                // Incremental or rebuild
                if (buildInfo.Rebuild)
                    arguments.Add("--no-incremental");
            }

            // Configuration
            arguments.Add("-c");
            arguments.Add(buildInfo.Configuration);

            // Verbosity
            arguments.Add("-v");
            arguments.Add("normal");

            // Logger
            AddLoggerArgument(buildInfo, arguments);

            // Custom properties
            foreach (var customProperty in buildInfo.CustomProperties)
            {
                arguments.Add("-p:" + (string)customProperty);
            }
        }

        private static void BuildPublishArguments(BuildInfo buildInfo, Collection<string> arguments)
        {
            arguments.Add("publish"); // `dotnet publish` command

            // Solution
            arguments.Add(buildInfo.Solution);

            // Restore
            // `dotnet publish` restores by default, unless requested not to
            if (!buildInfo.Restore)
                arguments.Add("--no-restore");

            // Incremental or rebuild
            // TODO: Not supported in `dotnet publish` (https://github.com/dotnet/sdk/issues/11099)
            // if (buildInfo.Rebuild)
            //     arguments.Add("--no-incremental");

            // Configuration
            arguments.Add("-c");
            arguments.Add(buildInfo.Configuration);

            // Runtime Identifier
            arguments.Add("-r");
            arguments.Add(buildInfo.RuntimeIdentifier!);

            // Self-published
            arguments.Add("--self-contained");
            arguments.Add("true");

            // Verbosity
            arguments.Add("-v");
            arguments.Add("normal");

            // Logger
            AddLoggerArgument(buildInfo, arguments);

            // Custom properties
            foreach (var customProperty in buildInfo.CustomProperties)
            {
                arguments.Add("-p:" + (string)customProperty);
            }

            // Publish output directory
            if (buildInfo.PublishOutputDir != null)
            {
                arguments.Add("-o");
                arguments.Add(buildInfo.PublishOutputDir);
            }
        }

        private static void AddLoggerArgument(BuildInfo buildInfo, Collection<string> arguments)
        {
            string buildLoggerPath = Path.Combine(Internals.GodotSharpDirs.DataEditorToolsDir,
                "GodotTools.BuildLogger.dll");

            arguments.Add(
                $"-l:{typeof(GodotBuildLogger).FullName},{buildLoggerPath};{buildInfo.LogsDirPath}");
        }

        private static void RemovePlatformVariable(StringDictionary environmentVariables)
        {
            // EnvironmentVariables is case sensitive? Seriously?

            var platformEnvironmentVariables = new List<string>();

            foreach (string env in environmentVariables.Keys)
            {
                if (env.ToUpperInvariant() == "PLATFORM")
                    platformEnvironmentVariables.Add(env);
            }

            foreach (string env in platformEnvironmentVariables)
                environmentVariables.Remove(env);
        }
    }
}
