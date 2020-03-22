using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using GodotTools.Build;
using GodotTools.Ides.Rider;
using GodotTools.Internals;
using GodotTools.Utils;
using static GodotTools.Internals.Globals;
using File = GodotTools.Utils.File;

namespace GodotTools
{
    public static class BuildManager
    {
        private static readonly List<BuildInfo> BuildsInProgress = new List<BuildInfo>();

        public const string PropNameMsbuildMono = "MSBuild (Mono)";
        public const string PropNameMsbuildVs = "MSBuild (VS Build Tools)";
        public const string PropNameMsbuildJetBrains = "MSBuild (JetBrains Rider)";

        public const string MsBuildIssuesFileName = "msbuild_issues.csv";
        public const string MsBuildLogFileName = "msbuild_log.txt";

        public enum BuildTool
        {
            MsBuildMono,
            MsBuildVs,
            JetBrainsMsBuild
        }

        private static void RemoveOldIssuesFile(BuildInfo buildInfo)
        {
            var issuesFile = GetIssuesFilePath(buildInfo);

            if (!File.Exists(issuesFile))
                return;

            File.Delete(issuesFile);
        }

        private static void ShowBuildErrorDialog(string message)
        {
            GodotSharpEditor.Instance.ShowErrorDialog(message, "Build error");
            GodotSharpEditor.Instance.BottomPanel.ShowBuildTab();
        }

        public static void RestartBuild(BuildTab buildTab) => throw new NotImplementedException();
        public static void StopBuild(BuildTab buildTab) => throw new NotImplementedException();

        private static string GetLogFilePath(BuildInfo buildInfo)
        {
            return Path.Combine(buildInfo.LogsDirPath, MsBuildLogFileName);
        }

        private static string GetIssuesFilePath(BuildInfo buildInfo)
        {
            return Path.Combine(buildInfo.LogsDirPath, MsBuildIssuesFileName);
        }

        private static void PrintVerbose(string text)
        {
            if (Godot.OS.IsStdoutVerbose())
                Godot.GD.Print(text);
        }

        public static bool Build(BuildInfo buildInfo)
        {
            if (BuildsInProgress.Contains(buildInfo))
                throw new InvalidOperationException("A build is already in progress");

            BuildsInProgress.Add(buildInfo);

            try
            {
                BuildTab buildTab = GodotSharpEditor.Instance.BottomPanel.GetBuildTabFor(buildInfo);
                buildTab.OnBuildStart();

                // Required in order to update the build tasks list
                Internal.GodotMainIteration();

                try
                {
                    RemoveOldIssuesFile(buildInfo);
                }
                catch (IOException e)
                {
                    buildTab.OnBuildExecFailed($"Cannot remove issues file: {GetIssuesFilePath(buildInfo)}");
                    Console.Error.WriteLine(e);
                }

                try
                {
                    int exitCode = BuildSystem.Build(buildInfo);

                    if (exitCode != 0)
                        PrintVerbose($"MSBuild exited with code: {exitCode}. Log file: {GetLogFilePath(buildInfo)}");

                    buildTab.OnBuildExit(exitCode == 0 ? BuildTab.BuildResults.Success : BuildTab.BuildResults.Error);

                    return exitCode == 0;
                }
                catch (Exception e)
                {
                    buildTab.OnBuildExecFailed($"The build method threw an exception.\n{e.GetType().FullName}: {e.Message}");
                    Console.Error.WriteLine(e);
                    return false;
                }
            }
            finally
            {
                BuildsInProgress.Remove(buildInfo);
            }
        }

        public static async Task<bool> BuildAsync(BuildInfo buildInfo)
        {
            if (BuildsInProgress.Contains(buildInfo))
                throw new InvalidOperationException("A build is already in progress");

            BuildsInProgress.Add(buildInfo);

            try
            {
                BuildTab buildTab = GodotSharpEditor.Instance.BottomPanel.GetBuildTabFor(buildInfo);

                try
                {
                    RemoveOldIssuesFile(buildInfo);
                }
                catch (IOException e)
                {
                    buildTab.OnBuildExecFailed($"Cannot remove issues file: {GetIssuesFilePath(buildInfo)}");
                    Console.Error.WriteLine(e);
                }

                try
                {
                    int exitCode = await BuildSystem.BuildAsync(buildInfo);

                    if (exitCode != 0)
                        PrintVerbose($"MSBuild exited with code: {exitCode}. Log file: {GetLogFilePath(buildInfo)}");

                    buildTab.OnBuildExit(exitCode == 0 ? BuildTab.BuildResults.Success : BuildTab.BuildResults.Error);

                    return exitCode == 0;
                }
                catch (Exception e)
                {
                    buildTab.OnBuildExecFailed($"The build method threw an exception.\n{e.GetType().FullName}: {e.Message}");
                    Console.Error.WriteLine(e);
                    return false;
                }
            }
            finally
            {
                BuildsInProgress.Remove(buildInfo);
            }
        }

        public static bool BuildProjectBlocking(string config, IEnumerable<string> godotDefines)
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return true; // No solution to build

            // Make sure the API assemblies are up to date before building the project.
            // We may not have had the chance to update the release API assemblies, and the debug ones
            // may have been deleted by the user at some point after they were loaded by the Godot editor.
            string apiAssembliesUpdateError = Internal.UpdateApiAssembliesFromPrebuilt(config == "ExportRelease" ? "Release" : "Debug");

            if (!string.IsNullOrEmpty(apiAssembliesUpdateError))
            {
                ShowBuildErrorDialog("Failed to update the Godot API assemblies");
                return false;
            }

            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var buildTool = (BuildTool)editorSettings.GetSetting("mono/builds/build_tool");

            using (var pr = new EditorProgress("mono_project_debug_build", "Building project solution...", 1))
            {
                pr.Step("Building project solution", 0);

                var buildInfo = new BuildInfo(GodotSharpDirs.ProjectSlnPath, config);

                // Add Godot defines
                string constants = buildTool != BuildTool.MsBuildMono ? "GodotDefineConstants=\"" : "GodotDefineConstants=\\\"";

                foreach (var godotDefine in godotDefines)
                    constants += $"GODOT_{godotDefine.ToUpper().Replace("-", "_").Replace(" ", "_").Replace(";", "_")};";

                if (Internal.GodotIsRealTDouble())
                    constants += "GODOT_REAL_T_IS_DOUBLE;";

                constants += buildTool != BuildTool.MsBuildMono ? "\"" : "\\\"";

                buildInfo.CustomProperties.Add(constants);

                if (!Build(buildInfo))
                {
                    ShowBuildErrorDialog("Failed to build project solution");
                    return false;
                }
            }

            return true;
        }

        public static bool EditorBuildCallback()
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return true; // No solution to build

            string editorScriptsMetadataPath = Path.Combine(GodotSharpDirs.ResMetadataDir, "scripts_metadata.editor");
            string playerScriptsMetadataPath = Path.Combine(GodotSharpDirs.ResMetadataDir, "scripts_metadata.editor_player");

            CsProjOperations.GenerateScriptsMetadata(GodotSharpDirs.ProjectCsProjPath, editorScriptsMetadataPath);

            if (File.Exists(editorScriptsMetadataPath))
                File.Copy(editorScriptsMetadataPath, playerScriptsMetadataPath);

            var currentPlayRequest = GodotSharpEditor.Instance.GodotIdeManager.GodotIdeServer.CurrentPlayRequest;

            if (currentPlayRequest != null)
            {
                if (currentPlayRequest.Value.HasDebugger)
                {
                    // Set the environment variable that will tell the player to connect to the IDE debugger
                    // TODO: We should probably add a better way to do this
                    Environment.SetEnvironmentVariable("GODOT_MONO_DEBUGGER_AGENT",
                        "--debugger-agent=transport=dt_socket" +
                        $",address={currentPlayRequest.Value.DebuggerHost}:{currentPlayRequest.Value.DebuggerPort}" +
                        ",server=n");
                }

                return true; // Requested play from an external editor/IDE which already built the project
            }

            var godotDefines = new[]
            {
                Godot.OS.GetName(),
                Internal.GodotIs32Bits() ? "32" : "64"
            };

            return BuildProjectBlocking("Debug", godotDefines);
        }

        public static void Initialize()
        {
            // Build tool settings
            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var msbuild = BuildTool.MsBuildMono;
            if (OS.IsWindows)
                msbuild = RiderPathManager.IsExternalEditorSetToRider(editorSettings)
                        ? BuildTool.JetBrainsMsBuild
                        : BuildTool.MsBuildVs;

            EditorDef("mono/builds/build_tool", msbuild);

            editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
            {
                ["type"] = Godot.Variant.Type.Int,
                ["name"] = "mono/builds/build_tool",
                ["hint"] = Godot.PropertyHint.Enum,
                ["hint_string"] = OS.IsWindows ?
                    $"{PropNameMsbuildMono},{PropNameMsbuildVs},{PropNameMsbuildJetBrains}" :
                    $"{PropNameMsbuildMono}"
            });

            EditorDef("mono/builds/print_build_output", false);
        }
    }
}
