using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using GodotTools.Build;
using GodotTools.Internals;
using GodotTools.Utils;
using static GodotTools.Internals.Globals;
using Error = Godot.Error;
using File = GodotTools.Utils.File;
using Directory = GodotTools.Utils.Directory;

namespace GodotTools
{
    public static class GodotSharpBuilds
    {
        private static readonly List<MonoBuildInfo> BuildsInProgress = new List<MonoBuildInfo>();

        public const string PropNameMsbuildMono = "MSBuild (Mono)";
        public const string PropNameMsbuildVs = "MSBuild (VS Build Tools)";

        public const string MsBuildIssuesFileName = "msbuild_issues.csv";
        public const string MsBuildLogFileName = "msbuild_log.txt";

        public enum BuildTool
        {
            MsBuildMono,
            MsBuildVs
        }

        private static void RemoveOldIssuesFile(MonoBuildInfo buildInfo)
        {
            var issuesFile = GetIssuesFilePath(buildInfo);

            if (!File.Exists(issuesFile))
                return;

            File.Delete(issuesFile);
        }

        private static string _ApiFolderName(ApiAssemblyType apiType)
        {
            ulong apiHash = apiType == ApiAssemblyType.Core ?
                Internal.GetCoreApiHash() :
                Internal.GetEditorApiHash();
            return $"{apiHash}_{BindingsGenerator.Version}_{BindingsGenerator.CsGlueVersion}";
        }

        private static void ShowBuildErrorDialog(string message)
        {
            GodotSharpEditor.Instance.ShowErrorDialog(message, "Build error");
            GodotSharpEditor.Instance.MonoBottomPanel.ShowBuildTab();
        }

        public static void RestartBuild(MonoBuildTab buildTab) => throw new NotImplementedException();
        public static void StopBuild(MonoBuildTab buildTab) => throw new NotImplementedException();

        private static string GetLogFilePath(MonoBuildInfo buildInfo)
        {
            return Path.Combine(buildInfo.LogsDirPath, MsBuildLogFileName);
        }

        private static string GetIssuesFilePath(MonoBuildInfo buildInfo)
        {
            return Path.Combine(buildInfo.LogsDirPath, MsBuildIssuesFileName);
        }

        private static void PrintVerbose(string text)
        {
            if (Godot.OS.IsStdoutVerbose())
                Godot.GD.Print(text);
        }

        public static bool Build(MonoBuildInfo buildInfo)
        {
            if (BuildsInProgress.Contains(buildInfo))
                throw new InvalidOperationException("A build is already in progress");

            BuildsInProgress.Add(buildInfo);

            try
            {
                MonoBuildTab buildTab = GodotSharpEditor.Instance.MonoBottomPanel.GetBuildTabFor(buildInfo);
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

                    buildTab.OnBuildExit(exitCode == 0 ? MonoBuildTab.BuildResults.Success : MonoBuildTab.BuildResults.Error);

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

        public static async Task<bool> BuildAsync(MonoBuildInfo buildInfo)
        {
            if (BuildsInProgress.Contains(buildInfo))
                throw new InvalidOperationException("A build is already in progress");

            BuildsInProgress.Add(buildInfo);

            try
            {
                MonoBuildTab buildTab = GodotSharpEditor.Instance.MonoBottomPanel.GetBuildTabFor(buildInfo);

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

                    buildTab.OnBuildExit(exitCode == 0 ? MonoBuildTab.BuildResults.Success : MonoBuildTab.BuildResults.Error);

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

        public static bool BuildApiSolution(string apiSlnDir, string config)
        {
            string apiSlnFile = Path.Combine(apiSlnDir, $"{ApiAssemblyNames.SolutionName}.sln");

            string coreApiAssemblyDir = Path.Combine(apiSlnDir, ApiAssemblyNames.Core, "bin", config);
            string coreApiAssemblyFile = Path.Combine(coreApiAssemblyDir, $"{ApiAssemblyNames.Core}.dll");

            string editorApiAssemblyDir = Path.Combine(apiSlnDir, ApiAssemblyNames.Editor, "bin", config);
            string editorApiAssemblyFile = Path.Combine(editorApiAssemblyDir, $"{ApiAssemblyNames.Editor}.dll");

            if (File.Exists(coreApiAssemblyFile) && File.Exists(editorApiAssemblyFile))
                return true; // The assemblies are in the output folder; assume the solution is already built

            var apiBuildInfo = new MonoBuildInfo(apiSlnFile, config);

            // TODO Replace this global NoWarn with '#pragma warning' directives on generated files,
            // once we start to actively document manually maintained C# classes
            apiBuildInfo.CustomProperties.Add("NoWarn=1591"); // Ignore missing documentation warnings

            if (Build(apiBuildInfo))
                return true;

            ShowBuildErrorDialog($"Failed to build {ApiAssemblyNames.SolutionName} solution.");
            return false;
        }

        public static bool BuildProjectBlocking(string config, IEnumerable<string> godotDefines)
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return true; // No solution to build

            // Make sure to update the API assemblies if they happen to be missing. Just in
            // case the user decided to delete them at some point after they were loaded.
            Internal.UpdateApiAssembliesFromPrebuilt();

            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var buildTool = (BuildTool)editorSettings.GetSetting("mono/builds/build_tool");

            using (var pr = new EditorProgress("mono_project_debug_build", "Building project solution...", 1))
            {
                pr.Step("Building project solution", 0);

                var buildInfo = new MonoBuildInfo(GodotSharpDirs.ProjectSlnPath, config);

                // Add Godot defines
                string constants = buildTool == BuildTool.MsBuildVs ? "GodotDefineConstants=\"" : "GodotDefineConstants=\\\"";

                foreach (var godotDefine in godotDefines)
                    constants += $"GODOT_{godotDefine.ToUpper().Replace("-", "_").Replace(" ", "_").Replace(";", "_")};";

                if (Internal.GodotIsRealTDouble())
                    constants += "GODOT_REAL_T_IS_DOUBLE;";

                constants += buildTool == BuildTool.MsBuildVs ? "\"" : "\\\"";

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

            CSharpProject.GenerateScriptsMetadata(GodotSharpDirs.ProjectCsProjPath, editorScriptsMetadataPath);

            if (File.Exists(editorScriptsMetadataPath))
                File.Copy(editorScriptsMetadataPath, playerScriptsMetadataPath);

            var godotDefines = new[]
            {
                Godot.OS.GetName(),
                Internal.GodotIs32Bits() ? "32" : "64"
            };

            return BuildProjectBlocking("Tools", godotDefines);
        }

        public static void Initialize()
        {
            // Build tool settings

            EditorDef("mono/builds/build_tool", OS.IsWindows() ? BuildTool.MsBuildVs : BuildTool.MsBuildMono);

            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();

            editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
            {
                ["type"] = Godot.Variant.Type.Int,
                ["name"] = "mono/builds/build_tool",
                ["hint"] = Godot.PropertyHint.Enum,
                ["hint_string"] = OS.IsWindows() ?
                    $"{PropNameMsbuildMono},{PropNameMsbuildVs}" :
                    $"{PropNameMsbuildMono}"
            });

            EditorDef("mono/builds/print_build_output", false);
        }
    }
}
