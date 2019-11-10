using Godot;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using GodotTools.Core;
using GodotTools.Internals;
using Directory = GodotTools.Utils.Directory;
using File = GodotTools.Utils.File;
using OS = GodotTools.Utils.OS;
using Path = System.IO.Path;

namespace GodotTools.Export
{
    public class ExportPlugin : EditorExportPlugin
    {
        private void AddFile(string srcPath, string dstPath, bool remap = false)
        {
            AddFile(dstPath, File.ReadAllBytes(srcPath), remap);
        }

        public override void _ExportFile(string path, string type, string[] features)
        {
            base._ExportFile(path, type, features);

            if (type != Internal.CSharpLanguageType)
                return;

            if (Path.GetExtension(path) != $".{Internal.CSharpLanguageExtension}")
                throw new ArgumentException($"Resource of type {Internal.CSharpLanguageType} has an invalid file extension: {path}", nameof(path));

            // TODO What if the source file is not part of the game's C# project

            bool includeScriptsContent = (bool) ProjectSettings.GetSetting("mono/export/include_scripts_content");

            if (!includeScriptsContent)
            {
                // We don't want to include the source code on exported games
                AddFile(path, new byte[] { }, remap: false);
                Skip();
            }
        }

        public override void _ExportBegin(string[] features, bool isDebug, string path, int flags)
        {
            base._ExportBegin(features, isDebug, path, flags);

            try
            {
                _ExportBeginImpl(features, isDebug, path, flags);
                // TODO: Handle _ExportBeginImpl return value. Do something on error once _ExportBegin supports failing.
            }
            catch (Exception e)
            {
                GD.PushError($"Failed to export project: {e.Message}");
                Console.Error.WriteLine(e);
                // TODO: Do something on error once _ExportBegin supports failing.
            }
        }

        private bool _ExportBeginImpl(string[] features, bool isDebug, string path, int flags)
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return true;

            string platform = DeterminePlatformFromFeatures(features);

            if (platform == null)
                throw new NotSupportedException("Target platform not supported");

            // TODO Right now there is no way to stop the export process with an error

            string buildConfig = isDebug ? "Debug" : "Release";

            string scriptsMetadataPath = Path.Combine(GodotSharpDirs.ResMetadataDir, $"scripts_metadata.{(isDebug ? "debug" : "release")}");
            CsProjOperations.GenerateScriptsMetadata(GodotSharpDirs.ProjectCsProjPath, scriptsMetadataPath);

            AddFile(scriptsMetadataPath, scriptsMetadataPath);

            // Turn export features into defines
            var godotDefines = features;

            if (!BuildManager.BuildProjectBlocking(buildConfig, godotDefines))
            {
                GD.PushError("Failed to build project");
                return false;
            }

            // Add dependency assemblies

            var dependencies = new Godot.Collections.Dictionary<string, string>();

            var projectDllName = (string) ProjectSettings.GetSetting("application/config/name");
            if (projectDllName.Empty())
            {
                projectDllName = "UnnamedProject";
            }

            string projectDllSrcDir = Path.Combine(GodotSharpDirs.ResTempAssembliesBaseDir, buildConfig);
            string projectDllSrcPath = Path.Combine(projectDllSrcDir, $"{projectDllName}.dll");

            dependencies[projectDllName] = projectDllSrcPath;

            {
                string templatesDir = Internal.FullTemplatesDir;
                string platformBclDir = Path.Combine(templatesDir, $"{platform}-bcl", platform);

                string customBclDir = Directory.Exists(platformBclDir) ? platformBclDir : string.Empty;

                internal_GetExportedAssemblyDependencies(projectDllName, projectDllSrcPath, buildConfig, customBclDir, dependencies);
            }

            string apiConfig = isDebug ? "Debug" : "Release";
            string resAssembliesDir = Path.Combine(GodotSharpDirs.ResAssembliesBaseDir, apiConfig);

            foreach (var dependency in dependencies)
            {
                string dependSrcPath = dependency.Value;
                string dependDstPath = Path.Combine(resAssembliesDir, dependSrcPath.GetFile());
                AddFile(dependSrcPath, dependDstPath);
            }

            // Mono specific export template extras (data dir)
            ExportDataDirectory(features, platform, isDebug, path);

            return true;
        }

        private static void ExportDataDirectory(ICollection<string> features, string platform, bool debug, string path)
        {
            if (!PlatformHasTemplateDir(platform))
                return;

            string bits = features.Contains("64") ? "64" : "32";
            string target = debug ? "release_debug" : "release";

            string TemplateDirName() => $"data.mono.{platform}.{bits}.{target}";

            string templateDirPath = Path.Combine(Internal.FullTemplatesDir, TemplateDirName());

            if (!Directory.Exists(templateDirPath))
            {
                templateDirPath = null;
                
                if (debug)
                {
                    target = "debug"; // Support both 'release_debug' and 'debug' for the template data directory name
                    templateDirPath = Path.Combine(Internal.FullTemplatesDir, TemplateDirName());

                    if (!Directory.Exists(templateDirPath))
                        templateDirPath = null;
                }
            }

            if (templateDirPath == null)
                throw new FileNotFoundException("Data template directory not found");

            string outputDir = new FileInfo(path).Directory?.FullName ??
                               throw new FileNotFoundException("Base directory not found");

            string outputDataDir = Path.Combine(outputDir, DataDirName);

            if (Directory.Exists(outputDataDir))
                Directory.Delete(outputDataDir, recursive: true); // Clean first

            Directory.CreateDirectory(outputDataDir);

            foreach (string dir in Directory.GetDirectories(templateDirPath, "*", SearchOption.AllDirectories))
            {
                Directory.CreateDirectory(Path.Combine(outputDataDir, dir.Substring(templateDirPath.Length + 1)));
            }

            foreach (string file in Directory.GetFiles(templateDirPath, "*", SearchOption.AllDirectories))
            {
                File.Copy(file, Path.Combine(outputDataDir, file.Substring(templateDirPath.Length + 1)));
            }
        }

        private static bool PlatformHasTemplateDir(string platform)
        {
            // OSX export templates are contained in a zip, so we place our custom template inside it and let Godot do the rest.
            return !new[] {OS.Platforms.OSX, OS.Platforms.Android, OS.Platforms.HTML5}.Contains(platform);
        }

        private static string DeterminePlatformFromFeatures(IEnumerable<string> features)
        {
            foreach (var feature in features)
            {
                if (OS.PlatformNameMap.TryGetValue(feature, out string platform))
                    return platform;
            }

            return null;
        }

        private static string DataDirName
        {
            get
            {
                var appName = (string) ProjectSettings.GetSetting("application/config/name");
                string appNameSafe = appName.ToSafeDirName(allowDirSeparator: false);
                return $"data_{appNameSafe}";
            }
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_GetExportedAssemblyDependencies(string projectDllName, string projectDllSrcPath,
            string buildConfig, string customBclDir, Godot.Collections.Dictionary<string, string> dependencies);
    }
}
