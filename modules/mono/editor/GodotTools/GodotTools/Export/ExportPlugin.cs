using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using GodotTools.Build;
using GodotTools.Core;
using GodotTools.Internals;
using static GodotTools.Internals.Globals;
using Directory = GodotTools.Utils.Directory;
using File = GodotTools.Utils.File;
using OS = GodotTools.Utils.OS;
using Path = System.IO.Path;

namespace GodotTools.Export
{
    public partial class ExportPlugin : EditorExportPlugin
    {
        public override string _GetName() => "C#";

        private List<string> _tempFolders = new List<string>();

        public override Godot.Collections.Array<Godot.Collections.Dictionary> _GetExportOptions(EditorExportPlatform platform)
        {
            return new Godot.Collections.Array<Godot.Collections.Dictionary>()
            {
                new Godot.Collections.Dictionary()
                {
                    {
                        "option", new Godot.Collections.Dictionary()
                        {
                            { "name", "dotnet/include_scripts_content" },
                            { "type", (int)Variant.Type.Bool }
                        }
                    },
                    { "default_value", false }
                }
            };
        }

        private string _maybeLastExportError;

        // With this method we can override how a file is exported in the PCK
        public override void _ExportFile(string path, string type, string[] features)
        {
            base._ExportFile(path, type, features);

            if (type != Internal.CSharpLanguageType)
                return;

            if (Path.GetExtension(path) != Internal.CSharpLanguageExtension)
                throw new ArgumentException(
                    $"Resource of type {Internal.CSharpLanguageType} has an invalid file extension: {path}",
                    nameof(path));

            // TODO What if the source file is not part of the game's C# project

            bool includeScriptsContent = (bool)GetOption("dotnet/include_scripts_content");

            if (!includeScriptsContent)
            {
                // We don't want to include the source code on exported games.

                // Sadly, Godot prints errors when adding an empty file (nothing goes wrong, it's just noise).
                // Because of this, we add a file which contains a line break.
                AddFile(path, System.Text.Encoding.UTF8.GetBytes("\n"), remap: false);

                // Tell the Godot exporter that we already took care of the file
                Skip();
            }
        }

        public override void _ExportBegin(string[] features, bool isDebug, string path, uint flags)
        {
            base._ExportBegin(features, isDebug, path, flags);

            try
            {
                _ExportBeginImpl(features, isDebug, path, flags);
            }
            catch (Exception e)
            {
                _maybeLastExportError = e.Message;

                // 'maybeLastExportError' cannot be null or empty if there was an error, so we
                // must consider the possibility of exceptions being thrown without a message.
                if (string.IsNullOrEmpty(_maybeLastExportError))
                    _maybeLastExportError = $"Exception thrown: {e.GetType().Name}";

                GD.PushError($"Failed to export project: {_maybeLastExportError}");
                Console.Error.WriteLine(e);
                // TODO: Do something on error once _ExportBegin supports failing.
            }
        }

        private void _ExportBeginImpl(string[] features, bool isDebug, string path, long flags)
        {
            _ = flags; // Unused

            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return;

            if (!DeterminePlatformFromFeatures(features, out string platform))
                throw new NotSupportedException("Target platform not supported.");

            if (!new[] { OS.Platforms.Windows, OS.Platforms.LinuxBSD, OS.Platforms.MacOS }
                    .Contains(platform))
            {
                throw new NotImplementedException("Target platform not yet implemented.");
            }

            string outputDir = new FileInfo(path).Directory?.FullName ??
                               throw new FileNotFoundException("Output base directory not found.");

            string buildConfig = isDebug ? "ExportDebug" : "ExportRelease";

            var archs = new List<string>();
            if (features.Contains("x86_64"))
            {
                archs.Add("x86_64");
            }
            else if (features.Contains("x86_32"))
            {
                archs.Add("x86_32");
            }
            else if (features.Contains("arm64"))
            {
                archs.Add("arm64");
            }
            else if (features.Contains("universal"))
            {
                if (platform == OS.Platforms.MacOS)
                {
                    archs.Add("x86_64");
                    archs.Add("arm64");
                }
            }

            foreach (var arch in archs)
            {
                string ridOS = DetermineRuntimeIdentifierOS(platform);
                string ridArch = DetermineRuntimeIdentifierArch(arch);
                string runtimeIdentifier = $"{ridOS}-{ridArch}";
                string projectDataDirName = $"{DetermineDataDirNameForProject()}_{arch}";
                if (platform == OS.Platforms.MacOS)
                {
                    projectDataDirName = Path.Combine("Contents", "Resources", projectDataDirName);
                }

                // Create temporary publish output directory

                string publishOutputTempDir = Path.Combine(Path.GetTempPath(), "godot-publish-dotnet",
                    $"{Process.GetCurrentProcess().Id}-{buildConfig}-{runtimeIdentifier}");

                _tempFolders.Add(publishOutputTempDir);

                if (!Directory.Exists(publishOutputTempDir))
                    Directory.CreateDirectory(publishOutputTempDir);

                // Execute dotnet publish

                if (!BuildManager.PublishProjectBlocking(buildConfig, platform,
                        runtimeIdentifier, publishOutputTempDir))
                {
                    throw new InvalidOperationException("Failed to build project.");
                }

                string soExt = ridOS switch
                {
                    OS.DotNetOS.Win or OS.DotNetOS.Win10 => "dll",
                    OS.DotNetOS.OSX or OS.DotNetOS.iOS => "dylib",
                    _ => "so"
                };

                if (!File.Exists(Path.Combine(publishOutputTempDir, $"{GodotSharpDirs.ProjectAssemblyName}.dll"))
                    // NativeAOT shared library output
                    && !File.Exists(Path.Combine(publishOutputTempDir, $"{GodotSharpDirs.ProjectAssemblyName}.{soExt}")))
                {
                    throw new NotSupportedException(
                        "Publish succeeded but project assembly not found in the output directory");
                }

                // Add to the exported project shared object list.

                foreach (string file in Directory.GetFiles(publishOutputTempDir, "*", SearchOption.AllDirectories))
                {
                    AddSharedObject(file, tags: null,
                        Path.Join(projectDataDirName,
                            Path.GetRelativePath(publishOutputTempDir, Path.GetDirectoryName(file))));
                }
            }
        }

        private string DetermineRuntimeIdentifierOS(string platform)
            => OS.DotNetOSPlatformMap[platform];

        private string DetermineRuntimeIdentifierArch(string arch)
        {
            return arch switch
            {
                "x86" => "x86",
                "x86_32" => "x86",
                "x64" => "x64",
                "x86_64" => "x64",
                "armeabi-v7a" => "arm",
                "arm64-v8a" => "arm64",
                "armv7" => "arm",
                "arm64" => "arm64",
                _ => throw new ArgumentOutOfRangeException(nameof(arch), arch, "Unexpected architecture")
            };
        }

        public override void _ExportEnd()
        {
            base._ExportEnd();

            string aotTempDir = Path.Combine(Path.GetTempPath(), $"godot-aot-{Process.GetCurrentProcess().Id}");

            if (Directory.Exists(aotTempDir))
                Directory.Delete(aotTempDir, recursive: true);

            foreach (string folder in _tempFolders)
            {
                Directory.Delete(folder, recursive: true);
            }
            _tempFolders.Clear();

            // TODO: The following is just a workaround until the export plugins can be made to abort with errors

            // We check for empty as well, because it's set to empty after hot-reloading
            if (!string.IsNullOrEmpty(_maybeLastExportError))
            {
                string lastExportError = _maybeLastExportError;
                _maybeLastExportError = null;

                GodotSharpEditor.Instance.ShowErrorDialog(lastExportError, "Failed to export C# project");
            }
        }

        private static bool DeterminePlatformFromFeatures(IEnumerable<string> features, out string platform)
        {
            foreach (var feature in features)
            {
                if (OS.PlatformFeatureMap.TryGetValue(feature, out platform))
                    return true;
            }

            platform = null;
            return false;
        }

        private static string DetermineDataDirNameForProject()
        {
            string appName = (string)ProjectSettings.GetSetting("application/config/name");
            string appNameSafe = appName.ToSafeDirName();
            return $"data_{appNameSafe}";
        }
    }
}
