using Godot;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using GodotTools.Core;
using GodotTools.Internals;
using static GodotTools.Internals.Globals;
using Directory = GodotTools.Utils.Directory;
using File = GodotTools.Utils.File;
using OS = GodotTools.Utils.OS;
using Path = System.IO.Path;

namespace GodotTools.Export
{
    public class ExportPlugin : EditorExportPlugin
    {
        public void RegisterExportSettings()
        {
            // TODO: These would be better as export preset options, but that doesn't seem to be supported yet

            GlobalDef("mono/export/include_scripts_content", false);
            GlobalDef("mono/export/export_assemblies_inside_pck", true);

            GlobalDef("mono/export/aot/enabled", false);
            GlobalDef("mono/export/aot/full_aot", false);

            // --aot or --aot=opt1,opt2 (use 'mono --aot=help AuxAssembly.dll' to list AOT options)
            GlobalDef("mono/export/aot/extra_aot_options", new string[] { });
            // --optimize/-O=opt1,opt2 (use 'mono --list-opt'' to list optimize options)
            GlobalDef("mono/export/aot/extra_optimizer_options", new string[] { });

            GlobalDef("mono/export/aot/android_toolchain_path", "");
        }

        private string maybeLastExportError;

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

            bool includeScriptsContent = (bool)ProjectSettings.GetSetting("mono/export/include_scripts_content");

            if (!includeScriptsContent)
            {
                // We don't want to include the source code on exported games.

                // Sadly, Godot prints errors when adding an empty file (nothing goes wrong, it's just noise).
                // Because of this, we add a file which contains a line break.
                AddFile(path, System.Text.Encoding.UTF8.GetBytes("\n"), remap: false);
                Skip();
            }
        }

        public override void _ExportBegin(string[] features, bool isDebug, string path, int flags)
        {
            base._ExportBegin(features, isDebug, path, flags);

            try
            {
                _ExportBeginImpl(features, isDebug, path, flags);
            }
            catch (Exception e)
            {
                maybeLastExportError = e.Message;
                GD.PushError($"Failed to export project: {e.Message}");
                Console.Error.WriteLine(e);
                // TODO: Do something on error once _ExportBegin supports failing.
            }
        }

        private void _ExportBeginImpl(string[] features, bool isDebug, string path, int flags)
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return;

            string platform = DeterminePlatformFromFeatures(features);

            if (platform == null)
                throw new NotSupportedException("Target platform not supported");

            string outputDir = new FileInfo(path).Directory?.FullName ??
                               throw new FileNotFoundException("Base directory not found");

            string buildConfig = isDebug ? "Debug" : "Release";

            string scriptsMetadataPath = Path.Combine(GodotSharpDirs.ResMetadataDir, $"scripts_metadata.{(isDebug ? "debug" : "release")}");
            CsProjOperations.GenerateScriptsMetadata(GodotSharpDirs.ProjectCsProjPath, scriptsMetadataPath);

            AddFile(scriptsMetadataPath, scriptsMetadataPath);

            // Turn export features into defines
            var godotDefines = features;

            if (!BuildManager.BuildProjectBlocking(buildConfig, godotDefines))
                throw new Exception("Failed to build project");

            // Add dependency assemblies

            var dependencies = new Godot.Collections.Dictionary<string, string>();

            var projectDllName = (string)ProjectSettings.GetSetting("application/config/name");
            if (projectDllName.Empty())
            {
                projectDllName = "UnnamedProject";
            }

            string projectDllSrcDir = Path.Combine(GodotSharpDirs.ResTempAssembliesBaseDir, buildConfig);
            string projectDllSrcPath = Path.Combine(projectDllSrcDir, $"{projectDllName}.dll");

            dependencies[projectDllName] = projectDllSrcPath;

            if (platform == OS.Platforms.Android)
            {
                string godotAndroidExtProfileDir = GetBclProfileDir("godot_android_ext");
                string monoAndroidAssemblyPath = Path.Combine(godotAndroidExtProfileDir, "Mono.Android.dll");

                if (!File.Exists(monoAndroidAssemblyPath))
                    throw new FileNotFoundException("Assembly not found: 'Mono.Android'", monoAndroidAssemblyPath);

                dependencies["Mono.Android"] = monoAndroidAssemblyPath;
            }

            var initialDependencies = dependencies.Duplicate();
            internal_GetExportedAssemblyDependencies(initialDependencies, buildConfig, DeterminePlatformBclDir(platform), dependencies);

            string outputDataDir = null;

            if (PlatformHasTemplateDir(platform))
                outputDataDir = ExportDataDirectory(features, platform, isDebug, outputDir);

            string apiConfig = isDebug ? "Debug" : "Release";
            string resAssembliesDir = Path.Combine(GodotSharpDirs.ResAssembliesBaseDir, apiConfig);

            bool assembliesInsidePck = (bool)ProjectSettings.GetSetting("mono/export/export_assemblies_inside_pck") || outputDataDir == null;

            if (!assembliesInsidePck)
            {
                string outputDataGameAssembliesDir = Path.Combine(outputDataDir, "Assemblies");
                if (!Directory.Exists(outputDataGameAssembliesDir))
                    Directory.CreateDirectory(outputDataGameAssembliesDir);
            }

            foreach (var dependency in dependencies)
            {
                string dependSrcPath = dependency.Value;

                if (assembliesInsidePck)
                {
                    string dependDstPath = Path.Combine(resAssembliesDir, dependSrcPath.GetFile());
                    AddFile(dependSrcPath, dependDstPath);
                }
                else
                {
                    string dependDstPath = Path.Combine(outputDataDir, "Assemblies", dependSrcPath.GetFile());
                    File.Copy(dependSrcPath, dependDstPath);
                }
            }

            // AOT

            if ((bool)ProjectSettings.GetSetting("mono/export/aot/enabled"))
            {
                AotCompileDependencies(features, platform, isDebug, outputDir, outputDataDir, dependencies);
            }
        }

        public override void _ExportEnd()
        {
            base._ExportEnd();

            string aotTempDir = Path.Combine(Path.GetTempPath(), $"godot-aot-{Process.GetCurrentProcess().Id}");

            if (Directory.Exists(aotTempDir))
                Directory.Delete(aotTempDir, recursive: true);

            // TODO: Just a workaround until the export plugins can be made to abort with errors
            if (!string.IsNullOrEmpty(maybeLastExportError)) // Check empty as well, because it's set to empty after hot-reloading
            {
                string lastExportError = maybeLastExportError;
                maybeLastExportError = null;

                GodotSharpEditor.Instance.ShowErrorDialog(lastExportError, "Failed to export C# project");
            }
        }

        private static string ExportDataDirectory(string[] features, string platform, bool isDebug, string outputDir)
        {
            string target = isDebug ? "release_debug" : "release";

            // NOTE: Bits is ok for now as all platforms with a data directory have it, but that may change in the future.
            string bits = features.Contains("64") ? "64" : "32";

            string TemplateDirName() => $"data.mono.{platform}.{bits}.{target}";

            string templateDirPath = Path.Combine(Internal.FullTemplatesDir, TemplateDirName());
            bool validTemplatePathFound = true;

            if (!Directory.Exists(templateDirPath))
            {
                validTemplatePathFound = false;

                if (isDebug)
                {
                    target = "debug"; // Support both 'release_debug' and 'debug' for the template data directory name
                    templateDirPath = Path.Combine(Internal.FullTemplatesDir, TemplateDirName());
                    validTemplatePathFound = true;

                    if (!Directory.Exists(templateDirPath))
                        validTemplatePathFound = false;
                }
            }

            if (!validTemplatePathFound)
                throw new FileNotFoundException("Data template directory not found", templateDirPath);

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

            return outputDataDir;
        }

        private void AotCompileDependencies(string[] features, string platform, bool isDebug, string outputDir, string outputDataDir, IDictionary<string, string> dependencies)
        {
            // TODO: WASM

            string bclDir = DeterminePlatformBclDir(platform) ?? typeof(object).Assembly.Location.GetBaseDir();

            string aotTempDir = Path.Combine(Path.GetTempPath(), $"godot-aot-{Process.GetCurrentProcess().Id}");

            if (!Directory.Exists(aotTempDir))
                Directory.CreateDirectory(aotTempDir);

            var assemblies = new Dictionary<string, string>();

            foreach (var dependency in dependencies)
            {
                string assemblyName = dependency.Key;
                string assemblyPath = dependency.Value;

                string assemblyPathInBcl = Path.Combine(bclDir, assemblyName + ".dll");

                if (File.Exists(assemblyPathInBcl))
                {
                    // Don't create teporaries for assemblies from the BCL
                    assemblies.Add(assemblyName, assemblyPathInBcl);
                }
                else
                {
                    string tempAssemblyPath = Path.Combine(aotTempDir, assemblyName + ".dll");
                    File.Copy(assemblyPath, tempAssemblyPath);
                    assemblies.Add(assemblyName, tempAssemblyPath);
                }
            }

            foreach (var assembly in assemblies)
            {
                string assemblyName = assembly.Key;
                string assemblyPath = assembly.Value;

                string sharedLibExtension = platform == OS.Platforms.Windows ? ".dll" :
                    platform == OS.Platforms.OSX ? ".dylib" :
                    platform == OS.Platforms.HTML5 ? ".wasm" :
                    ".so";

                string outputFileName = assemblyName + ".dll" + sharedLibExtension;

                if (platform == OS.Platforms.Android)
                {
                    // Not sure if the 'lib' prefix is an Android thing or just Godot being picky,
                    // but we use '-aot-' as well just in case to avoid conflicts with other libs.
                    outputFileName = "lib-aot-" + outputFileName;
                }

                string outputFilePath = null;
                string tempOutputFilePath;

                switch (platform)
                {
                    case OS.Platforms.OSX:
                        tempOutputFilePath = Path.Combine(aotTempDir, outputFileName);
                        break;
                    case OS.Platforms.Android:
                        tempOutputFilePath = Path.Combine(aotTempDir, "%%ANDROID_ABI%%", outputFileName);
                        break;
                    case OS.Platforms.HTML5:
                        tempOutputFilePath = Path.Combine(aotTempDir, outputFileName);
                        outputFilePath = Path.Combine(outputDir, outputFileName);
                        break;
                    default:
                        tempOutputFilePath = Path.Combine(aotTempDir, outputFileName);
                        outputFilePath = Path.Combine(outputDataDir, "Mono", platform == OS.Platforms.Windows ? "bin" : "lib", outputFileName);
                        break;
                }

                var data = new Dictionary<string, string>();
                var enabledAndroidAbis = platform == OS.Platforms.Android ? GetEnabledAndroidAbis(features).ToArray() : null;

                if (platform == OS.Platforms.Android)
                {
                    Debug.Assert(enabledAndroidAbis != null);

                    foreach (var abi in enabledAndroidAbis)
                    {
                        data["abi"] = abi;
                        var outputFilePathForThisAbi = tempOutputFilePath.Replace("%%ANDROID_ABI%%", abi);

                        AotCompileAssembly(platform, isDebug, data, assemblyPath, outputFilePathForThisAbi);

                        AddSharedObject(outputFilePathForThisAbi, tags: new[] { abi });
                    }
                }
                else
                {
                    string bits = features.Contains("64") ? "64" : features.Contains("64") ? "32" : null;

                    if (bits != null)
                        data["bits"] = bits;

                    AotCompileAssembly(platform, isDebug, data, assemblyPath, tempOutputFilePath);

                    if (platform == OS.Platforms.OSX)
                    {
                        AddSharedObject(tempOutputFilePath, tags: null);
                    }
                    else
                    {
                        Debug.Assert(outputFilePath != null);
                        File.Copy(tempOutputFilePath, outputFilePath);
                    }
                }
            }
        }

        private static void AotCompileAssembly(string platform, bool isDebug, Dictionary<string, string> data, string assemblyPath, string outputFilePath)
        {
            // Make sure the output directory exists
            Directory.CreateDirectory(outputFilePath.GetBaseDir());

            string exeExt = OS.IsWindows ? ".exe" : string.Empty;

            string monoCrossDirName = DetermineMonoCrossDirName(platform, data);
            string monoCrossRoot = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "aot-compilers", monoCrossDirName);
            string monoCrossBin = Path.Combine(monoCrossRoot, "bin");

            string toolPrefix = DetermineToolPrefix(monoCrossBin);
            string monoExeName = System.IO.File.Exists(Path.Combine(monoCrossBin, $"{toolPrefix}mono{exeExt}")) ? "mono" : "mono-sgen";

            string compilerCommand = Path.Combine(monoCrossBin, $"{toolPrefix}{monoExeName}{exeExt}");

            bool fullAot = (bool)ProjectSettings.GetSetting("mono/export/aot/full_aot");

            string EscapeOption(string option) => option.Contains(',') ? $"\"{option}\"" : option;
            string OptionsToString(IEnumerable<string> options) => string.Join(",", options.Select(EscapeOption));

            var aotOptions = new List<string>();
            var optimizerOptions = new List<string>();

            if (fullAot)
                aotOptions.Add("full");

            aotOptions.Add(isDebug ? "soft-debug" : "nodebug");

            if (platform == OS.Platforms.Android)
            {
                string abi = data["abi"];

                string androidToolchain = (string)ProjectSettings.GetSetting("mono/export/aot/android_toolchain_path");

                if (string.IsNullOrEmpty(androidToolchain))
                {
                    androidToolchain = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "android-toolchains", $"{abi}"); // TODO: $"{abi}-{apiLevel}{(clang?"clang":"")}"

                    if (!Directory.Exists(androidToolchain))
                        throw new FileNotFoundException("Missing android toolchain. Specify one in the AOT export settings.");
                }
                else if (!Directory.Exists(androidToolchain))
                {
                    throw new FileNotFoundException("Android toolchain not found: " + androidToolchain);
                }

                var androidToolPrefixes = new Dictionary<string, string>
                {
                    ["armeabi-v7a"] = "arm-linux-androideabi-",
                    ["arm64-v8a"] = "aarch64-linux-android-",
                    ["x86"] = "i686-linux-android-",
                    ["x86_64"] = "x86_64-linux-android-"
                };

                aotOptions.Add("tool-prefix=" + Path.Combine(androidToolchain, "bin", androidToolPrefixes[abi]));

                string triple = GetAndroidTriple(abi);
                aotOptions.Add($"mtriple={triple}");
            }

            aotOptions.Add($"outfile={outputFilePath}");

            var extraAotOptions = (string[])ProjectSettings.GetSetting("mono/export/aot/extra_aot_options");
            var extraOptimizerOptions = (string[])ProjectSettings.GetSetting("mono/export/aot/extra_optimizer_options");

            if (extraAotOptions.Length > 0)
                aotOptions.AddRange(extraAotOptions);

            if (extraOptimizerOptions.Length > 0)
                optimizerOptions.AddRange(extraOptimizerOptions);

            var compilerArgs = new List<string>();

            if (isDebug)
                compilerArgs.Add("--debug"); // Required for --aot=soft-debug

            compilerArgs.Add(aotOptions.Count > 0 ? $"--aot={OptionsToString(aotOptions)}" : "--aot");

            if (optimizerOptions.Count > 0)
                compilerArgs.Add($"-O={OptionsToString(optimizerOptions)}");

            compilerArgs.Add(ProjectSettings.GlobalizePath(assemblyPath));

            // TODO: Once we move to .NET Standard 2.1 we can use ProcessStartInfo.ArgumentList instead
            string CmdLineArgsToString(IEnumerable<string> args)
            {
                // Not perfect, but as long as we are careful...
                return string.Join(" ", args.Select(arg => arg.Contains(" ") ? $@"""{arg}""" : arg));
            }

            using (var process = new Process())
            {
                process.StartInfo = new ProcessStartInfo(compilerCommand, CmdLineArgsToString(compilerArgs))
                {
                    UseShellExecute = false
                };

                string platformBclDir = DeterminePlatformBclDir(platform);
                process.StartInfo.EnvironmentVariables.Add("MONO_PATH", string.IsNullOrEmpty(platformBclDir) ?
                    typeof(object).Assembly.Location.GetBaseDir() :
                    platformBclDir);

                Console.WriteLine($"Running: \"{process.StartInfo.FileName}\" {process.StartInfo.Arguments}");

                if (!process.Start())
                    throw new Exception("Failed to start process for Mono AOT compiler");

                process.WaitForExit();

                if (process.ExitCode != 0)
                    throw new Exception($"Mono AOT compiler exited with error code: {process.ExitCode}");

                if (!System.IO.File.Exists(outputFilePath))
                    throw new Exception("Mono AOT compiler finished successfully but the output file is missing");
            }
        }

        private static string DetermineMonoCrossDirName(string platform, IReadOnlyDictionary<string, string> data)
        {
            switch (platform)
            {
                case OS.Platforms.Windows:
                case OS.Platforms.UWP:
                {
                    string arch = data["bits"] == "64" ? "x86_64" : "i686";
                    return $"windows-{arch}";
                }
                case OS.Platforms.OSX:
                {
                    string arch = "x86_64";
                    return $"{platform}-{arch}";
                }
                case OS.Platforms.X11:
                case OS.Platforms.Server:
                {
                    string arch = data["bits"] == "64" ? "x86_64" : "i686";
                    return $"linux-{arch}";
                }
                case OS.Platforms.Haiku:
                {
                    string arch = data["bits"] == "64" ? "x86_64" : "i686";
                    return $"{platform}-{arch}";
                }
                case OS.Platforms.Android:
                {
                    string abi = data["abi"];
                    return $"{platform}-{abi}";
                }
                case OS.Platforms.HTML5:
                    return "wasm-wasm32";
                default:
                    throw new NotSupportedException();
            }
        }

        private static string DetermineToolPrefix(string monoCrossBin)
        {
            string exeExt = OS.IsWindows ? ".exe" : string.Empty;

            if (System.IO.File.Exists(Path.Combine(monoCrossBin, $"mono{exeExt}")))
                return string.Empty;

            if (System.IO.File.Exists(Path.Combine(monoCrossBin, $"mono-sgen{exeExt}" + exeExt)))
                return string.Empty;

            var files = new DirectoryInfo(monoCrossBin).GetFiles($"*mono{exeExt}" + exeExt, SearchOption.TopDirectoryOnly);
            if (files.Length > 0)
            {
                string fileName = files[0].Name;
                return fileName.Substring(0, fileName.Length - $"mono{exeExt}".Length);
            }

            files = new DirectoryInfo(monoCrossBin).GetFiles($"*mono-sgen{exeExt}" + exeExt, SearchOption.TopDirectoryOnly);
            if (files.Length > 0)
            {
                string fileName = files[0].Name;
                return fileName.Substring(0, fileName.Length - $"mono-sgen{exeExt}".Length);
            }

            throw new FileNotFoundException($"Cannot find the mono runtime executable in {monoCrossBin}");
        }

        private static IEnumerable<string> GetEnabledAndroidAbis(string[] features)
        {
            var androidAbis = new[]
            {
                "armeabi-v7a",
                "arm64-v8a",
                "x86",
                "x86_64"
            };

            return androidAbis.Where(features.Contains);
        }

        private static string GetAndroidTriple(string abi)
        {
            var abiArchs = new Dictionary<string, string>
            {
                ["armeabi-v7a"] = "armv7",
                ["arm64-v8a"] = "aarch64-v8a",
                ["x86"] = "i686",
                ["x86_64"] = "x86_64"
            };

            string arch = abiArchs[abi];

            return $"{arch}-linux-android";
        }

        private static bool PlatformHasTemplateDir(string platform)
        {
            // OSX export templates are contained in a zip, so we place our custom template inside it and let Godot do the rest.
            return !new[] { OS.Platforms.OSX, OS.Platforms.Android, OS.Platforms.HTML5 }.Contains(platform);
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

        private static string GetBclProfileDir(string profile)
        {
            string templatesDir = Internal.FullTemplatesDir;
            return Path.Combine(templatesDir, "bcl", profile);
        }

        private static string DeterminePlatformBclDir(string platform)
        {
            string templatesDir = Internal.FullTemplatesDir;
            string platformBclDir = Path.Combine(templatesDir, "bcl", platform);

            if (!File.Exists(Path.Combine(platformBclDir, "mscorlib.dll")))
            {
                string profile = DeterminePlatformBclProfile(platform);
                platformBclDir = Path.Combine(templatesDir, "bcl", profile);

                if (!File.Exists(Path.Combine(platformBclDir, "mscorlib.dll")))
                {
                    if (PlatformRequiresCustomBcl(platform))
                        throw new FileNotFoundException($"Missing BCL (Base Class Library) for platform: {platform}");

                    platformBclDir = null; // Use the one we're running on
                }
            }

            return platformBclDir;
        }

        /// <summary>
        /// Determines whether the BCL bundled with the Godot editor can be used for the target platform,
        /// or if it requires a custom BCL that must be distributed with the export templates.
        /// </summary>
        private static bool PlatformRequiresCustomBcl(string platform)
        {
            if (new[] { OS.Platforms.Android, OS.Platforms.HTML5 }.Contains(platform))
                return true;

            // The 'net_4_x' BCL is not compatible between Windows and the other platforms.
            // We use the names 'net_4_x_win' and 'net_4_x' to differentiate between the two.

            bool isWinOrUwp = new[]
            {
                OS.Platforms.Windows,
                OS.Platforms.UWP
            }.Contains(platform);

            return OS.IsWindows ? !isWinOrUwp : isWinOrUwp;
        }

        private static string DeterminePlatformBclProfile(string platform)
        {
            switch (platform)
            {
                case OS.Platforms.Windows:
                case OS.Platforms.UWP:
                    return "net_4_x_win";
                case OS.Platforms.OSX:
                case OS.Platforms.X11:
                case OS.Platforms.Server:
                case OS.Platforms.Haiku:
                    return "net_4_x";
                case OS.Platforms.Android:
                    return "monodroid";
                case OS.Platforms.HTML5:
                    return "wasm";
                default:
                    throw new NotSupportedException();
            }
        }

        private static string DataDirName
        {
            get
            {
                var appName = (string)ProjectSettings.GetSetting("application/config/name");
                string appNameSafe = appName.ToSafeDirName(allowDirSeparator: false);
                return $"data_{appNameSafe}";
            }
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_GetExportedAssemblyDependencies(Godot.Collections.Dictionary<string, string> initialDependencies,
            string buildConfig, string customBclDir, Godot.Collections.Dictionary<string, string> dependencies);
    }
}
