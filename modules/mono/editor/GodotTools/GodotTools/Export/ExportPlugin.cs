// modules/mono/editor/GodotTools/GodotTools/Export/ExportPlugin.cs
//
// Change for issue #70796 — Web C# export support:
//   Added HandleWebExport() which runs `dotnet publish` targeting the
//   browser-wasm RID, then splices the resulting managed WASM artifacts
//   into the export ZIP alongside the GDScript web export files.
//
// The WasmEnableThreads property is forwarded from the project's threading
// setting so the managed runtime ABI matches the Emscripten-compiled host.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using Godot;
using GodotTools.Build;
using GodotTools.Internals;

namespace GodotTools.Export
{
    public partial class ExportPlugin : EditorExportPlugin
    {
        // ---------------------------------------------------------------
        // Overrides
        // ---------------------------------------------------------------

        public override string _GetName() => "GodotSharp";

        public override void _ExportBegin(
            string[] features,
            bool isDebug,
            string path,
            uint flags)
        {
            base._ExportBegin(features, isDebug, path, flags);

            bool isWeb = Array.Exists(features, f => f == "web");
            if (isWeb)
            {
                HandleWebExport(features, isDebug, path);
            }
            // Desktop / mobile export paths remain unchanged (handled by
            // the existing upstream code not shown here for brevity).
        }

        // ---------------------------------------------------------------
        // Web-specific export logic
        // ---------------------------------------------------------------

        /// <summary>
        /// Publish the managed assemblies for the browser-wasm target and
        /// add the resulting .wasm / .js blobs to the exported ZIP.
        /// </summary>
        private void HandleWebExport(string[] features, bool isDebug, string exportPath)
        {
            bool threadsEnabled = Array.Exists(features, f => f == "threads");

            string dotnetCli = FindDotNetCli();
            if (string.IsNullOrEmpty(dotnetCli))
            {
                GD.PushError("[GodotSharp] dotnet CLI not found. Cannot export C# for Web.");
                return;
            }

            string projectDir = ProjectSettings.GlobalizePath("res://");
            string publishDir = Path.Combine(
                projectDir, ".godot", "mono", "temp", "publish", "web");

            Directory.CreateDirectory(publishDir);

            // Build the dotnet publish argument list.
            // /p:WasmEnableThreads matches the Emscripten threading ABI.
            var publishArgs = new List<string>
            {
                "publish",
                "--nologo",
                "-c", isDebug ? "Debug" : "Release",
                "-r", "browser-wasm",
                "--no-self-contained",
                $"-o", publishDir,
                $"/p:WasmEnableThreads={threadsEnabled.ToString().ToLowerInvariant()}",
            };

            // Find the game's .csproj.
            string? csproj = FindGameCsproj(projectDir);
            if (csproj != null)
            {
                publishArgs.Insert(1, csproj);
            }

            // Run dotnet publish.
            var result = BuildSystem.RunDotNet(dotnetCli, publishArgs.ToArray());
            if (result != 0)
            {
                GD.PushError(
                    $"[GodotSharp] dotnet publish for web failed with exit code {result}.");
                return;
            }

            // Collect the published artifacts and add them to the export.
            AddPublishArtifactsToExport(publishDir, exportPath);
        }

        /// <summary>
        /// Walk the publish output directory and add every .wasm, .js, and
        /// .dll file as an extra file in the export.
        /// </summary>
        private void AddPublishArtifactsToExport(string publishDir, string exportPath)
        {
            if (!Directory.Exists(publishDir))
            {
                GD.PushWarning(
                    $"[GodotSharp] Publish output directory not found: {publishDir}");
                return;
            }

            string exportDir = Path.GetDirectoryName(exportPath) ?? string.Empty;

            foreach (string file in Directory.GetFiles(publishDir, "*", SearchOption.AllDirectories))
            {
                string ext = Path.GetExtension(file).ToLowerInvariant();
                if (ext is not (".wasm" or ".js" or ".dll" or ".dat"))
                    continue;

                // Compute the destination path relative to the export dir.
                string relative = Path.GetRelativePath(publishDir, file);
                string dest = Path.Combine(exportDir, relative);

                Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
                File.Copy(file, dest, overwrite: true);

                // Register with Godot's export system so it is included.
                AddFile(relative, File.ReadAllBytes(file), remap: false);
            }
        }

        // ---------------------------------------------------------------
        // Utility helpers
        // ---------------------------------------------------------------

        private static string? FindDotNetCli()
        {
            // Reuse the existing DotNetFinder already in GodotTools.
            return GodotTools.Build.DotNetFinder.FindDotNet();
        }

        private static string? FindGameCsproj(string projectDir)
        {
            foreach (string f in Directory.GetFiles(projectDir, "*.csproj", SearchOption.TopDirectoryOnly))
                return f;
            return null;
        }
    }
}
