using System;
using System.Collections.Generic;
using System.Linq;
using Godot;
using GodotTools.Export;

namespace GodotTools.Expor
{
    public class ExportPlugin : EditorExportPlugin
    {
        private PublishConfig publishConfig;

        public ExportPlugin(PublishConfig publishConfig)
        {
            this.publishConfig = publishConfig;
        }

        private void _ExportBeginImpl(string[] features, bool isDebug, string path, long flags)
        {
            _ = flags; // Unused.

            if (!ProjectContainsDotNet())
                return;

            string osName = GetExportPlatform().GetOsName();

            if (!TryDeterminePlatformFromOSName(osName, out string? platform))
                throw new NotSupportedException("Target platform not supported.");

            if (!new[] { OS.Platforms.Windows, OS.Platforms.LinuxBSD, OS.Platforms.MacOS, OS.Platforms.Android, OS.Platforms.iOS, OS.Platforms.Web }
                    .Contains(platform))
            {
                throw new NotImplementedException("Target platform not yet implemented.");
            }

            // For web platform, always target wasm architecture
            if (platform == OS.Platforms.Web && publishConfig.Archs.Count == 0)
            {
                publishConfig.Archs.Add("wasm");
            }
        }

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
                "arm32" => "arm",
                "arm64" => "arm64",
                "wasm" => "wasm",
                _ => throw new ArgumentOutOfRangeException(nameof(arch), arch, "Unexpected architecture")
            };
        }

        private bool ProjectContainsDotNet()
        {
            // Implementation to check if project contains .NET
            return true; // Placeholder
        }

        private bool TryDeterminePlatformFromOSName(string osName, out string? platform)
        {
            // Implementation to determine platform from OS name
            platform = osName;
            return true; // Placeholder
        }
    }
}
