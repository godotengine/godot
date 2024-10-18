using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using OS = GodotTools.Utils.OS;

namespace GodotTools.Build
{
    public static class DotNetFinder
    {
        public static string? FindDotNetExe()
        {
            // In the future, this method may do more than just search in PATH. We could look in
            // known locations or use Godot's linked nethost to search from the hostfxr location.

            if (OS.IsMacOS)
            {
                if (RuntimeInformation.OSArchitecture == Architecture.X64)
                {
                    string dotnet_x64 = "/usr/local/share/dotnet/x64/dotnet"; // Look for x64 version, when running under Rosetta 2.
                    if (File.Exists(dotnet_x64))
                    {
                        return dotnet_x64;
                    }
                }
                string dotnet = "/usr/local/share/dotnet/dotnet"; // Look for native version.
                if (File.Exists(dotnet))
                {
                    return dotnet;
                }
            }

            return OS.PathWhich("dotnet");
        }

        public static bool TryFindDotNetSdk(
            Version expectedVersion,
            [NotNullWhen(true)] out Version? version,
            [NotNullWhen(true)] out string? path
        )
        {
            version = null;
            path = null;

            string? dotNetExe = FindDotNetExe();

            if (string.IsNullOrEmpty(dotNetExe))
                return false;

            // Detect default sdk installed in the system (the newest, usually).
            if (!GetDotnetVersion(dotNetExe, null, out var systemSdkVersion))
            {
                // This should never happen, dotnet always has at least one sdk installed.
                return false;
            }

            // Check whether the project requires a different sdk version from the default we detected above.
            if (GetDotnetVersion(dotNetExe, Environment.CurrentDirectory, out version) && version != systemSdkVersion)
            {
                // The project has a required sdk version, use that one instead of the one we guessed.
                expectedVersion = version;
            }

            using Process process = new Process();
            process.StartInfo = new ProcessStartInfo(dotNetExe, "--list-sdks")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true
            };

            if (OperatingSystem.IsWindows())
            {
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
            }

            process.StartInfo.EnvironmentVariables["DOTNET_CLI_UI_LANGUAGE"] = "en-US";

            var lines = new List<string>();

            process.OutputDataReceived += (_, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                    lines.Add(e.Data);
            };

            try
            {
                process.Start();
            }
            catch
            {
                return false;
            }

            process.BeginOutputReadLine();
            process.WaitForExit();

            Version? latestVersionMatch = null;
            string? matchPath = null;

            foreach (var line in lines)
            {
                string[] sdkLineParts = line.Trim()
                    .Split(' ', 2, StringSplitOptions.TrimEntries);

                if (sdkLineParts.Length < 2)
                    continue;

                if (!Version.TryParse(sdkLineParts[0], out var lineVersion))
                    continue;

                // This is the exact version we're looking for.
                if (lineVersion == expectedVersion)
                {
                    latestVersionMatch = lineVersion;
                    matchPath = sdkLineParts[1].TrimStart('[').TrimEnd(']');
                    break;
                }

                // We're looking for the exact same major version.
                if (lineVersion.Major != expectedVersion.Major)
                    continue;

                if (latestVersionMatch != null && lineVersion < latestVersionMatch)
                    continue;

                latestVersionMatch = lineVersion;
                matchPath = sdkLineParts[1].TrimStart('[').TrimEnd(']');
            }

            if (latestVersionMatch == null)
                return false;

            version = latestVersionMatch;
            path = Path.Combine(matchPath!, version.ToString());

            return true;
        }

        private static bool GetDotnetVersion(string dotNetExe, [CanBeNull] string projectPath, out Version version)
        {
            version = null;

            var lines = new List<string>();

            using Process process = new Process();

            process.StartInfo = new ProcessStartInfo(dotNetExe, "--version")
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                WorkingDirectory = projectPath ?? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            };

            if (OperatingSystem.IsWindows())
            {
                process.StartInfo.StandardOutputEncoding = Encoding.UTF8;
            }

            process.StartInfo.EnvironmentVariables["DOTNET_CLI_UI_LANGUAGE"] = "en-US";

            process.OutputDataReceived += (_, e) =>
            {
                if (!string.IsNullOrWhiteSpace(e.Data))
                    lines.Add(e.Data);
            };

            try
            {
                process.Start();
            }
            catch
            {
                return false;
            }

            process.BeginOutputReadLine();
            process.WaitForExit();

            return lines.Count > 0 && Version.TryParse(lines[0].Trim(), out version);
        }
    }
}
