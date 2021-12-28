using Godot.NativeInterop;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Text;
using GodotTools.Internals;

namespace GodotTools.Utils
{
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public static class OS
    {
        private static class Names
        {
            public const string Windows = "Windows";
            public const string MacOS = "macOS";
            public const string Linux = "Linux";
            public const string FreeBSD = "FreeBSD";
            public const string NetBSD = "NetBSD";
            public const string BSD = "BSD";
            public const string Server = "Server";
            public const string UWP = "UWP";
            public const string Haiku = "Haiku";
            public const string Android = "Android";
            public const string iOS = "iOS";
            public const string HTML5 = "HTML5";
        }

        public static class Platforms
        {
            public const string Windows = "windows";
            public const string MacOS = "osx";
            public const string LinuxBSD = "linuxbsd";
            public const string Server = "server";
            public const string UWP = "uwp";
            public const string Haiku = "haiku";
            public const string Android = "android";
            public const string iOS = "iphone";
            public const string HTML5 = "javascript";
        }

        private static class DotNetOS
        {
            public const string Win = "win";
            public const string OSX = "osx";
            public const string Linux = "linux";
            public const string Win10 = "win10";
            public const string Android = "android";
            public const string iOS = "ios";
            public const string Browser = "browser";
        }

        public static readonly Dictionary<string, string> PlatformFeatureMap = new Dictionary<string, string>(
            // Export `features` may be in lower case
            StringComparer.InvariantCultureIgnoreCase
        )
        {
            ["Windows"] = Platforms.Windows,
            ["macOS"] = Platforms.MacOS,
            ["LinuxBSD"] = Platforms.LinuxBSD,
            // "X11" for compatibility, temporarily, while we are on an outdated branch
            ["X11"] = Platforms.LinuxBSD,
            ["Server"] = Platforms.Server,
            ["UWP"] = Platforms.UWP,
            ["Haiku"] = Platforms.Haiku,
            ["Android"] = Platforms.Android,
            ["iOS"] = Platforms.iOS,
            ["HTML5"] = Platforms.HTML5
        };

        public static readonly Dictionary<string, string> PlatformNameMap = new Dictionary<string, string>
        {
            [Names.Windows] = Platforms.Windows,
            [Names.MacOS] = Platforms.MacOS,
            [Names.Linux] = Platforms.LinuxBSD,
            [Names.FreeBSD] = Platforms.LinuxBSD,
            [Names.NetBSD] = Platforms.LinuxBSD,
            [Names.BSD] = Platforms.LinuxBSD,
            [Names.Server] = Platforms.Server,
            [Names.UWP] = Platforms.UWP,
            [Names.Haiku] = Platforms.Haiku,
            [Names.Android] = Platforms.Android,
            [Names.iOS] = Platforms.iOS,
            [Names.HTML5] = Platforms.HTML5
        };

        public static readonly Dictionary<string, string> DotNetOSPlatformMap = new Dictionary<string, string>
        {
            [Platforms.Windows] = DotNetOS.Win,
            [Platforms.MacOS] = DotNetOS.OSX,
            // TODO:
            // Does .NET 6 support BSD variants? If it does, it may need the name `unix`
            // instead of `linux` in the runtime identifier. This would be a problem as
            // Godot has a single export profile for both, named LinuxBSD.
            [Platforms.LinuxBSD] = DotNetOS.Linux,
            [Platforms.Server] = DotNetOS.Linux,
            [Platforms.UWP] = DotNetOS.Win10,
            [Platforms.Android] = DotNetOS.Android,
            [Platforms.iOS] = DotNetOS.iOS,
            [Platforms.HTML5] = DotNetOS.Browser
        };

        private static bool IsOS(string name)
        {
            Internal.godot_icall_Utils_OS_GetPlatformName(out godot_string dest);
            using (dest)
            {
                string platformName = Marshaling.ConvertStringToManaged(dest);
                return name.Equals(platformName, StringComparison.OrdinalIgnoreCase);
            }
        }

        private static bool IsAnyOS(IEnumerable<string> names)
        {
            Internal.godot_icall_Utils_OS_GetPlatformName(out godot_string dest);
            using (dest)
            {
                string platformName = Marshaling.ConvertStringToManaged(dest);
                return names.Any(p => p.Equals(platformName, StringComparison.OrdinalIgnoreCase));
            }
        }

        private static readonly IEnumerable<string> LinuxBSDPlatforms =
            new[] { Names.Linux, Names.FreeBSD, Names.NetBSD, Names.BSD };

        private static readonly IEnumerable<string> UnixLikePlatforms =
            new[] { Names.MacOS, Names.Server, Names.Haiku, Names.Android, Names.iOS }
                .Concat(LinuxBSDPlatforms).ToArray();

        private static readonly Lazy<bool> _isWindows = new Lazy<bool>(() => IsOS(Names.Windows));
        private static readonly Lazy<bool> _isMacOS = new Lazy<bool>(() => IsOS(Names.MacOS));
        private static readonly Lazy<bool> _isLinuxBSD = new Lazy<bool>(() => IsAnyOS(LinuxBSDPlatforms));
        private static readonly Lazy<bool> _isServer = new Lazy<bool>(() => IsOS(Names.Server));
        private static readonly Lazy<bool> _isUWP = new Lazy<bool>(() => IsOS(Names.UWP));
        private static readonly Lazy<bool> _isHaiku = new Lazy<bool>(() => IsOS(Names.Haiku));
        private static readonly Lazy<bool> _isAndroid = new Lazy<bool>(() => IsOS(Names.Android));
        private static readonly Lazy<bool> _isiOS = new Lazy<bool>(() => IsOS(Names.iOS));
        private static readonly Lazy<bool> _isHTML5 = new Lazy<bool>(() => IsOS(Names.HTML5));
        private static readonly Lazy<bool> _isUnixLike = new Lazy<bool>(() => IsAnyOS(UnixLikePlatforms));

        // TODO SupportedOSPlatformGuard once we target .NET 6
        // [SupportedOSPlatformGuard("windows")]
        public static bool IsWindows => _isWindows.Value || IsUWP;

        // [SupportedOSPlatformGuard("osx")]
        public static bool IsMacOS => _isMacOS.Value;

        // [SupportedOSPlatformGuard("linux")]
        public static bool IsLinuxBSD => _isLinuxBSD.Value;

        // [SupportedOSPlatformGuard("linux")]
        public static bool IsServer => _isServer.Value;

        // [SupportedOSPlatformGuard("windows")]
        public static bool IsUWP => _isUWP.Value;

        public static bool IsHaiku => _isHaiku.Value;

        // [SupportedOSPlatformGuard("android")]
        public static bool IsAndroid => _isAndroid.Value;

        // [SupportedOSPlatformGuard("ios")]
        public static bool IsiOS => _isiOS.Value;

        // [SupportedOSPlatformGuard("browser")]
        public static bool IsHTML5 => _isHTML5.Value;
        public static bool IsUnixLike => _isUnixLike.Value;

        public static char PathSep => IsWindows ? ';' : ':';

        [return: MaybeNull]
        public static string PathWhich([NotNull] string name)
        {
            if (IsWindows)
                return PathWhichWindows(name);

            return PathWhichUnix(name);
        }

        [return: MaybeNull]
        private static string PathWhichWindows([NotNull] string name)
        {
            string[] windowsExts =
                Environment.GetEnvironmentVariable("PATHEXT")?.Split(PathSep) ?? Array.Empty<string>();
            string[] pathDirs = Environment.GetEnvironmentVariable("PATH")?.Split(PathSep);
            char[] invalidPathChars = Path.GetInvalidPathChars();

            var searchDirs = new List<string>();

            if (pathDirs != null)
            {
                foreach (var pathDir in pathDirs)
                {
                    if (pathDir.IndexOfAny(invalidPathChars) != -1)
                        continue;

                    searchDirs.Add(pathDir);
                }
            }

            string nameExt = Path.GetExtension(name);
            bool hasPathExt = !string.IsNullOrEmpty(nameExt) &&
                windowsExts.Contains(nameExt, StringComparer.OrdinalIgnoreCase);

            searchDirs.Add(System.IO.Directory.GetCurrentDirectory()); // last in the list

            if (hasPathExt)
                return searchDirs.Select(dir => Path.Combine(dir, name)).FirstOrDefault(File.Exists);

            return (from dir in searchDirs
                select Path.Combine(dir, name)
                into path
                from ext in windowsExts
                select path + ext).FirstOrDefault(File.Exists);
        }

        [return: MaybeNull]
        private static string PathWhichUnix([NotNull] string name)
        {
            string[] pathDirs = Environment.GetEnvironmentVariable("PATH")?.Split(PathSep);
            char[] invalidPathChars = Path.GetInvalidPathChars();

            var searchDirs = new List<string>();

            if (pathDirs != null)
            {
                foreach (var pathDir in pathDirs)
                {
                    if (pathDir.IndexOfAny(invalidPathChars) != -1)
                        continue;

                    searchDirs.Add(pathDir);
                }
            }

            searchDirs.Add(System.IO.Directory.GetCurrentDirectory()); // last in the list

            return searchDirs.Select(dir => Path.Combine(dir, name))
                .FirstOrDefault(path =>
                {
                    using godot_string pathIn = Marshaling.ConvertStringToNative(path);
                    return File.Exists(path) && Internal.godot_icall_Utils_OS_UnixFileHasExecutableAccess(pathIn);
                });
        }

        public static void RunProcess(string command, IEnumerable<string> arguments)
        {
            var startInfo = new ProcessStartInfo(command)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };

            foreach (string arg in arguments)
                startInfo.ArgumentList.Add(arg);

            using Process process = Process.Start(startInfo);

            if (process == null)
                throw new Exception("No process was started");

            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            if (IsWindows && process.Id > 0)
                User32Dll.AllowSetForegroundWindow(process.Id); // Allows application to focus itself
        }

        public static int ExecuteCommand(string command, IEnumerable<string> arguments)
        {
            var startInfo = new ProcessStartInfo(command)
            {
                // Print the output
                RedirectStandardOutput = false,
                RedirectStandardError = false,
                UseShellExecute = false
            };

            foreach (string arg in arguments)
                startInfo.ArgumentList.Add(arg);

            Console.WriteLine(startInfo.GetCommandLineDisplay(new StringBuilder("Executing: ")).ToString());

            using var process = new Process { StartInfo = startInfo };
            process.Start();
            process.WaitForExit();

            return process.ExitCode;
        }

        private static void AppendProcessFileNameForDisplay(this StringBuilder builder, string fileName)
        {
            if (builder.Length > 0)
                builder.Append(' ');

            if (fileName.Contains(' '))
            {
                builder.Append('"');
                builder.Append(fileName);
                builder.Append('"');
            }
            else
            {
                builder.Append(fileName);
            }
        }

        private static void AppendProcessArgumentsForDisplay(this StringBuilder builder,
            Collection<string> argumentList)
        {
            // This is intended just for reading. It doesn't need to be a valid command line.
            // E.g.: We don't handle escaping of quotes.

            foreach (string argument in argumentList)
            {
                if (builder.Length > 0)
                    builder.Append(' ');

                if (argument.Contains(' '))
                {
                    builder.Append('"');
                    builder.Append(argument);
                    builder.Append('"');
                }
                else
                {
                    builder.Append(argument);
                }
            }
        }

        public static StringBuilder GetCommandLineDisplay(
            this ProcessStartInfo startInfo,
            StringBuilder optionalBuilder = null
        )
        {
            var builder = optionalBuilder ?? new StringBuilder();

            builder.AppendProcessFileNameForDisplay(startInfo.FileName);

            if (startInfo.ArgumentList.Count == 0)
            {
                builder.Append(' ');
                builder.Append(startInfo.Arguments);
            }
            else
            {
                builder.AppendProcessArgumentsForDisplay(startInfo.ArgumentList);
            }

            return builder;
        }
    }
}
