using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;

namespace GodotTools.Utils
{
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public static class OS
    {
        [MethodImpl(MethodImplOptions.InternalCall)]
        static extern string GetPlatformName();

        public static class Names
        {
            public const string Windows = "Windows";
            public const string OSX = "OSX";
            public const string X11 = "X11";
            public const string Server = "Server";
            public const string UWP = "UWP";
            public const string Haiku = "Haiku";
            public const string Android = "Android";
            public const string HTML5 = "HTML5";
        }

        public static class Platforms
        {
            public const string Windows = "windows";
            public const string OSX = "osx";
            public const string X11 = "x11";
            public const string Server = "server";
            public const string UWP = "uwp";
            public const string Haiku = "haiku";
            public const string Android = "android";
            public const string HTML5 = "javascript";
        }

        public static readonly Dictionary<string, string> PlatformNameMap = new Dictionary<string, string>
        {
            [Names.Windows] = Platforms.Windows,
            [Names.OSX] = Platforms.OSX,
            [Names.X11] = Platforms.X11,
            [Names.Server] = Platforms.Server,
            [Names.UWP] = Platforms.UWP,
            [Names.Haiku] = Platforms.Haiku,
            [Names.Android] = Platforms.Android,
            [Names.HTML5] = Platforms.HTML5
        };

        private static bool IsOS(string name)
        {
            return name.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsWindows => IsOS(Names.Windows);

        public static bool IsOSX => IsOS(Names.OSX);

        public static bool IsX11 => IsOS(Names.X11);

        public static bool IsServer => IsOS(Names.Server);

        public static bool IsUWP => IsOS(Names.UWP);

        public static bool IsHaiku => IsOS(Names.Haiku);

        public static bool IsAndroid => IsOS(Names.Android);

        public static bool IsHTML5 => IsOS(Names.HTML5);

        private static bool? _isUnixCache;
        private static readonly string[] UnixLikePlatforms = {Names.OSX, Names.X11, Names.Server, Names.Haiku, Names.Android};

        public static bool IsUnixLike()
        {
            if (_isUnixCache.HasValue)
                return _isUnixCache.Value;

            string osName = GetPlatformName();
            _isUnixCache = UnixLikePlatforms.Any(p => p.Equals(osName, StringComparison.OrdinalIgnoreCase));
            return _isUnixCache.Value;
        }

        public static char PathSep => IsWindows ? ';' : ':';

        public static string PathWhich(string name)
        {
            string[] windowsExts = IsWindows ? Environment.GetEnvironmentVariable("PATHEXT")?.Split(PathSep) : null;
            string[] pathDirs = Environment.GetEnvironmentVariable("PATH")?.Split(PathSep);

            var searchDirs = new List<string>();

            if (pathDirs != null)
                searchDirs.AddRange(pathDirs);

            searchDirs.Add(System.IO.Directory.GetCurrentDirectory()); // last in the list

            foreach (var dir in searchDirs)
            {
                string path = Path.Combine(dir, name);

                if (IsWindows && windowsExts != null)
                {
                    foreach (var extension in windowsExts)
                    {
                        string pathWithExtension = path + extension;

                        if (File.Exists(pathWithExtension))
                            return pathWithExtension;
                    }
                }
                else
                {
                    if (File.Exists(path))
                        return path;
                }
            }

            return null;
        }

        public static void RunProcess(string command, IEnumerable<string> arguments)
        {
            // TODO: Once we move to .NET Standard 2.1 we can use ProcessStartInfo.ArgumentList instead
            string CmdLineArgsToString(IEnumerable<string> args)
            {
                // Not perfect, but as long as we are careful...
                return string.Join(" ", args.Select(arg => arg.Contains(" ") ? $@"""{arg}""" : arg));
            }

            var startInfo = new ProcessStartInfo(command, CmdLineArgsToString(arguments))
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false
            };

            using (Process process = Process.Start(startInfo))
            {
                if (process == null)
                    throw new Exception("No process was started");

                process.BeginOutputReadLine();
                process.BeginErrorReadLine();
            }
        }
    }
}
