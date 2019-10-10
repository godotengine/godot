using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;

namespace GodotTools.Utils
{
    public static class OS
    {
        [MethodImpl(MethodImplOptions.InternalCall)]
        extern static string GetPlatformName();

        const string HaikuName = "Haiku";
        const string OSXName = "OSX";
        const string ServerName = "Server";
        const string UWPName = "UWP";
        const string WindowsName = "Windows";
        const string X11Name = "X11";

        public static bool IsHaiku()
        {
            return HaikuName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsOSX()
        {
            return OSXName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsServer()
        {
            return ServerName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsUWP()
        {
            return UWPName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsWindows()
        {
            return WindowsName.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        public static bool IsX11()
        {
            return X11Name.Equals(GetPlatformName(), StringComparison.OrdinalIgnoreCase);
        }

        private static bool? _isUnixCache;
        private static readonly string[] UnixPlatforms = {HaikuName, OSXName, ServerName, X11Name};

        public static bool IsUnix()
        {
            if (_isUnixCache.HasValue)
                return _isUnixCache.Value;

            string osName = GetPlatformName();
            _isUnixCache = UnixPlatforms.Any(p => p.Equals(osName, StringComparison.OrdinalIgnoreCase));
            return _isUnixCache.Value;
        }

        public static char PathSep => IsWindows() ? ';' : ':';

        public static string PathWhich(string name)
        {
            string[] windowsExts = IsWindows() ? Environment.GetEnvironmentVariable("PATHEXT")?.Split(PathSep) : null;
            string[] pathDirs = Environment.GetEnvironmentVariable("PATH")?.Split(PathSep);

            var searchDirs = new List<string>();

            if (pathDirs != null)
                searchDirs.AddRange(pathDirs);

            searchDirs.Add(System.IO.Directory.GetCurrentDirectory()); // last in the list

            foreach (var dir in searchDirs)
            {
                string path = Path.Combine(dir, name);

                if (IsWindows() && windowsExts != null)
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
            string CmdLineArgsToString(IEnumerable<string> args)
            {
                return string.Join(" ", args.Select(arg => arg.Contains(" ") ? $@"""{arg}""" : arg));
            }

            ProcessStartInfo startInfo = new ProcessStartInfo(command, CmdLineArgsToString(arguments))
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
