using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace GodotTools.Core
{
    public static class StringExtensions
    {
        private static readonly string _driveRoot = Path.GetPathRoot(Environment.CurrentDirectory);

        public static string RelativeToPath(this string path, string dir)
        {
            // Make sure the directory ends with a path separator
            dir = Path.Combine(dir, " ").TrimEnd();

            if (Path.DirectorySeparatorChar == '\\')
                dir = dir.Replace("/", "\\") + "\\";

            var fullPath = new Uri(Path.GetFullPath(path), UriKind.Absolute);
            var relRoot = new Uri(Path.GetFullPath(dir), UriKind.Absolute);

            // MakeRelativeUri converts spaces to %20, hence why we need UnescapeDataString
            return Uri.UnescapeDataString(relRoot.MakeRelativeUri(fullPath).ToString());
        }

        public static string NormalizePath(this string path)
        {
            if (string.IsNullOrEmpty(path))
                return path;

            bool rooted = path.IsAbsolutePath();

            path = path.Replace('\\', '/');
            path = path[path.Length - 1] == '/' ? path.Substring(0, path.Length - 1) : path;

            string[] parts = path.Split(new[] {'/'}, StringSplitOptions.RemoveEmptyEntries);

            path = string.Join(Path.DirectorySeparatorChar.ToString(), parts).Trim();

            if (!rooted)
                return path;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                string maybeDrive = parts[0];
                if (maybeDrive.Length == 2 && maybeDrive[1] == ':')
                    return path; // Already has drive letter
            }

            return Path.DirectorySeparatorChar + path;
        }

        public static bool IsAbsolutePath(this string path)
        {
            return path.StartsWith("/", StringComparison.Ordinal) ||
                   path.StartsWith("\\", StringComparison.Ordinal) ||
                   path.StartsWith(_driveRoot, StringComparison.Ordinal);
        }

        public static string ToSafeDirName(this string dirName, bool allowDirSeparator = false)
        {
            var invalidChars = new List<string> {":", "*", "?", "\"", "<", ">", "|"};

            if (allowDirSeparator)
            {
                // Directory separators are allowed, but disallow ".." to avoid going up the filesystem
                invalidChars.Add("..");
            }
            else
            {
                invalidChars.Add("/");
            }

            string safeDirName = dirName.Replace("\\", "/").Trim();

            foreach (string invalidChar in invalidChars)
                safeDirName = safeDirName.Replace(invalidChar, "-");

            // Avoid reserved names that conflict with Godot assemblies
            if (safeDirName == "GodotSharp" || safeDirName == "GodotSharpEditor")
            {
                safeDirName += "_";
            }

            return safeDirName;
        }
    }
}
