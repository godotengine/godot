using System;
using System.Collections.Generic;
using System.IO;

namespace GodotTools.Core
{
    public static class StringExtensions
    {
        public static string RelativeToPath(this string path, string dir)
        {
            // Make sure the directory ends with a path separator
            dir = Path.Combine(dir, " ").TrimEnd();

            if (Path.DirectorySeparatorChar == '\\')
                dir = dir.Replace("/", "\\") + "\\";

            Uri fullPath = new Uri(Path.GetFullPath(path), UriKind.Absolute);
            Uri relRoot = new Uri(Path.GetFullPath(dir), UriKind.Absolute);

            return relRoot.MakeRelativeUri(fullPath).ToString();
        }

        public static string NormalizePath(this string path)
        {
            bool rooted = path.IsAbsolutePath();

            path = path.Replace('\\', '/');

            string[] parts = path.Split(new[] { '/' }, StringSplitOptions.RemoveEmptyEntries);

            path = string.Join(Path.DirectorySeparatorChar.ToString(), parts).Trim();

            return rooted ? Path.DirectorySeparatorChar + path : path;
        }

        private static readonly string driveRoot = Path.GetPathRoot(Environment.CurrentDirectory);

        public static bool IsAbsolutePath(this string path)
        {
            return path.StartsWith("/", StringComparison.Ordinal) ||
                   path.StartsWith("\\", StringComparison.Ordinal) ||
                   path.StartsWith(driveRoot, StringComparison.Ordinal);
        }

        public static string CsvEscape(this string value, char delimiter = ',')
        {
            bool hasSpecialChar = value.IndexOfAny(new char[] { '\"', '\n', '\r', delimiter }) != -1;

            if (hasSpecialChar)
                return "\"" + value.Replace("\"", "\"\"") + "\"";

            return value;
        }

        public static string ToSafeDirName(this string dirName, bool allowDirSeparator)
        {
            var invalidChars = new List<string> { ":", "*", "?", "\"", "<", ">", "|" };

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

            return safeDirName;
        }
    }
}
