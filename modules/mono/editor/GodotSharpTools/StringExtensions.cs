using System;
using System.IO;

namespace GodotSharpTools
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

            return rooted ? Path.DirectorySeparatorChar.ToString() + path : path;
        }

        private static readonly string driveRoot = Path.GetPathRoot(Environment.CurrentDirectory);

        public static bool IsAbsolutePath(this string path)
        {
            return path.StartsWith("/") || path.StartsWith("\\") || path.StartsWith(driveRoot);
        }

        public static string CsvEscape(this string value, char delimiter = ',')
        {
            bool hasSpecialChar = value.IndexOfAny(new char[] { '\"', '\n', '\r', delimiter }) != -1;

            if (hasSpecialChar)
                return "\"" + value.Replace("\"", "\"\"") + "\"";

            return value;
        }
    }
}
