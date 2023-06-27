using System;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using Godot;
using GodotTools.Core;

namespace GodotTools.Utils
{
    public static class FsPathUtils
    {
        private static readonly string ResourcePath = ProjectSettings.GlobalizePath("res://");

        private static bool PathStartsWithAlreadyNorm(this string childPath, string parentPath)
        {
            // This won't work for Linux/macOS case insensitive file systems, but it's enough for our current problems
            bool caseSensitive = !OS.IsWindows;

            string parentPathNorm = parentPath.NormalizePath() + Path.DirectorySeparatorChar;
            string childPathNorm = childPath.NormalizePath() + Path.DirectorySeparatorChar;

            return childPathNorm.StartsWith(parentPathNorm,
                caseSensitive ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
        }

        public static bool PathStartsWith(this string childPath, string parentPath)
        {
            string childPathNorm = childPath.NormalizePath() + Path.DirectorySeparatorChar;
            string parentPathNorm = parentPath.NormalizePath() + Path.DirectorySeparatorChar;

            return childPathNorm.PathStartsWithAlreadyNorm(parentPathNorm);
        }

        [return: MaybeNull]
        public static string LocalizePathWithCaseChecked(string path)
        {
            string pathNorm = path.NormalizePath() + Path.DirectorySeparatorChar;
            string resourcePathNorm = ResourcePath.NormalizePath() + Path.DirectorySeparatorChar;

            if (!pathNorm.PathStartsWithAlreadyNorm(resourcePathNorm))
                return null;

            string result = "res://" + pathNorm.Substring(resourcePathNorm.Length);

            // Remove the last separator we added
            return result.Substring(0, result.Length - 1);
        }
    }
}
