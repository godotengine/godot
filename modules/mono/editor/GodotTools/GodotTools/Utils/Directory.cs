using System.IO;
using Godot;

namespace GodotTools.Utils
{
    public static class Directory
    {
        private static string GlobalizePath(this string path)
        {
            return ProjectSettings.GlobalizePath(path);
        }

        public static bool Exists(string path)
        {
            return System.IO.Directory.Exists(path.GlobalizePath());
        }

        /// Create directory recursively
        public static DirectoryInfo CreateDirectory(string path)
        {
            return System.IO.Directory.CreateDirectory(path.GlobalizePath());
        }

        public static void Delete(string path, bool recursive)
        {
            System.IO.Directory.Delete(path.GlobalizePath(), recursive);
        }

        public static void CopyDirectory(string sourceDir, string destinationDir, bool recursive)
        {
            // Check if the source directory exists
            if (!Exists(sourceDir))
                throw new DirectoryNotFoundException($"Source directory not found: {sourceDir.GlobalizePath()}");

            // Cache directories before we start copying
            string[] dirs = GetDirectories(sourceDir, "*", SearchOption.TopDirectoryOnly);

            // Create the destination directory
            CreateDirectory(destinationDir);

            // Get the files in the source directory and copy to the destination directory
            foreach (string file in GetFiles(sourceDir, "*", SearchOption.TopDirectoryOnly))
            {
                GodotTools.Utils.File.Copy(file, Path.Combine(destinationDir, file.GetFile()));
            }

            // If recursive and copying subdirectories, recursively call this method
            if (recursive)
            {
                foreach (string subDir in dirs)
                {
                    CopyDirectory(subDir, Path.Combine(destinationDir, new DirectoryInfo(subDir).Name), true);
                }
            }
        }

        public static string[] GetDirectories(string path, string searchPattern, SearchOption searchOption)
        {
            return System.IO.Directory.GetDirectories(path.GlobalizePath(), searchPattern, searchOption);
        }

        public static string[] GetFiles(string path, string searchPattern, SearchOption searchOption)
        {
            return System.IO.Directory.GetFiles(path.GlobalizePath(), searchPattern, searchOption);
        }
    }
}
