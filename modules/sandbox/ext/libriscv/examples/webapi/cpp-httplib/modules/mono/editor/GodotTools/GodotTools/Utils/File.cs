using System;
using Godot;

namespace GodotTools.Utils
{
    public static class File
    {
        private static string GlobalizePath(this string path)
        {
            return ProjectSettings.GlobalizePath(path);
        }

        public static void WriteAllText(string path, string contents)
        {
            System.IO.File.WriteAllText(path.GlobalizePath(), contents);
        }

        public static bool Exists(string path)
        {
            return System.IO.File.Exists(path.GlobalizePath());
        }

        public static DateTime GetLastWriteTime(string path)
        {
            return System.IO.File.GetLastWriteTime(path.GlobalizePath());
        }

        public static void Delete(string path)
        {
            System.IO.File.Delete(path.GlobalizePath());
        }

        public static void Copy(string sourceFileName, string destFileName)
        {
            System.IO.File.Copy(sourceFileName.GlobalizePath(), destFileName.GlobalizePath(), overwrite: true);
        }

        public static byte[] ReadAllBytes(string path)
        {
            return System.IO.File.ReadAllBytes(path.GlobalizePath());
        }
    }
}
