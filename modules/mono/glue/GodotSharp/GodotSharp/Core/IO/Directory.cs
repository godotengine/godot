using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Godot
{
    public partial class Directory
    {
        private static void ThrowIfParamPathIsNullOrEmpty(string path, string paramName)
        {
            if (path == null)
                throw new ArgumentNullException(paramName, "File name cannot be null.");
            if (path.Length == 0)
                throw new ArgumentException("Empty file name is not legal.", paramName);
        }

        // TODO: Need a replacement for DirectoryInfo, which is what this method would return.
        public static string GetParent(string path)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            string parent = path.GetBaseDir();

            if (string.IsNullOrEmpty(parent))
                return null;

            return parent;
        }

        // TODO: Need a replacement for DirectoryInfo, which is what this method would return.
        public static string CreateDirectory(string path)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            using var dir = new Directory();

            Error error;

            // TODO: Replace this workaround (MakeDirRecursive fails if the directory or any of the sub-directories already exists)
            while ((error = dir.MakeDirRecursive(path)) == Error.AlreadyExists && !Exists(path))
            {
            }

            error.ThrowOnError();

            return path;
        }

        public static bool Exists(string path)
        {
            try
            {
                if (string.IsNullOrEmpty(path))
                    return false;

                using var dir = new Directory();
                return dir.DirExists(path);
            }
            catch (ArgumentException)
            {
            }
            catch (IOException)
            {
            }
            catch (UnauthorizedAccessException)
            {
            }

            return false;
        }

        private enum SearchKind
        {
            Files,
            Directories,
            All
        }

        public static string[] GetDirectories(string path)
            => GetDirectories(path, SearchOption.TopDirectoryOnly);

        public static string[] GetDirectories(string path, SearchOption searchOption)
            => EnumerateDirectories(path, searchOption).ToArray();

        public static string[] GetFiles(string path)
            => GetFiles(path, SearchOption.TopDirectoryOnly);

        public static string[] GetFiles(string path, SearchOption searchOption)
            => EnumerateFiles(path, searchOption).ToArray();

        public static string[] GetFileSystemEntries(string path)
            => GetFileSystemEntries(path, SearchOption.TopDirectoryOnly);

        public static string[] GetFileSystemEntries(string path, SearchOption searchOption)
            => EnumerateFileSystemEntries(path, searchOption).ToArray();

        public static IEnumerable<string> EnumerateDirectories(string path)
            => EnumerateDirectories(path, SearchOption.TopDirectoryOnly);

        public static IEnumerable<string> EnumerateDirectories(string path, SearchOption searchOption)
            => _EnumeratePaths(path, SearchKind.Directories, searchOption).ToArray();

        public static IEnumerable<string> EnumerateFiles(string path)
            => EnumerateFiles(path, SearchOption.TopDirectoryOnly);

        public static IEnumerable<string> EnumerateFiles(string path, SearchOption searchOption)
            => _EnumeratePaths(path, SearchKind.Files, searchOption).ToArray();

        public static IEnumerable<string> EnumerateFileSystemEntries(string path)
            => EnumerateFileSystemEntries(path, SearchOption.TopDirectoryOnly);

        public static IEnumerable<string> EnumerateFileSystemEntries(string path, SearchOption searchOption)
            => _EnumeratePaths(path, SearchKind.All, searchOption).ToArray();

        private static IEnumerable<string> _EnumeratePaths(
            string path,
            SearchKind searchKind,
            SearchOption options)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            using var dir = new Directory();

            var dirQueue = new Queue<string>();
            dirQueue.Enqueue(path);

            while (dirQueue.TryDequeue(out string currentDir))
            {
                dir.Open(currentDir).ThrowOnError();

                dir.ListDirBegin(skipNavigational: true, skipHidden: false).ThrowOnError();

                string next;
                while (!string.IsNullOrEmpty(next = dir.GetNext()))
                {
                    next = currentDir.PlusFile(next);

                    bool isDir = dir.CurrentIsDir();

                    if (isDir)
                    {
                        if (options == SearchOption.AllDirectories)
                            dirQueue.Enqueue(next);

                        if (searchKind == SearchKind.Directories || searchKind == SearchKind.All)
                            yield return next;
                    }
                    else
                    {
                        if (searchKind == SearchKind.Files || searchKind == SearchKind.All)
                            yield return next;
                    }
                }

                dir.ListDirEnd();
            }
        }

        public static string GetDirectoryRoot(string path)
        {
            if (path == null)
                throw new ArgumentNullException(nameof(path));

            string fullPath = Path.GetFullPath(path);
            string root = Path.GetPathRoot(fullPath)!;

            return root;
        }

        public static string GetCurrentDirectory() => System.Environment.CurrentDirectory;

        public static void Delete(string path)
        {
            ThrowIfParamPathIsNullOrEmpty(path, nameof(path));

            if (File.Exists(path))
                throw new IOException($"A file with the same name and location already exists: '{path}'.");
            if (!Exists(path))
                throw new DirectoryNotFoundException($"The directory could not found: '{path}'.");

            using var dir = new Directory();
            dir.Remove(path).ThrowOnError();
        }

        public static void Delete(string path, bool recursive)
            => throw new NotImplementedException();

        public static string[] GetLogicalDrives()
        {
            using var dir = new Directory();
            int driveCount = dir.GetDriveCount();
            var logicalDrives = new string[driveCount];

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                for (int i = 0; i < driveCount; i++)
                {
                    logicalDrives[i] = dir.GetDrive(i) + "\\";
                }
            }
            else
            {
                for (int i = 0; i < driveCount; i++)
                    logicalDrives[i] = dir.GetDrive(i);
            }

            return logicalDrives;
        }
    }
}
