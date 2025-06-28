using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Threading;
using Godot;
using Godot.NativeInterop;
using Microsoft.VisualStudio.SolutionPersistence;
using Microsoft.VisualStudio.SolutionPersistence.Serializer;

namespace GodotTools.Internals
{
    public static class GodotSharpDirs
    {
        public static string ResMetadataDir
        {
            get
            {
                Internal.godot_icall_GodotSharpDirs_ResMetadataDir(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }

        public static string MonoUserDir
        {
            get
            {
                Internal.godot_icall_GodotSharpDirs_MonoUserDir(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }

        public static string BuildLogsDirs
        {
            get
            {
                Internal.godot_icall_GodotSharpDirs_BuildLogsDirs(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }

        public static string DataEditorToolsDir
        {
            get
            {
                Internal.godot_icall_GodotSharpDirs_DataEditorToolsDir(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }


        public static string CSharpProjectName
        {
            get
            {
                Internal.godot_icall_GodotSharpDirs_CSharpProjectName(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }

        [MemberNotNull("_projectAssemblyName", "_projectSlnPath", "_projectCsProjPath")]
        public static void DetermineProjectLocation()
        {
            _projectAssemblyName = (string?)ProjectSettings.GetSetting("dotnet/project/assembly_name");
            if (string.IsNullOrEmpty(_projectAssemblyName))
            {
                _projectAssemblyName = CSharpProjectName;
                ProjectSettings.SetSetting("dotnet/project/assembly_name", _projectAssemblyName);
            }

            string? slnParentDir = (string?)ProjectSettings.GetSetting("dotnet/project/solution_directory");
            if (string.IsNullOrEmpty(slnParentDir))
                slnParentDir = "res://";
            else if (!slnParentDir.StartsWith("res://", StringComparison.Ordinal))
                slnParentDir = "res://" + slnParentDir;

            // The csproj should be in the same folder as project.godot.
            string csprojParentDir = "res://";

            // Set csproj path first and use it to find the sln/slnx file with the assembly
            _projectCsProjPath = Path.Combine(ProjectSettings.GlobalizePath(csprojParentDir),
                string.Concat(_projectAssemblyName, ".csproj"));

            _projectSlnPath = FindSolutionFileWithAssemblyName(slnParentDir, _projectAssemblyName);
        }

        private static string FindSolutionFileWithAssemblyName(string directory, string assemblyName)
        {
            // Will convert ".." to load solutions from parent directory when appropriate
            string slnAbsolutePath = Path.GetFullPath(ProjectSettings.GlobalizePath(directory));

            List<string> solutionFilePaths = new();
            solutionFilePaths.AddRange(Directory.GetFiles(slnAbsolutePath, "*.sln"));
            solutionFilePaths.AddRange(Directory.GetFiles(slnAbsolutePath, "*.slnx"));

            if (solutionFilePaths.Count == 0)
                return Path.Combine(slnAbsolutePath, $"{assemblyName}.sln");

            List<string> matchingSolutions = new();

            foreach (string solutionFilePath in solutionFilePaths)
            {
                ISolutionSerializer? serializer = SolutionSerializers.GetSerializerByMoniker(solutionFilePath);
                if (serializer is null)
                    continue;

                string? solutionDirectory = Path.GetDirectoryName(solutionFilePath);
                if (solutionDirectory is null)
                    continue;

                var solution = serializer.OpenAsync(solutionFilePath, CancellationToken.None).Result;

                foreach (var project in solution.SolutionProjects)
                {
                    // Convert '\' path separators on Windows to '/' to match Godot's Unix style separators
                    var absoluteProjectFilePath = Path.GetFullPath(project.FilePath, solutionDirectory).Replace('\\', '/');

                    if (string.Equals(absoluteProjectFilePath, _projectCsProjPath, StringComparison.Ordinal))
                        matchingSolutions.Add(solutionFilePath);
                }
            }

            switch (matchingSolutions.Count)
            {
                case 1:
                    return matchingSolutions[0];

                case > 1:
                    GD.PushError(
                        $"Multiple solutions containing a project with assembly name '{assemblyName}' were found:\n"
                        + $"{string.Join('\n', matchingSolutions).Replace('\\', '/')}\n"
                        + "Please ensure only one solution contains the project assembly.\n"
                        + "If you have recently migrated to .slnx please ensure that you have removed the unused .sln.");
                    break;
            }

            return Path.Combine(slnAbsolutePath, $"{assemblyName}.sln");
        }

        private static string? _projectAssemblyName;
        private static string? _projectSlnPath;
        private static string? _projectCsProjPath;

        public static string ProjectAssemblyName
        {
            get
            {
                if (_projectAssemblyName == null)
                    DetermineProjectLocation();
                return _projectAssemblyName;
            }
        }

        public static string ProjectSlnPath
        {
            get
            {
                if (_projectSlnPath == null)
                    DetermineProjectLocation();
                return _projectSlnPath;
            }
        }

        public static string ProjectCsProjPath
        {
            get
            {
                if (_projectCsProjPath == null)
                    DetermineProjectLocation();
                return _projectCsProjPath;
            }
        }

        public static string ProjectBaseOutputPath
        {
            get
            {
                if (_projectCsProjPath == null)
                    DetermineProjectLocation();
                return Path.Combine(Path.GetDirectoryName(_projectCsProjPath)!, ".godot", "mono", "temp", "bin");
            }
        }

        public static string LogsDirPathFor(string solution, string configuration)
            => Path.Combine(BuildLogsDirs, $"{solution.Md5Text()}_{configuration}");

        public static string LogsDirPathFor(string configuration)
            => LogsDirPathFor(ProjectSlnPath, configuration);
    }
}
