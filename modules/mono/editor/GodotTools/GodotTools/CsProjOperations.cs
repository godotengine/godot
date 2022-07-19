using Godot;
using System;
using System.Linq;
using Godot.Collections;
using GodotTools.Internals;
using GodotTools.ProjectEditor;
using File = GodotTools.Utils.File;
using Directory = GodotTools.Utils.Directory;

namespace GodotTools
{
    public static class CsProjOperations
    {
        public static string GenerateGameProject(string dir, string name)
        {
            try
            {
                return ProjectGenerator.GenAndSaveGameProject(dir, name);
            }
            catch (Exception e)
            {
                GD.PushError(e.ToString());
                return string.Empty;
            }
        }

        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        private static ulong ConvertToTimestamp(this DateTime value)
        {
            TimeSpan elapsedTime = value - Epoch;
            return (ulong)elapsedTime.TotalSeconds;
        }

        private static bool TryParseFileMetadata(string includeFile, ulong modifiedTime, out Dictionary fileMetadata)
        {
            fileMetadata = null;

            var parseError = ScriptClassParser.ParseFile(includeFile, out var classes, out string errorStr);

            if (parseError != Error.Ok)
            {
                GD.PushError($"Failed to determine namespace and class for script: {includeFile}. Parse error: {errorStr ?? parseError.ToString()}");
                return false;
            }

            string searchName = System.IO.Path.GetFileNameWithoutExtension(includeFile);

            var firstMatch = classes.FirstOrDefault(classDecl =>
                    classDecl.BaseCount != 0 && // If it doesn't inherit anything, it can't be a Godot.Object.
                    classDecl.SearchName == searchName // Filter by the name we're looking for
            );

            if (firstMatch == null)
                return false; // Not found

            fileMetadata = new Dictionary
            {
                ["modified_time"] = $"{modifiedTime}",
                ["class"] = new Dictionary
                {
                    ["namespace"] = firstMatch.Namespace,
                    ["class_name"] = firstMatch.Name,
                    ["nested"] = firstMatch.Nested
                }
            };

            return true;
        }

        public static void GenerateScriptsMetadata(string projectPath, string outputPath)
        {
            var metadataDict = Internal.GetScriptsMetadataOrNothing().Duplicate();

            bool IsUpToDate(string includeFile, ulong modifiedTime)
            {
                return metadataDict.TryGetValue(includeFile, out var oldFileVar) &&
                       ulong.TryParse(((Dictionary)oldFileVar)["modified_time"] as string,
                           out ulong storedModifiedTime) && storedModifiedTime == modifiedTime;
            }

            var outdatedFiles = ProjectUtils.GetIncludeFiles(projectPath, "Compile")
                .Select(path => ("res://" + path).SimplifyGodotPath())
                .ToDictionary(path => path, path => File.GetLastWriteTime(path).ConvertToTimestamp())
                .Where(pair => !IsUpToDate(includeFile: pair.Key, modifiedTime: pair.Value))
                .ToArray();

            foreach (var pair in outdatedFiles)
            {
                metadataDict.Remove(pair.Key);

                string includeFile = pair.Key;

                if (TryParseFileMetadata(includeFile, modifiedTime: pair.Value, out var fileMetadata))
                    metadataDict[includeFile] = fileMetadata;
            }

            string json = metadataDict.Count <= 0 ? "{}" : JSON.Print(metadataDict);

            string baseDir = outputPath.GetBaseDir();

            if (!Directory.Exists(baseDir))
                Directory.CreateDirectory(baseDir);

            File.WriteAllText(outputPath, json);
        }
    }
}
