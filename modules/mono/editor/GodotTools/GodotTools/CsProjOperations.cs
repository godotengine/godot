using Godot;
using System;
using Godot.Collections;
using GodotTools.Internals;
using GodotTools.ProjectEditor;
using static GodotTools.Internals.Globals;
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
                return ProjectGenerator.GenGameProject(dir, name, compileItems: new string[] { });
            }
            catch (Exception e)
            {
                GD.PushError(e.ToString());
                return string.Empty;
            }
        }

        public static void AddItem(string projectPath, string itemType, string include)
        {
            if (!(bool)GlobalDef("mono/project/auto_update_project", true))
                return;

            ProjectUtils.AddItemToProjectChecked(projectPath, itemType, include);
        }

        private static readonly DateTime Epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        private static ulong ConvertToTimestamp(this DateTime value)
        {
            TimeSpan elapsedTime = value - Epoch;
            return (ulong)elapsedTime.TotalSeconds;
        }

        public static void GenerateScriptsMetadata(string projectPath, string outputPath)
        {
            if (File.Exists(outputPath))
                File.Delete(outputPath);

            var oldDict = Internal.GetScriptsMetadataOrNothing();
            var newDict = new Godot.Collections.Dictionary<string, object>();

            foreach (var includeFile in ProjectUtils.GetIncludeFiles(projectPath, "Compile"))
            {
                string projectIncludeFile = ("res://" + includeFile).SimplifyGodotPath();

                ulong modifiedTime = File.GetLastWriteTime(projectIncludeFile).ConvertToTimestamp();

                if (oldDict.TryGetValue(projectIncludeFile, out var oldFileVar))
                {
                    var oldFileDict = (Dictionary)oldFileVar;

                    if (ulong.TryParse(oldFileDict["modified_time"] as string, out ulong storedModifiedTime))
                    {
                        if (storedModifiedTime == modifiedTime)
                        {
                            // No changes so no need to parse again
                            newDict[projectIncludeFile] = oldFileDict;
                            continue;
                        }
                    }
                }

                Error parseError = ScriptClassParser.ParseFile(projectIncludeFile, out var classes, out string errorStr);
                if (parseError != Error.Ok)
                {
                    GD.PushError($"Failed to determine namespace and class for script: {projectIncludeFile}. Parse error: {errorStr ?? parseError.ToString()}");
                    continue;
                }

                string searchName = System.IO.Path.GetFileNameWithoutExtension(projectIncludeFile);

                var classDict = new Dictionary();

                foreach (var classDecl in classes)
                {
                    if (classDecl.BaseCount == 0)
                        continue; // Does not inherit nor implement anything, so it can't be a script class

                    string classCmp = classDecl.Nested ?
                        classDecl.Name.Substring(classDecl.Name.LastIndexOf(".", StringComparison.Ordinal) + 1) :
                        classDecl.Name;

                    if (classCmp != searchName)
                        continue;

                    classDict["namespace"] = classDecl.Namespace;
                    classDict["class_name"] = classDecl.Name;
                    classDict["nested"] = classDecl.Nested;
                    break;
                }

                if (classDict.Count == 0)
                    continue; // Not found

                newDict[projectIncludeFile] = new Dictionary { ["modified_time"] = $"{modifiedTime}", ["class"] = classDict };
            }

            if (newDict.Count > 0)
            {
                string json = JSON.Print(newDict);

                string baseDir = outputPath.GetBaseDir();

                if (!Directory.Exists(baseDir))
                    Directory.CreateDirectory(baseDir);

                File.WriteAllText(outputPath, json);
            }
        }
    }
}
