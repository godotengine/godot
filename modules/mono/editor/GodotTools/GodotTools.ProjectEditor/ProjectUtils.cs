using GodotTools.Core;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using DotNet.Globbing;
using Microsoft.Build.Construction;

namespace GodotTools.ProjectEditor
{
    public static class ProjectUtils
    {
        public static void AddItemToProjectChecked(string projectPath, string itemType, string include)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            var normalizedInclude = include.RelativeToPath(dir).Replace("/", "\\");

            if (root.AddItemChecked(itemType, normalizedInclude))
                root.Save();
        }

        public static void RenameItemInProjectChecked(string projectPath, string itemType, string oldInclude, string newInclude)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            var normalizedOldInclude = oldInclude.NormalizePath();
            var normalizedNewInclude = newInclude.NormalizePath();

            var item = root.FindItemOrNullAbs(itemType, normalizedOldInclude);

            if (item == null)
                return;

            item.Include = normalizedNewInclude.RelativeToPath(dir).Replace("/", "\\");
            root.Save();
        }

        public static void RemoveItemFromProjectChecked(string projectPath, string itemType, string include)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            var normalizedInclude = include.NormalizePath();

            if (root.RemoveItemChecked(itemType, normalizedInclude))
                root.Save();
        }

        public static void RenameItemsToNewFolderInProjectChecked(string projectPath, string itemType, string oldFolder, string newFolder)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            bool dirty = false;

            var oldFolderNormalized = oldFolder.NormalizePath();
            var newFolderNormalized = newFolder.NormalizePath();
            string absOldFolderNormalized = Path.GetFullPath(oldFolderNormalized).NormalizePath();
            string absNewFolderNormalized = Path.GetFullPath(newFolderNormalized).NormalizePath();

            foreach (var item in root.FindAllItemsInFolder(itemType, oldFolderNormalized))
            {
                string absPathNormalized = Path.GetFullPath(item.Include).NormalizePath();
                string absNewIncludeNormalized = absNewFolderNormalized + absPathNormalized.Substring(absOldFolderNormalized.Length);
                item.Include = absNewIncludeNormalized.RelativeToPath(dir).Replace("/", "\\");
                dirty = true;
            }

            if (dirty)
                root.Save();
        }

        public static void RemoveItemsInFolderFromProjectChecked(string projectPath, string itemType, string folder)
        {
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            var folderNormalized = folder.NormalizePath();

            var itemsToRemove = root.FindAllItemsInFolder(itemType, folderNormalized).ToList();

            if (itemsToRemove.Count > 0)
            {
                foreach (var item in itemsToRemove)
                    item.Parent.RemoveChild(item);

                root.Save();
            }
        }

        private static string[] GetAllFilesRecursive(string rootDirectory, string mask)
        {
            string[] files = Directory.GetFiles(rootDirectory, mask, SearchOption.AllDirectories);

            // We want relative paths
            for (int i = 0; i < files.Length; i++)
            {
                files[i] = files[i].RelativeToPath(rootDirectory);
            }

            return files;
        }

        public static string[] GetIncludeFiles(string projectPath, string itemType)
        {
            var result = new List<string>();
            var existingFiles = GetAllFilesRecursive(Path.GetDirectoryName(projectPath), "*.cs");

            var globOptions = new GlobOptions();
            globOptions.Evaluation.CaseInsensitive = false;

            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            foreach (var itemGroup in root.ItemGroups)
            {
                if (itemGroup.Condition.Length != 0)
                    continue;

                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType != itemType)
                        continue;

                    string normalizedInclude = item.Include.NormalizePath();

                    var glob = Glob.Parse(normalizedInclude, globOptions);

                    // TODO Check somehow if path has no blob to avoid the following loop...

                    foreach (var existingFile in existingFiles)
                    {
                        if (glob.IsMatch(existingFile))
                        {
                            result.Add(existingFile);
                        }
                    }
                }
            }

            return result.ToArray();
        }

        ///  Simple function to make sure the Api assembly references are configured correctly
        public static void FixApiHintPath(string projectPath)
        {
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            bool dirty = false;

            void AddPropertyIfNotPresent(string name, string condition, string value)
            {
                if (root.PropertyGroups
                    .Any(g => (g.Condition == string.Empty || g.Condition.Trim() == condition) &&
                              g.Properties
                                  .Any(p => p.Name == name &&
                                            p.Value == value &&
                                            (p.Condition.Trim() == condition || g.Condition.Trim() == condition))))
                {
                    return;
                }

                root.AddProperty(name, value).Condition = " " + condition + " ";
                dirty = true;
            }

            AddPropertyIfNotPresent(name: "ApiConfiguration",
                condition: "'$(Configuration)' != 'ExportRelease'",
                value: "Debug");
            AddPropertyIfNotPresent(name: "ApiConfiguration",
                condition: "'$(Configuration)' == 'ExportRelease'",
                value: "Release");

            void SetReferenceHintPath(string referenceName, string condition, string hintPath)
            {
                foreach (var itemGroup in root.ItemGroups.Where(g =>
                    g.Condition.Trim() == string.Empty || g.Condition.Trim() == condition))
                {
                    var references = itemGroup.Items.Where(item =>
                        item.ItemType == "Reference" &&
                        item.Include == referenceName &&
                        (item.Condition.Trim() == condition || itemGroup.Condition.Trim() == condition));

                    var referencesWithHintPath = references.Where(reference =>
                        reference.Metadata.Any(m => m.Name == "HintPath"));

                    if (referencesWithHintPath.Any(reference => reference.Metadata
                        .Any(m => m.Name == "HintPath" && m.Value == hintPath)))
                    {
                        // Found a Reference item with the right HintPath
                        return;
                    }

                    var referenceWithHintPath = referencesWithHintPath.FirstOrDefault();
                    if (referenceWithHintPath != null)
                    {
                        // Found a Reference item with a wrong HintPath
                        foreach (var metadata in referenceWithHintPath.Metadata.ToList()
                            .Where(m => m.Name == "HintPath"))
                        {
                            // Safe to remove as we duplicate with ToList() to loop
                            referenceWithHintPath.RemoveChild(metadata);
                        }

                        referenceWithHintPath.AddMetadata("HintPath", hintPath);
                        dirty = true;
                        return;
                    }

                    var referenceWithoutHintPath = references.FirstOrDefault();
                    if (referenceWithoutHintPath != null)
                    {
                        // Found a Reference item without a HintPath
                        referenceWithoutHintPath.AddMetadata("HintPath", hintPath);
                        dirty = true;
                        return;
                    }
                }

                // Found no Reference item at all. Add it.
                root.AddItem("Reference", referenceName).Condition = " " + condition + " ";
                dirty = true;
            }

            const string coreProjectName = "GodotSharp";
            const string editorProjectName = "GodotSharpEditor";

            const string coreCondition = "";
            const string editorCondition = "'$(Configuration)' == 'Debug'";

            var coreHintPath = $"$(ProjectDir)/.mono/assemblies/$(ApiConfiguration)/{coreProjectName}.dll";
            var editorHintPath = $"$(ProjectDir)/.mono/assemblies/$(ApiConfiguration)/{editorProjectName}.dll";

            SetReferenceHintPath(coreProjectName, coreCondition, coreHintPath);
            SetReferenceHintPath(editorProjectName, editorCondition, editorHintPath);

            if (dirty)
                root.Save();
        }

        public static void MigrateFromOldConfigNames(string projectPath)
        {
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            bool dirty = false;

            bool hasGodotProjectGeneratorVersion = false;
            bool foundOldConfiguration = false;

            foreach (var propertyGroup in root.PropertyGroups.Where(g => g.Condition == string.Empty))
            {
                if (!hasGodotProjectGeneratorVersion && propertyGroup.Properties.Any(p => p.Name == "GodotProjectGeneratorVersion"))
                    hasGodotProjectGeneratorVersion = true;

                foreach (var configItem in propertyGroup.Properties
                    .Where(p => p.Condition.Trim() == "'$(Configuration)' == ''" && p.Value == "Tools"))
                {
                    configItem.Value = "Debug";
                    foundOldConfiguration = true;
                    dirty = true;
                }
            }

            if (!hasGodotProjectGeneratorVersion)
            {
                root.PropertyGroups.First(g => g.Condition == string.Empty)?
                    .AddProperty("GodotProjectGeneratorVersion", Assembly.GetExecutingAssembly().GetName().Version.ToString());
                dirty = true;
            }

            if (!foundOldConfiguration)
            {
                var toolsConditions = new[]
                {
                    "'$(Configuration)|$(Platform)' == 'Tools|AnyCPU'",
                    "'$(Configuration)|$(Platform)' != 'Tools|AnyCPU'",
                    "'$(Configuration)' == 'Tools'",
                    "'$(Configuration)' != 'Tools'"
                };

                foundOldConfiguration = root.PropertyGroups
                    .Any(g => toolsConditions.Any(c => c == g.Condition.Trim()));
            }

            if (foundOldConfiguration)
            {
                void MigrateConfigurationConditions(string oldConfiguration, string newConfiguration)
                {
                    void MigrateConditions(string oldCondition, string newCondition)
                    {
                        foreach (var propertyGroup in root.PropertyGroups.Where(g => g.Condition.Trim() == oldCondition))
                        {
                            propertyGroup.Condition = " " + newCondition + " ";
                            dirty = true;
                        }

                        foreach (var propertyGroup in root.PropertyGroups)
                        {
                            foreach (var prop in propertyGroup.Properties.Where(p => p.Condition.Trim() == oldCondition))
                            {
                                prop.Condition = " " + newCondition + " ";
                                dirty = true;
                            }
                        }

                        foreach (var itemGroup in root.ItemGroups.Where(g => g.Condition.Trim() == oldCondition))
                        {
                            itemGroup.Condition = " " + newCondition + " ";
                            dirty = true;
                        }

                        foreach (var itemGroup in root.ItemGroups)
                        {
                            foreach (var item in itemGroup.Items.Where(item => item.Condition.Trim() == oldCondition))
                            {
                                item.Condition = " " + newCondition + " ";
                                dirty = true;
                            }
                        }
                    }

                    foreach (var op in new[] {"==", "!="})
                    {
                        MigrateConditions($"'$(Configuration)|$(Platform)' {op} '{oldConfiguration}|AnyCPU'", $"'$(Configuration)|$(Platform)' {op} '{newConfiguration}|AnyCPU'");
                        MigrateConditions($"'$(Configuration)' {op} '{oldConfiguration}'", $"'$(Configuration)' {op} '{newConfiguration}'");
                    }
                }

                MigrateConfigurationConditions("Debug", "ExportDebug");
                MigrateConfigurationConditions("Release", "ExportRelease");
                MigrateConfigurationConditions("Tools", "Debug"); // Must be last
            }


            if (dirty)
                root.Save();
        }
    }
}
