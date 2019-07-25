using GodotTools.Core;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
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
                    .Any(g => (g.Condition == string.Empty || g.Condition == condition) &&
                              g.Properties
                                  .Any(p => p.Name == name &&
                                            p.Value == value &&
                                            (p.Condition == condition || g.Condition == condition))))
                {
                    return;
                }

                root.AddProperty(name, value).Condition = condition;
                dirty = true;
            }

            AddPropertyIfNotPresent(name: "ApiConfiguration",
                condition: " '$(Configuration)' != 'Release' ",
                value: "Debug");
            AddPropertyIfNotPresent(name: "ApiConfiguration",
                condition: " '$(Configuration)' == 'Release' ",
                value: "Release");

            void SetReferenceHintPath(string referenceName, string condition, string hintPath)
            {
                foreach (var itemGroup in root.ItemGroups.Where(g =>
                    g.Condition == string.Empty || g.Condition == condition))
                {
                    var references = itemGroup.Items.Where(item =>
                        item.ItemType == "Reference" &&
                        item.Include == referenceName &&
                        (item.Condition == condition || itemGroup.Condition == condition));

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
                root.AddItem("Reference", referenceName).Condition = condition;
                dirty = true;
            }

            const string coreProjectName = "GodotSharp";
            const string editorProjectName = "GodotSharpEditor";

            const string coreCondition = "";
            const string editorCondition = " '$(Configuration)' == 'Tools' ";

            var coreHintPath = $"$(ProjectDir)/.mono/assemblies/$(ApiConfiguration)/{coreProjectName}.dll";
            var editorHintPath = $"$(ProjectDir)/.mono/assemblies/$(ApiConfiguration)/{editorProjectName}.dll";

            SetReferenceHintPath(coreProjectName, coreCondition, coreHintPath);
            SetReferenceHintPath(editorProjectName, editorCondition, editorHintPath);

            if (dirty)
                root.Save();
        }
    }
}
