using System.Collections.Generic;
using System.IO;
using DotNet.Globbing;
using Microsoft.Build.Construction;

namespace GodotSharpTools.Project
{
    public static class ProjectUtils
    {
        public static void AddItemToProjectChecked(string projectPath, string itemType, string include)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            var normalizedInclude = include.RelativeToPath(dir).Replace("/", "\\");

            if (root.AddItemChecked(itemType, normalizedInclude))
                root.Save();
        }

        private static string[] GetAllFilesRecursive(string rootDirectory, string mask)
        {
            string[] files = Directory.GetFiles(rootDirectory, mask, SearchOption.AllDirectories);

            // We want relative paths
            for (int i = 0; i < files.Length; i++) {
                files[i] = files[i].RelativeToPath(rootDirectory);
            }

            return files;
        }

        public static string[] GetIncludeFiles(string projectPath, string itemType)
        {
            var result = new List<string>();
            var existingFiles = GetAllFilesRecursive(Path.GetDirectoryName(projectPath), "*.cs");

            GlobOptions globOptions = new GlobOptions();
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

                    // TODO Check somehow if path has no blog to avoid the following loop...

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
    }
}
