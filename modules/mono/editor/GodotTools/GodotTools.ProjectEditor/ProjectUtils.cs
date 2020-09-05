using System;
using GodotTools.Core;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Microsoft.Build.Construction;
using Microsoft.Build.Globbing;

namespace GodotTools.ProjectEditor
{
    public sealed class MSBuildProject
    {
        internal ProjectRootElement Root { get; set; }

        public bool HasUnsavedChanges { get; set; }

        public void Save() => Root.Save();

        public MSBuildProject(ProjectRootElement root)
        {
            Root = root;
        }
    }

    public static class ProjectUtils
    {
        public static MSBuildProject Open(string path)
        {
            var root = ProjectRootElement.Open(path);
            return root != null ? new MSBuildProject(root) : null;
        }

        private static List<string> GetAllFilesRecursive(string rootDirectory, string mask)
        {
            string[] files = Directory.GetFiles(rootDirectory, mask, SearchOption.AllDirectories);

            // We want relative paths
            for (int i = 0; i < files.Length; i++)
            {
                files[i] = files[i].RelativeToPath(rootDirectory);
            }

            return new List<string>(files);
        }

        // NOTE: Assumes auto-including items. Only used by the scripts metadata generator, which will be replaced with source generators in the future.
        public static IEnumerable<string> GetIncludeFiles(string projectPath, string itemType)
        {
            var excluded = new List<string>();
            var includedFiles = GetAllFilesRecursive(Path.GetDirectoryName(projectPath), "*.cs");

            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            foreach (var item in root.Items)
            {
                if (string.IsNullOrEmpty(item.Condition))
                    continue;

                if (item.ItemType != itemType)
                    continue;

                string normalizedRemove = item.Remove.NormalizePath();

                var glob = MSBuildGlob.Parse(normalizedRemove);
                excluded.AddRange(includedFiles.Where(includedFile => glob.IsMatch(includedFile)));
            }

            includedFiles.RemoveAll(f => excluded.Contains(f));

            return includedFiles;
        }

        public static void MigrateToProjectSdksStyle(MSBuildProject project, string projectName)
        {
            var origRoot = project.Root;

            if (!string.IsNullOrEmpty(origRoot.Sdk))
                return;

            project.Root = ProjectGenerator.GenGameProject(projectName);
            project.Root.FullPath = origRoot.FullPath;
            project.HasUnsavedChanges = true;
        }

        public static void EnsureGodotSdkIsUpToDate(MSBuildProject project)
        {
            var root = project.Root;
            string godotSdkAttrValue = ProjectGenerator.GodotSdkAttrValue;

            if (!string.IsNullOrEmpty(root.Sdk) && root.Sdk.Trim().Equals(godotSdkAttrValue, StringComparison.OrdinalIgnoreCase))
                return;

            root.Sdk = godotSdkAttrValue;
            project.HasUnsavedChanges = true;
        }
    }
}
