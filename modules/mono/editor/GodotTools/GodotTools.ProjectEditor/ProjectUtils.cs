using System;
using GodotTools.Core;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Xml;
using System.Xml.Linq;
using JetBrains.Annotations;
using Microsoft.Build.Construction;
using Microsoft.Build.Globbing;
using Semver;

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

        [PublicAPI]
        public static void AddItemToProjectChecked(string projectPath, string itemType, string include)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            if (root.AreDefaultCompileItemsEnabled())
            {
                // No need to add. It's already included automatically by the MSBuild Sdk.
                // This assumes the source file is inside the project directory and not manually excluded in the csproj
                return;
            }

            var normalizedInclude = include.RelativeToPath(dir).Replace("/", "\\");

            if (root.AddItemChecked(itemType, normalizedInclude))
                root.Save();
        }

        public static void RenameItemInProjectChecked(string projectPath, string itemType, string oldInclude, string newInclude)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            if (root.AreDefaultCompileItemsEnabled())
            {
                // No need to add. It's already included automatically by the MSBuild Sdk.
                // This assumes the source file is inside the project directory and not manually excluded in the csproj
                return;
            }

            var normalizedOldInclude = oldInclude.NormalizePath();
            var normalizedNewInclude = newInclude.NormalizePath();

            var item = root.FindItemOrNullAbs(itemType, normalizedOldInclude);

            if (item == null)
                return;

            // Check if the found item include already matches the new path
            var glob = MSBuildGlob.Parse(item.Include);
            if (glob.IsMatch(normalizedNewInclude))
                return;

            // Otherwise, if the item include uses globbing it's better to add a new item instead of modifying
            if (!string.IsNullOrEmpty(glob.WildcardDirectoryPart) || glob.FilenamePart.Contains("*"))
            {
                root.AddItem(itemType, normalizedNewInclude.RelativeToPath(dir).Replace("/", "\\"));
                root.Save();
                return;
            }

            item.Include = normalizedNewInclude.RelativeToPath(dir).Replace("/", "\\");
            root.Save();
        }

        public static void RemoveItemFromProjectChecked(string projectPath, string itemType, string include)
        {
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            if (root.AreDefaultCompileItemsEnabled())
            {
                // No need to add. It's already included automatically by the MSBuild Sdk.
                // This assumes the source file is inside the project directory and not manually excluded in the csproj
                return;
            }

            var normalizedInclude = include.NormalizePath();

            var item = root.FindItemOrNullAbs(itemType, normalizedInclude);

            // Couldn't find an existing item that matches to remove
            if (item == null)
                return;

            var glob = MSBuildGlob.Parse(item.Include);

            // If the item include uses globbing don't remove it
            if (!string.IsNullOrEmpty(glob.WildcardDirectoryPart) || glob.FilenamePart.Contains("*"))
            {
                return;
            }

            item.Parent.RemoveChild(item);
            root.Save();
        }

        public static void RenameItemsToNewFolderInProjectChecked(string projectPath, string itemType, string oldFolder, string newFolder)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            if (root.AreDefaultCompileItemsEnabled())
            {
                // No need to add. It's already included automatically by the MSBuild Sdk.
                // This assumes the source file is inside the project directory and not manually excluded in the csproj
                return;
            }

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

            if (root.AreDefaultCompileItemsEnabled())
            {
                // No need to add. It's already included automatically by the MSBuild Sdk.
                // This assumes the source file is inside the project directory and not manually excluded in the csproj
                return;
            }

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

            var root = ProjectRootElement.Open(projectPath);
            Debug.Assert(root != null);

            if (root.AreDefaultCompileItemsEnabled())
            {
                var excluded = new List<string>();
                result.AddRange(existingFiles);

                foreach (var item in root.Items)
                {
                    if (string.IsNullOrEmpty(item.Condition))
                        continue;

                    if (item.ItemType != itemType)
                        continue;


                    string normalizedRemove = item.Remove.NormalizePath();

                    var glob = MSBuildGlob.Parse(normalizedRemove);

                    excluded.AddRange(result.Where(includedFile => glob.IsMatch(includedFile)));
                }

                result.RemoveAll(f => excluded.Contains(f));
            }

            foreach (var itemGroup in root.ItemGroups)
            {
                if (itemGroup.Condition.Length != 0)
                    continue;

                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType != itemType)
                        continue;

                    string normalizedInclude = item.Include.NormalizePath();

                    var glob = MSBuildGlob.Parse(normalizedInclude);

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

        public static void MigrateToProjectSdksStyle(MSBuildProject project, string projectName)
        {
            var root = project.Root;

            if (!string.IsNullOrEmpty(root.Sdk))
                return;

            root.Sdk = $"{ProjectGenerator.GodotSdkNameToUse}/{ProjectGenerator.GodotSdkVersionToUse}";

            root.ToolsVersion = null;
            root.DefaultTargets = null;

            root.AddProperty("TargetFramework", "net472");

            // Remove obsolete properties, items and elements. We're going to be conservative
            // here to minimize the chances of introducing breaking changes. As such we will
            // only remove elements that could potentially cause issues with the Godot.NET.Sdk.

            void RemoveElements(IEnumerable<ProjectElement> elements)
            {
                foreach (var element in elements)
                    element.Parent.RemoveChild(element);
            }

            // Default Configuration

            RemoveElements(root.PropertyGroups.SelectMany(g => g.Properties)
                .Where(p => p.Name == "Configuration" && p.Condition.Trim() == "'$(Configuration)' == ''" && p.Value == "Debug"));

            // Default Platform

            RemoveElements(root.PropertyGroups.SelectMany(g => g.Properties)
                .Where(p => p.Name == "Platform" && p.Condition.Trim() == "'$(Platform)' == ''" && p.Value == "AnyCPU"));

            // Simple properties

            var yabaiProperties = new[]
            {
                "OutputPath",
                "BaseIntermediateOutputPath",
                "IntermediateOutputPath",
                "TargetFrameworkVersion",
                "ProjectTypeGuids",
                "ApiConfiguration"
            };

            RemoveElements(root.PropertyGroups.SelectMany(g => g.Properties)
                .Where(p => yabaiProperties.Contains(p.Name)));

            // Configuration dependent properties

            var yabaiPropertiesForConfigs = new[]
            {
                "DebugSymbols",
                "DebugType",
                "Optimize",
                "DefineConstants",
                "ErrorReport",
                "WarningLevel",
                "ConsolePause"
            };

            var configNames = new[]
            {
                "ExportDebug", "ExportRelease", "Debug",
                "Tools", "Release" // Include old config names as well in case it's upgrading from 3.2.1 or older
            };

            foreach (var config in configNames)
            {
                var group = root.PropertyGroups
                    .FirstOrDefault(g => g.Condition.Trim() == $"'$(Configuration)|$(Platform)' == '{config}|AnyCPU'");

                if (group == null)
                    continue;

                RemoveElements(group.Properties.Where(p => yabaiPropertiesForConfigs.Contains(p.Name)));

                if (group.Count == 0)
                {
                    // No more children, safe to delete the group
                    group.Parent.RemoveChild(group);
                }
            }

            // Godot API References

            var apiAssemblies = new[] { ApiAssemblyNames.Core, ApiAssemblyNames.Editor };

            RemoveElements(root.ItemGroups.SelectMany(g => g.Items)
                .Where(i => i.ItemType == "Reference" && apiAssemblies.Contains(i.Include)));

            // Microsoft.NETFramework.ReferenceAssemblies PackageReference

            RemoveElements(root.ItemGroups.SelectMany(g => g.Items).Where(i =>
                i.ItemType == "PackageReference" &&
                i.Include.Equals("Microsoft.NETFramework.ReferenceAssemblies", StringComparison.OrdinalIgnoreCase)));

            // Imports

            var yabaiImports = new[]
            {
                "$(MSBuildBinPath)/Microsoft.CSharp.targets",
                "$(MSBuildBinPath)Microsoft.CSharp.targets"
            };

            RemoveElements(root.Imports.Where(import => yabaiImports.Contains(
                import.Project.Replace("\\", "/").Replace("//", "/"))));

            // 'EnableDefaultCompileItems' and 'GenerateAssemblyInfo' are kept enabled by default
            // on new projects, but when migrating old projects we disable them to avoid errors.
            root.AddProperty("EnableDefaultCompileItems", "false");
            root.AddProperty("GenerateAssemblyInfo", "false");

            // Older AssemblyInfo.cs cause the following error:
            // 'Properties/AssemblyInfo.cs(19,28): error CS8357:
            // The specified version string contains wildcards, which are not compatible with determinism.
            // Either remove wildcards from the version string, or disable determinism for this compilation.'
            // We disable deterministic builds to prevent this. The user can then fix this manually when desired
            // by fixing 'AssemblyVersion("1.0.*")' to not use wildcards.
            root.AddProperty("Deterministic", "false");

            project.HasUnsavedChanges = true;

            var xDoc = XDocument.Parse(root.RawXml);

            if (xDoc.Root == null)
                return; // Too bad, we will have to keep the xmlns/namespace and xml declaration

            XElement GetElement(XDocument doc, string name, string value, string parentName)
            {
                foreach (var node in doc.DescendantNodes())
                {
                    if (!(node is XElement element))
                        continue;
                    if (element.Name.LocalName.Equals(name) && element.Value == value &&
                        element.Parent != null && element.Parent.Name.LocalName.Equals(parentName))
                    {
                        return element;
                    }
                }

                return null;
            }

            // Add comment about Microsoft.NET.Sdk properties disabled during migration

            GetElement(xDoc, name: "EnableDefaultCompileItems", value: "false", parentName: "PropertyGroup")
                .AddBeforeSelf(new XComment("The following properties were overridden during migration to prevent errors.\n" +
                                            "    Enabling them may require other manual changes to the project and its files."));

            void RemoveNamespace(XElement element)
            {
                element.Attributes().Where(x => x.IsNamespaceDeclaration).Remove();
                element.Name = element.Name.LocalName;

                foreach (var node in element.DescendantNodes())
                {
                    if (node is XElement xElement)
                    {
                        // Need to do the same for all children recursively as it adds it to them for some reason...
                        RemoveNamespace(xElement);
                    }
                }
            }

            // Remove xmlns/namespace
            RemoveNamespace(xDoc.Root);

            // Remove xml declaration
            xDoc.Nodes().FirstOrDefault(node => node.NodeType == XmlNodeType.XmlDeclaration)?.Remove();

            string projectFullPath = root.FullPath;

            root = ProjectRootElement.Create(xDoc.CreateReader());
            root.FullPath = projectFullPath;

            project.Root = root;
        }

        public static void EnsureGodotSdkIsUpToDate(MSBuildProject project)
        {
            string godotSdkAttrValue = $"{ProjectGenerator.GodotSdkNameToUse}/{ProjectGenerator.GodotSdkVersionToUse}";

            var root = project.Root;
            string rootSdk = root.Sdk?.Trim();

            if (!string.IsNullOrEmpty(rootSdk))
            {
                // Check if the version is already the same.
                if (rootSdk.Equals(godotSdkAttrValue, StringComparison.OrdinalIgnoreCase))
                    return;

                // We also allow higher versions as long as the major and minor are the same.
                var semVerToUse = SemVersion.Parse(ProjectGenerator.GodotSdkVersionToUse);
                var godotSdkAttrLaxValueRegex = new Regex($@"^{ProjectGenerator.GodotSdkNameToUse}/(?<ver>.*)$");

                var match = godotSdkAttrLaxValueRegex.Match(rootSdk);

                if (match.Success &&
                    SemVersion.TryParse(match.Groups["ver"].Value, out var semVerDetected) &&
                    semVerDetected.Major == semVerToUse.Major &&
                    semVerDetected.Minor == semVerToUse.Minor &&
                    semVerDetected > semVerToUse)
                {
                    return;
                }
            }

            root.Sdk = godotSdkAttrValue;
            project.HasUnsavedChanges = true;
        }
    }
}
