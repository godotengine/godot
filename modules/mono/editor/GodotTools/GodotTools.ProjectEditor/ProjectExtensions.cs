using GodotTools.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Build.Construction;
using Microsoft.Build.Globbing;

namespace GodotTools.ProjectEditor
{
    public static class ProjectExtensions
    {
        public static ProjectItemElement FindItemOrNull(this ProjectRootElement root, string itemType, string include, bool noCondition = false)
        {
            string normalizedInclude = include.NormalizePath();

            foreach (var itemGroup in root.ItemGroups)
            {
                if (noCondition && itemGroup.Condition.Length != 0)
                    continue;

                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType != itemType)
                        continue;

                    var glob = MSBuildGlob.Parse(item.Include.NormalizePath());

                    if (glob.IsMatch(normalizedInclude))
                        return item;
                }
            }

            return null;
        }

        public static ProjectItemElement FindItemOrNullAbs(this ProjectRootElement root, string itemType, string include, bool noCondition = false)
        {
            string normalizedInclude = Path.GetFullPath(include).NormalizePath();

            foreach (var itemGroup in root.ItemGroups)
            {
                if (noCondition && itemGroup.Condition.Length != 0)
                    continue;

                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType != itemType)
                        continue;

                    var glob = MSBuildGlob.Parse(Path.GetFullPath(item.Include).NormalizePath());

                    if (glob.IsMatch(normalizedInclude))
                        return item;
                }
            }

            return null;
        }

        public static IEnumerable<ProjectItemElement> FindAllItemsInFolder(this ProjectRootElement root, string itemType, string folder)
        {
            string absFolderNormalizedWithSep = Path.GetFullPath(folder).NormalizePath() + Path.DirectorySeparatorChar;

            foreach (var itemGroup in root.ItemGroups)
            {
                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType != itemType)
                        continue;

                    string absPathNormalized = Path.GetFullPath(item.Include).NormalizePath();

                    if (absPathNormalized.StartsWith(absFolderNormalizedWithSep))
                        yield return item;
                }
            }
        }

        public static bool HasItem(this ProjectRootElement root, string itemType, string include, bool noCondition = false)
        {
            return root.FindItemOrNull(itemType, include, noCondition) != null;
        }

        public static bool AddItemChecked(this ProjectRootElement root, string itemType, string include)
        {
            if (!root.HasItem(itemType, include, noCondition: true))
            {
                root.AddItem(itemType, include);
                return true;
            }

            return false;
        }

        public static bool RemoveItemChecked(this ProjectRootElement root, string itemType, string include)
        {
            var item = root.FindItemOrNullAbs(itemType, include);
            if (item != null)
            {
                item.Parent.RemoveChild(item);
                return true;
            }

            return false;
        }

        public static Guid GetGuid(this ProjectRootElement root)
        {
            foreach (var property in root.Properties)
            {
                if (property.Name == "ProjectGuid")
                    return Guid.Parse(property.Value);
            }

            return Guid.Empty;
        }

        public static bool AreDefaultCompileItemsEnabled(this ProjectRootElement root)
        {
            var enableDefaultCompileItemsProps = root.PropertyGroups
                .Where(g => string.IsNullOrEmpty(g.Condition))
                .SelectMany(g => g.Properties
                    .Where(p => p.Name == "EnableDefaultCompileItems" && string.IsNullOrEmpty(p.Condition)));

            bool enableDefaultCompileItems = true;
            foreach (var prop in enableDefaultCompileItemsProps)
                enableDefaultCompileItems = prop.Value.Equals("true", StringComparison.OrdinalIgnoreCase);

            return enableDefaultCompileItems;
        }
    }
}
