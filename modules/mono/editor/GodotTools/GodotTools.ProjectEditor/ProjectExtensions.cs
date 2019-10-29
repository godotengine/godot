using GodotTools.Core;
using System;
using DotNet.Globbing;
using Microsoft.Build.Construction;

namespace GodotTools.ProjectEditor
{
    public static class ProjectExtensions
    {
        public static bool HasItem(this ProjectRootElement root, string itemType, string include)
        {
            GlobOptions globOptions = new GlobOptions();
            globOptions.Evaluation.CaseInsensitive = false;

            string normalizedInclude = include.NormalizePath();

            foreach (var itemGroup in root.ItemGroups)
            {
                if (itemGroup.Condition.Length != 0)
                    continue;

                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType != itemType)
                        continue;

                    var glob = Glob.Parse(item.Include.NormalizePath(), globOptions);

                    if (glob.IsMatch(normalizedInclude))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        public static bool AddItemChecked(this ProjectRootElement root, string itemType, string include)
        {
            if (!root.HasItem(itemType, include))
            {
                root.AddItem(itemType, include);
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
    }
}
