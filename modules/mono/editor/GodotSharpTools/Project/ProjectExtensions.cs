using System;
using Microsoft.Build.Construction;

namespace GodotSharpTools.Project
{
    public static class ProjectExtensions
    {
        public static bool HasItem(this ProjectRootElement root, string itemType, string include)
        {
            string includeNormalized = include.NormalizePath();

            foreach (var itemGroup in root.ItemGroups)
            {
                if (itemGroup.Condition.Length != 0)
                    continue;

                foreach (var item in itemGroup.Items)
                {
                    if (item.ItemType == itemType)
                    {
                        if (item.Include.NormalizePath() == includeNormalized)
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
