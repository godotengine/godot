using System;
using System.IO;
using Microsoft.Build.Construction;

namespace GodotSharpTools.Project
{
    public static class ProjectUtils
    {
        public static void AddItemToProjectChecked(string projectPath, string itemType, string include)
        {
            var dir = Directory.GetParent(projectPath).FullName;
            var root = ProjectRootElement.Open(projectPath);
            if (root.AddItemChecked(itemType, include.RelativeToPath(dir).Replace("/", "\\")))
                root.Save();
        }
    }
}
