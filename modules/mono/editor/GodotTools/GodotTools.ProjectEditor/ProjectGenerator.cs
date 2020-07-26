using System;
using System.IO;
using Microsoft.Build.Construction;
using Microsoft.Build.Evaluation;

namespace GodotTools.ProjectEditor
{
    public static class ProjectGenerator
    {
        public const string GodotSdkVersionToUse = "4.0.0-dev2";

        public static string GodotSdkAttrValue => $"Godot.NET.Sdk/{GodotSdkVersionToUse}";

        public static ProjectRootElement GenGameProject(string name)
        {
            if (name.Length == 0)
                throw new ArgumentException("Project name is empty", nameof(name));

            var root = ProjectRootElement.Create(NewProjectFileOptions.None);

            root.Sdk = GodotSdkAttrValue;

            var mainGroup = root.AddPropertyGroup();
            mainGroup.AddProperty("TargetFramework", "netstandard2.1");

            string sanitizedName = IdentifierUtils.SanitizeQualifiedIdentifier(name, allowEmptyIdentifiers: true);

            // If the name is not a valid namespace, manually set RootNamespace to a sanitized one.
            if (sanitizedName != name)
                mainGroup.AddProperty("RootNamespace", sanitizedName);

            return root;
        }

        public static string GenAndSaveGameProject(string dir, string name)
        {
            if (name.Length == 0)
                throw new ArgumentException("Project name is empty", nameof(name));

            string path = Path.Combine(dir, name + ".csproj");

            var root = GenGameProject(name);

            root.Save(path);

            return Guid.NewGuid().ToString().ToUpper();
        }
    }
}
