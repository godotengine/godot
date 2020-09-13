using System;
using System.IO;
using System.Text;
using Microsoft.Build.Construction;
using Microsoft.Build.Evaluation;

namespace GodotTools.ProjectEditor
{
    public static class ProjectGenerator
    {
<<<<<<< HEAD
        public const string GodotSdkVersionToUse = "4.0.0-dev2";
=======
        public const string GodotSdkVersionToUse = "3.2.3";
>>>>>>> audio-bus-effect-fixed

        public static string GodotSdkAttrValue => $"Godot.NET.Sdk/{GodotSdkVersionToUse}";

        public static ProjectRootElement GenGameProject(string name)
        {
            if (name.Length == 0)
                throw new ArgumentException("Project name is empty", nameof(name));

            var root = ProjectRootElement.Create(NewProjectFileOptions.None);

            root.Sdk = GodotSdkAttrValue;

            var mainGroup = root.AddPropertyGroup();
<<<<<<< HEAD
            mainGroup.AddProperty("TargetFramework", "netstandard2.1");
=======
            mainGroup.AddProperty("TargetFramework", "net472");
>>>>>>> audio-bus-effect-fixed

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

            // Save (without BOM)
            root.Save(path, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            return Guid.NewGuid().ToString().ToUpper();
        }
    }
}
