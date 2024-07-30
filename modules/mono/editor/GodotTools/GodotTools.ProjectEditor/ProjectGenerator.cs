using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Build.Construction;
using Microsoft.Build.Evaluation;
using GodotTools.Shared;

namespace GodotTools.ProjectEditor
{
    public static class ProjectGenerator
    {
        public static string GodotSdkAttrValue => $"Godot.NET.Sdk/{GeneratedGodotNupkgsVersions.GodotNETSdk}";

        public static ProjectRootElement GenGameProject(string name, string additionalDefines)
        {
            if (name.Length == 0)
                throw new ArgumentException("Project name is empty.", nameof(name));

            var root = ProjectRootElement.Create(NewProjectFileOptions.None);

            root.Sdk = GodotSdkAttrValue;

            var mainGroup = root.AddPropertyGroup();
            mainGroup.AddProperty("TargetFramework", "net6.0");

            if (!string.IsNullOrWhiteSpace(additionalDefines))
            {
                var defineString = ValidateAdditionalDefines(additionalDefines);

                mainGroup.AddProperty("DefineConstants", $"$(DefineConstants);{defineString}");
            }

            var net7 = mainGroup.AddProperty("TargetFramework", "net7.0");
            net7.Condition = " '$(GodotTargetPlatform)' == 'android' ";

            var net8 = mainGroup.AddProperty("TargetFramework", "net8.0");
            net8.Condition = " '$(GodotTargetPlatform)' == 'ios' ";

            mainGroup.AddProperty("EnableDynamicLoading", "true");

            string sanitizedName = IdentifierUtils.SanitizeQualifiedIdentifier(name, allowEmptyIdentifiers: true);

            // If the name is not a valid namespace, manually set RootNamespace to a sanitized one.
            if (sanitizedName != name)
                mainGroup.AddProperty("RootNamespace", sanitizedName);

            return root;
        }

        public static string GenAndSaveGameProject(string dir, string name, string additionalDefines)
        {
            if (name.Length == 0)
                throw new ArgumentException("Project name is empty.", nameof(name));

            string path = Path.Combine(dir, name + ".csproj");

            var root = GenGameProject(name, additionalDefines);

            // Save (without BOM)
            root.Save(path, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));

            return Guid.NewGuid().ToString().ToUpperInvariant();
        }

        public static string? ValidateAdditionalDefines(string input)
        {
            var defines = input.Split(";").Select(symbol => symbol.Trim()).Where(symbol => !string.IsNullOrEmpty(symbol));

            if (defines.Count() == 0)
                return string.Empty;

            var formatMatcher = new Regex("^[0-9A-Za-z_]+$");

            if (!defines.All(symbol => formatMatcher.IsMatch(symbol)))
                throw new ArgumentException($"Invalid C# defines specification: `{input}`");

            return string.Join(";", defines);
        }
    }
}
