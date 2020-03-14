using GodotTools.Core;
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Microsoft.Build.Construction;

namespace GodotTools.ProjectEditor
{
    public static class ProjectGenerator
    {
        private const string CoreApiProjectName = "GodotSharp";
        private const string EditorApiProjectName = "GodotSharpEditor";

        public static string GenGameProject(string dir, string name, IEnumerable<string> compileItems)
        {
            string path = Path.Combine(dir, name + ".csproj");

            ProjectPropertyGroupElement mainGroup;
            var root = CreateLibraryProject(name, "Debug", out mainGroup);

            mainGroup.SetProperty("OutputPath", Path.Combine(".mono", "temp", "bin", "$(Configuration)"));
            mainGroup.SetProperty("BaseIntermediateOutputPath", Path.Combine(".mono", "temp", "obj"));
            mainGroup.SetProperty("IntermediateOutputPath", Path.Combine("$(BaseIntermediateOutputPath)", "$(Configuration)"));
            mainGroup.SetProperty("ApiConfiguration", "Debug").Condition = " '$(Configuration)' != 'ExportRelease' ";
            mainGroup.SetProperty("ApiConfiguration", "Release").Condition = " '$(Configuration)' == 'ExportRelease' ";

            var debugGroup = root.AddPropertyGroup();
            debugGroup.Condition = " '$(Configuration)|$(Platform)' == 'Debug|AnyCPU' ";
            debugGroup.AddProperty("DebugSymbols", "true");
            debugGroup.AddProperty("DebugType", "portable");
            debugGroup.AddProperty("Optimize", "false");
            debugGroup.AddProperty("DefineConstants", "$(GodotDefineConstants);GODOT;DEBUG;TOOLS;");
            debugGroup.AddProperty("ErrorReport", "prompt");
            debugGroup.AddProperty("WarningLevel", "4");
            debugGroup.AddProperty("ConsolePause", "false");

            var coreApiRef = root.AddItem("Reference", CoreApiProjectName);
            coreApiRef.AddMetadata("HintPath", Path.Combine("$(ProjectDir)", ".mono", "assemblies", "$(ApiConfiguration)", CoreApiProjectName + ".dll"));
            coreApiRef.AddMetadata("Private", "False");

            var editorApiRef = root.AddItem("Reference", EditorApiProjectName);
            editorApiRef.Condition = " '$(Configuration)' == 'Debug' ";
            editorApiRef.AddMetadata("HintPath", Path.Combine("$(ProjectDir)", ".mono", "assemblies", "$(ApiConfiguration)", EditorApiProjectName + ".dll"));
            editorApiRef.AddMetadata("Private", "False");

            GenAssemblyInfoFile(root, dir, name);

            foreach (var item in compileItems)
            {
                root.AddItem("Compile", item.RelativeToPath(dir).Replace("/", "\\"));
            }

            root.Save(path);

            return root.GetGuid().ToString().ToUpper();
        }

        private static void GenAssemblyInfoFile(ProjectRootElement root, string dir, string name, string[] assemblyLines = null, string[] usingDirectives = null)
        {
            string propertiesDir = Path.Combine(dir, "Properties");
            if (!Directory.Exists(propertiesDir))
                Directory.CreateDirectory(propertiesDir);

            string usingDirectivesText = string.Empty;

            if (usingDirectives != null)
            {
                foreach (var usingDirective in usingDirectives)
                    usingDirectivesText += "\nusing " + usingDirective + ";";
            }

            string assemblyLinesText = string.Empty;

            if (assemblyLines != null)
                assemblyLinesText += string.Join("\n", assemblyLines) + "\n";

            string content = string.Format(AssemblyInfoTemplate, usingDirectivesText, name, assemblyLinesText);

            string assemblyInfoFile = Path.Combine(propertiesDir, "AssemblyInfo.cs");

            File.WriteAllText(assemblyInfoFile, content);

            root.AddItem("Compile", assemblyInfoFile.RelativeToPath(dir).Replace("/", "\\"));
        }

        public static ProjectRootElement CreateLibraryProject(string name, string defaultConfig, out ProjectPropertyGroupElement mainGroup)
        {
            if (string.IsNullOrEmpty(name))
                throw new ArgumentException($"{nameof(name)} cannot be empty", nameof(name));

            var root = ProjectRootElement.Create();
            root.DefaultTargets = "Build";

            mainGroup = root.AddPropertyGroup();
            mainGroup.AddProperty("Configuration", defaultConfig).Condition = " '$(Configuration)' == '' ";
            mainGroup.AddProperty("Platform", "AnyCPU").Condition = " '$(Platform)' == '' ";
            mainGroup.AddProperty("ProjectGuid", "{" + Guid.NewGuid().ToString().ToUpper() + "}");
            mainGroup.AddProperty("OutputType", "Library");
            mainGroup.AddProperty("OutputPath", Path.Combine("bin", "$(Configuration)"));
            mainGroup.AddProperty("RootNamespace", IdentifierUtils.SanitizeQualifiedIdentifier(name, allowEmptyIdentifiers: true));
            mainGroup.AddProperty("AssemblyName", name);
            mainGroup.AddProperty("TargetFrameworkVersion", "v4.7");
            mainGroup.AddProperty("GodotProjectGeneratorVersion", Assembly.GetExecutingAssembly().GetName().Version.ToString());

            var exportDebugGroup = root.AddPropertyGroup();
            exportDebugGroup.Condition = " '$(Configuration)|$(Platform)' == 'ExportDebug|AnyCPU' ";
            exportDebugGroup.AddProperty("DebugSymbols", "true");
            exportDebugGroup.AddProperty("DebugType", "portable");
            exportDebugGroup.AddProperty("Optimize", "false");
            exportDebugGroup.AddProperty("DefineConstants", "$(GodotDefineConstants);GODOT;DEBUG;");
            exportDebugGroup.AddProperty("ErrorReport", "prompt");
            exportDebugGroup.AddProperty("WarningLevel", "4");
            exportDebugGroup.AddProperty("ConsolePause", "false");

            var exportReleaseGroup = root.AddPropertyGroup();
            exportReleaseGroup.Condition = " '$(Configuration)|$(Platform)' == 'ExportRelease|AnyCPU' ";
            exportReleaseGroup.AddProperty("DebugType", "portable");
            exportReleaseGroup.AddProperty("Optimize", "true");
            exportReleaseGroup.AddProperty("DefineConstants", "$(GodotDefineConstants);GODOT;");
            exportReleaseGroup.AddProperty("ErrorReport", "prompt");
            exportReleaseGroup.AddProperty("WarningLevel", "4");
            exportReleaseGroup.AddProperty("ConsolePause", "false");

            // References
            var referenceGroup = root.AddItemGroup();
            referenceGroup.AddItem("Reference", "System");

            root.AddImport(Path.Combine("$(MSBuildBinPath)", "Microsoft.CSharp.targets").Replace("/", "\\"));

            return root;
        }

        private const string AssemblyInfoTemplate =
            @"using System.Reflection;{0}

// Information about this assembly is defined by the following attributes.
// Change them to the values specific to your project.

[assembly: AssemblyTitle(""{1}"")]
[assembly: AssemblyDescription("""")]
[assembly: AssemblyConfiguration("""")]
[assembly: AssemblyCompany("""")]
[assembly: AssemblyProduct("""")]
[assembly: AssemblyCopyright("""")]
[assembly: AssemblyTrademark("""")]
[assembly: AssemblyCulture("""")]

// The assembly version has the format ""{{Major}}.{{Minor}}.{{Build}}.{{Revision}}"".
// The form ""{{Major}}.{{Minor}}.*"" will automatically update the build and revision,
// and ""{{Major}}.{{Minor}}.{{Build}}.*"" will update just the revision.

[assembly: AssemblyVersion(""1.0.*"")]

// The following attributes are used to specify the signing key for the assembly,
// if desired. See the Mono documentation for more information about signing.

//[assembly: AssemblyDelaySign(false)]
//[assembly: AssemblyKeyFile("""")]
{2}";
    }
}
