using System;
using System.IO;
using Microsoft.Build.Construction;

namespace GodotSharpTools.Project
{
    public static class ProjectGenerator
    {
        public static string GenCoreApiProject(string dir, string[] compileItems)
        {
            string path = Path.Combine(dir, CoreApiProject + ".csproj");

            ProjectPropertyGroupElement mainGroup;
            var root = CreateLibraryProject(CoreApiProject, out mainGroup);

            mainGroup.AddProperty("DocumentationFile", Path.Combine("$(OutputPath)", "$(AssemblyName).xml"));
            mainGroup.SetProperty("RootNamespace", "Godot");

            GenAssemblyInfoFile(root, dir, CoreApiProject,
                                new string[] { "[assembly: InternalsVisibleTo(\"" + EditorApiProject + "\")]" },
                                new string[] { "System.Runtime.CompilerServices" });

            foreach (var item in compileItems)
            {
                root.AddItem("Compile", item.RelativeToPath(dir).Replace("/", "\\"));
            }

            root.Save(path);

            return root.GetGuid().ToString().ToUpper();
        }

        public static string GenEditorApiProject(string dir, string coreApiHintPath, string[] compileItems)
        {
            string path = Path.Combine(dir, EditorApiProject + ".csproj");

            ProjectPropertyGroupElement mainGroup;
            var root = CreateLibraryProject(EditorApiProject, out mainGroup);

            mainGroup.AddProperty("DocumentationFile", Path.Combine("$(OutputPath)", "$(AssemblyName).xml"));
            mainGroup.SetProperty("RootNamespace", "Godot");

            GenAssemblyInfoFile(root, dir, EditorApiProject);

            foreach (var item in compileItems)
            {
                root.AddItem("Compile", item.RelativeToPath(dir).Replace("/", "\\"));
            }

            var coreApiRef = root.AddItem("Reference", CoreApiProject);
            coreApiRef.AddMetadata("HintPath", coreApiHintPath);
            coreApiRef.AddMetadata("Private", "False");

            root.Save(path);

            return root.GetGuid().ToString().ToUpper();
        }

        public static string GenGameProject(string dir, string name, string[] compileItems)
        {
            string path = Path.Combine(dir, name + ".csproj");

            ProjectPropertyGroupElement mainGroup;
            var root = CreateLibraryProject(name, out mainGroup);

            mainGroup.SetProperty("OutputPath", Path.Combine(".mono", "temp", "bin", "$(Configuration)"));
            mainGroup.SetProperty("BaseIntermediateOutputPath", Path.Combine(".mono", "temp", "obj"));
            mainGroup.SetProperty("IntermediateOutputPath", Path.Combine("$(BaseIntermediateOutputPath)", "$(Configuration)"));

            var toolsGroup = root.AddPropertyGroup();
            toolsGroup.Condition = " '$(Configuration)|$(Platform)' == 'Tools|AnyCPU' ";
            toolsGroup.AddProperty("DebugSymbols", "true");
            toolsGroup.AddProperty("DebugType", "full");
            toolsGroup.AddProperty("Optimize", "false");
            toolsGroup.AddProperty("DefineConstants", "DEBUG;TOOLS;");
            toolsGroup.AddProperty("ErrorReport", "prompt");
            toolsGroup.AddProperty("WarningLevel", "4");
            toolsGroup.AddProperty("ConsolePause", "false");

            var coreApiRef = root.AddItem("Reference", CoreApiProject);
            coreApiRef.AddMetadata("HintPath", Path.Combine("$(ProjectDir)", ".mono", "assemblies", CoreApiProject + ".dll"));
            coreApiRef.AddMetadata("Private", "False");

            var editorApiRef = root.AddItem("Reference", EditorApiProject);
            editorApiRef.Condition = " '$(Configuration)' == 'Tools' ";
            editorApiRef.AddMetadata("HintPath", Path.Combine("$(ProjectDir)", ".mono", "assemblies", EditorApiProject + ".dll"));
            editorApiRef.AddMetadata("Private", "False");

            GenAssemblyInfoFile(root, dir, name);

            foreach (var item in compileItems)
            {
                root.AddItem("Compile", item.RelativeToPath(dir).Replace("/", "\\"));
            }

            root.Save(path);

            return root.GetGuid().ToString().ToUpper();
        }

        public static void GenAssemblyInfoFile(ProjectRootElement root, string dir, string name, string[] assemblyLines = null, string[] usingDirectives = null)
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
            {
                foreach (var assemblyLine in assemblyLines)
                    assemblyLinesText += string.Join("\n", assemblyLines) + "\n";
            }

            string content = string.Format(assemblyInfoTemplate, usingDirectivesText, name, assemblyLinesText);

            string assemblyInfoFile = Path.Combine(propertiesDir, "AssemblyInfo.cs");

            File.WriteAllText(assemblyInfoFile, content);

            root.AddItem("Compile", assemblyInfoFile.RelativeToPath(dir).Replace("/", "\\"));
        }

        public static ProjectRootElement CreateLibraryProject(string name, out ProjectPropertyGroupElement mainGroup)
        {
            var root = ProjectRootElement.Create();
            root.DefaultTargets = "Build";

            mainGroup = root.AddPropertyGroup();
            mainGroup.AddProperty("Configuration", "Debug").Condition = " '$(Configuration)' == '' ";
            mainGroup.AddProperty("Platform", "AnyCPU").Condition = " '$(Platform)' == '' ";
            mainGroup.AddProperty("ProjectGuid", "{" + Guid.NewGuid().ToString().ToUpper() + "}");
            mainGroup.AddProperty("OutputType", "Library");
            mainGroup.AddProperty("OutputPath", Path.Combine("bin", "$(Configuration)"));
            mainGroup.AddProperty("RootNamespace", name);
            mainGroup.AddProperty("AssemblyName", name);
            mainGroup.AddProperty("TargetFrameworkVersion", "v4.5");

            var debugGroup = root.AddPropertyGroup();
            debugGroup.Condition = " '$(Configuration)|$(Platform)' == 'Debug|AnyCPU' ";
            debugGroup.AddProperty("DebugSymbols", "true");
            debugGroup.AddProperty("DebugType", "full");
            debugGroup.AddProperty("Optimize", "false");
            debugGroup.AddProperty("DefineConstants", "DEBUG;");
            debugGroup.AddProperty("ErrorReport", "prompt");
            debugGroup.AddProperty("WarningLevel", "4");
            debugGroup.AddProperty("ConsolePause", "false");

            var releaseGroup = root.AddPropertyGroup();
            releaseGroup.Condition = " '$(Configuration)|$(Platform)' == 'Release|AnyCPU' ";
            releaseGroup.AddProperty("DebugType", "full");
            releaseGroup.AddProperty("Optimize", "true");
            releaseGroup.AddProperty("ErrorReport", "prompt");
            releaseGroup.AddProperty("WarningLevel", "4");
            releaseGroup.AddProperty("ConsolePause", "false");

            // References
            var referenceGroup = root.AddItemGroup();
            referenceGroup.AddItem("Reference", "System");

            root.AddImport(Path.Combine("$(MSBuildBinPath)", "Microsoft.CSharp.targets").Replace("/", "\\"));

            return root;
        }

        private static void AddItems(ProjectRootElement elem, string groupName, params string[] items)
        {
            var group = elem.AddItemGroup();

            foreach (var item in items)
            {
                group.AddItem(groupName, item);
            }
        }

        public const string CoreApiProject = "GodotSharp";
        public const string EditorApiProject = "GodotSharpEditor";

        private const string assemblyInfoTemplate =
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
