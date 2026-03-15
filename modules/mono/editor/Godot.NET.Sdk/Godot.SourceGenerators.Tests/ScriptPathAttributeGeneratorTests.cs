using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Testing;
using Microsoft.CodeAnalysis.Testing;
using Microsoft.CodeAnalysis.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPathAttributeGeneratorTests
{
    private static (string, SourceText) MakeAssemblyScriptTypesGeneratedSource(ICollection<string> types)
    {
        return (
            Path.Combine("Godot.SourceGenerators", "Godot.SourceGenerators.ScriptPathAttributeGenerator", "AssemblyScriptTypes.generated.cs"),
            SourceText.From($$"""
            [assembly:Godot.AssemblyHasScriptsAttribute(new System.Type[] {{{string.Join(", ", types.Select(type => $"typeof({type})"))}}})]

            """, Encoding.UTF8)
        );
    }

    /// <summary>
    /// Creates a verifier with a custom GodotProjectDir, for testing out-of-tree source paths.
    /// </summary>
    private static CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.Test MakeVerifierWithCustomProjectDir(
        string godotProjectDir, string assemblyName = "TestProject")
    {
        var verifier = new CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.Test();

        verifier.TestState.AnalyzerConfigFiles.Add(("/.globalconfig", $"""
        is_global = true
        build_property.GodotProjectDir = {godotProjectDir}
        """));

        verifier.SolutionTransforms.Add((Solution solution, ProjectId projectId) =>
        {
            Project project = solution.GetProject(projectId)!
                .WithAssemblyName(assemblyName);
            return project.Solution;
        });

        return verifier;
    }

    [Fact]
    public async Task ScriptBoilerplate()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "ScriptBoilerplate.cs" },
            new string[] { "ScriptBoilerplate_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::ScriptBoilerplate" }));
        await verifier.RunAsync();
    }

    [Fact]
    public async Task FooBar()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "Foo.cs", "Bar.cs" },
            new string[] { "Foo_ScriptPath.generated.cs", "Bar_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::Foo", "global::Bar" }));
        await verifier.RunAsync();
    }

    [Fact]
    public async Task Generic()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "Generic.cs" },
            new string[] { "Generic(Of T)_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::Generic<>" }));
        await verifier.RunAsync();
    }

    [Fact]
    public async Task GenericMultipleClassesSameName()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            Array.Empty<string>(),
            new string[] { "Generic(Of T)_ScriptPath.generated.cs" }
        );
        verifier.TestState.Sources.Add(("Generic.cs", File.ReadAllText(Path.Combine(Constants.SourceFolderPath, "Generic.GD0003.cs"))));
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::Generic<>", "global::Generic<,>", "global::Generic" }));
        await verifier.RunAsync();
    }

    [Fact]
    public async Task NamespaceMultipleClassesSameName()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            Array.Empty<string>(),
            new string[] { "NamespaceA.SameName_ScriptPath.generated.cs" }
        );
        verifier.TestState.Sources.Add(("SameName.cs", File.ReadAllText(Path.Combine(Constants.SourceFolderPath, "SameName.GD0003.cs"))));
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::NamespaceA.SameName", "global::NamespaceB.SameName" }));
        await verifier.RunAsync();
    }

    /// <summary>
    /// Tests that an in-tree external assembly script (source inside Godot project dir)
    /// gets a normal res:// path.
    /// </summary>
    [Fact]
    public async Task ExternalAssemblyInTree()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "ExternalScript.cs" },
            new string[] { "ExternalModule.ExternalScript_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(
            MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::ExternalModule.ExternalScript" }));
        await verifier.RunAsync();
    }

    /// <summary>
    /// Tests that an out-of-tree external assembly script (source outside Godot project dir)
    /// gets a synthetic csharp:// path instead of an invalid res://../ path.
    /// </summary>
    [Fact]
    public async Task ExternalAssemblyOutOfTree()
    {
        // Set GodotProjectDir to a subdirectory that won't contain the source file.
        // The Roslyn test framework resolves source paths relative to CWD. By setting
        // GodotProjectDir to a non-existent subdirectory, the computed relative path
        // will start with "../", triggering the csharp:// synthetic path logic.
        string deepProjectDir = Path.Combine(Constants.ExecutingAssemblyPath, "nonexistent", "godot_project");

        var verifier = MakeVerifierWithCustomProjectDir(deepProjectDir, "ExternalLib");

        string source = File.ReadAllText(Path.Combine(Constants.SourceFolderPath, "ExternalScript.cs"));
        verifier.TestState.Sources.Add(("ExternalScript.cs", SourceText.From(source)));

        // Skip exact source check because SourceFile contains a machine-dependent absolute path.
        verifier.TestBehaviors |=
            Microsoft.CodeAnalysis.Testing.TestBehaviors.SkipGeneratedSourcesCheck;

        await verifier.RunAsync();
    }
}
