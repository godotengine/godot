using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis.Text;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPathAttributeGeneratorTests
{
    private static (string, SourceText) MakeAssemblyScriptTypesGeneratedSource(params string[] types)
    {
        var typesArrayContent = string.Join(", ", types.Select(type => $"typeof({type})"));
        return (
            Path.Combine(
                "Godot.SourceGenerators",
                "Godot.SourceGenerators.ScriptPathAttributeGenerator",
                "AssemblyScriptTypes.generated.cs"
            ),
            SourceText.From($$"""
            [assembly:Godot.AssemblyHasScriptsAttribute(new System.Type[] {{{typesArrayContent}}})]

            """, Encoding.UTF8)
        );
    }

    [Fact]
    private async void DisableGenerator()
    {
        await CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "ScriptBoilerplate.cs" },
            DisabledGenerators = new string[] { "ScriptPathAttribute" },
        }).RunAsync();
    }

    [Fact(Skip = "This is currently failing due to an upstream issue with ';' in the editorconfig file format.")]
    private async void DisableGeneratorButNotFirst()
    {
        await CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "ScriptBoilerplate.cs" },
            DisabledGenerators = new string[] { "SomePlaceholder", "ScriptPathAttribute" },
        }).RunAsync();
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier()
            .WithSources("ScriptBoilerplate.cs")
            .WithGeneratedSources("ScriptBoilerplate_ScriptPath.generated.cs");
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource("global::ScriptBoilerplate"));
        await verifier.RunAsync();
    }

    [Fact]
    public async void FooBar()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier()
            .WithSources("Foo.cs", "Bar.cs")
            .WithGeneratedSources("Foo_ScriptPath.generated.cs", "Bar_ScriptPath.generated.cs");
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource("global::Foo", "global::Bar"));
        await verifier.RunAsync();
    }

    [Fact]
    public async void Generic()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier()
            .WithSources("Generic.cs")
            .WithGeneratedSources("Generic(Of T)_ScriptPath.generated.cs");
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource("global::Generic<>"));
        await verifier.RunAsync();
    }

    [Fact]
    public async void GenericMultipleClassesSameName()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier()
            .WithSources((Filename: "Generic.cs", ContentFilename: "Generic.GD0003.cs"))
            .WithGeneratedSources("Generic(Of T)_ScriptPath.generated.cs");
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(
            "global::Generic<>", "global::Generic<,>", "global::Generic"
        ));
        await verifier.RunAsync();
    }

    [Fact]
    public async void NamespaceMultipleClassesSameName()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier()
            .WithSources((Filename: "SameName.cs", ContentFilename: "SameName.GD0003.cs"))
            .WithGeneratedSources("NamespaceA.SameName_ScriptPath.generated.cs");
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(
            "global::NamespaceA.SameName", "global::NamespaceB.SameName"
        ));
        await verifier.RunAsync();
    }
}
