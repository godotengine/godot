using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.CodeAnalysis.Text;
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

    [Fact]
    public async void ScriptBoilerplate()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "ScriptBoilerplate.cs" },
            new string[] { "ScriptBoilerplate_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::ScriptBoilerplate" }));
        await verifier.RunAsync();
    }

    [Fact]
    public async void FooBar()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "Foo.cs", "Bar.cs" },
            new string[] { "Foo_ScriptPath.generated.cs", "Bar_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::Foo", "global::Bar" }));
        await verifier.RunAsync();
    }

    [Fact]
    public async void Generic()
    {
        var verifier = CSharpSourceGeneratorVerifier<ScriptPathAttributeGenerator>.MakeVerifier(
            new string[] { "Generic.cs" },
            new string[] { "Generic_ScriptPath.generated.cs" }
        );
        verifier.TestState.GeneratedSources.Add(MakeAssemblyScriptTypesGeneratedSource(new string[] { "global::Generic" }));
        await verifier.RunAsync();
    }
}
