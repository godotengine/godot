using Microsoft.CodeAnalysis.Testing;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptSerializationGeneratorTests
{
    [Fact]
    public async void DisableGenerator()
    {
        await CSharpSourceGeneratorVerifier<ScriptSerializationGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "ScriptBoilerplate.cs" },
            DisabledGenerators = new string[] { "ScriptSerialization" },
        }).RunAsync();
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptSerializationGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            CompilerDiagnostics = CompilerDiagnostics.None,
            Sources = new string[] { "ScriptBoilerplate.cs" },
            GeneratedSources = new string[]
            {
                "ScriptBoilerplate_ScriptSerialization.generated.cs",
                "OuterClass.NestedClass_ScriptSerialization.generated.cs",
            },
        }).RunAsync();
    }
}
