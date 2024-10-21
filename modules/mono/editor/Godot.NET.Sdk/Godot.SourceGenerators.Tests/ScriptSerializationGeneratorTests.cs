using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptSerializationGeneratorTests
{
    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptSerializationGenerator>.VerifyNoCompilerDiagnostics(
            "ScriptBoilerplate.cs",
            "ScriptBoilerplate_ScriptSerialization.generated.cs", "OuterClass.NestedClass_ScriptSerialization.generated.cs"
        );
    }

    [Fact]
    public async void Inheritance()
    {
        await CSharpSourceGeneratorVerifier<ScriptSerializationGenerator>.VerifyNoCompilerDiagnostics(
            new string[] { "Inheritance.cs" },
            new string[] { "InheritanceBase_ScriptSerialization.generated.cs", "InheritanceChild_ScriptSerialization.generated.cs" }
        );
    }
}
