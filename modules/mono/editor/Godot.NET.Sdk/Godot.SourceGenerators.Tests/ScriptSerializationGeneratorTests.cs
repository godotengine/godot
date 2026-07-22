using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptSerializationGeneratorTests
{
    [Fact]
    public async Task ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptSerializationGenerator>.VerifyNoCompilerDiagnostics(
            "ScriptBoilerplate.cs",
            "ScriptBoilerplate_ScriptSerialization.generated.cs", "OuterClass.NestedClass_ScriptSerialization.generated.cs"
        );
    }

    [Fact]
    public async Task Inheritance()
    {
        await CSharpSourceGeneratorVerifier<ScriptSerializationGenerator>.VerifyNoCompilerDiagnostics(
            new string[] { "Inheritance.cs" },
            new string[] { "InheritanceBase_ScriptSerialization.generated.cs", "InheritanceChild_ScriptSerialization.generated.cs" }
        );
    }
}
