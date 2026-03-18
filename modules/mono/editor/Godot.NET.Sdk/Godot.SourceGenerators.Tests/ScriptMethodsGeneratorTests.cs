using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptMethodsGeneratorTests
{
    [Fact]
    public async void Methods()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.Verify(
            "Methods.cs",
            "Methods_ScriptMethods.generated.cs"
        );
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.Verify(
            "ScriptBoilerplate.cs",
            "ScriptBoilerplate_ScriptMethods.generated.cs", "OuterClass.NestedClass_ScriptMethods.generated.cs"
        );
    }
}
