using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptMethodsGeneratorTests
{
    [Fact]
    public async void DisableGenerator()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "ScriptBoilerplate.cs" },
            DisabledGenerators = new string[] { "ScriptMethods" },
        }).RunAsync();
    }

    [Fact]
    public async void Methods()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.MakeVerifier()
            .WithSources("Methods.cs")
            .WithGeneratedSources("Methods_ScriptMethods.generated.cs")
            .RunAsync();
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.MakeVerifier()
            .WithSources("ScriptBoilerplate.cs")
            .WithGeneratedSources(
                "ScriptBoilerplate_ScriptMethods.generated.cs",
                "OuterClass.NestedClass_ScriptMethods.generated.cs"
            )
            .RunAsync();
    }
}
