using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPropertyDefValGeneratorTests
{
    [Fact]
    public async void DisableGenerator()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "ExportedFields.cs", "ExportedProperties.cs" },
            DisabledGenerators = new string[] { "ScriptPropertyDefVal" },
        }).RunAsync();
    }

    [Fact]
    public async void ExportedFields()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.MakeVerifier()
            .WithSources("ExportedFields.cs", "MoreExportedFields.cs")
            .WithGeneratedSources("ExportedFields_ScriptPropertyDefVal.generated.cs")
            .RunAsync();
    }

    [Fact]
    public async void ExportedProperties()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportedProperties.cs",
            "ExportedProperties_ScriptPropertyDefVal.generated.cs"
        );
    }
}
