using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPropertiesGeneratorTests
{
    [Fact]
    public async void DisableGenerator()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "ScriptBoilerplate.cs" },
            DisabledGenerators = new string[] { "ScriptProperties" },
        }).RunAsync();
    }

    [Fact]
    public async void ExportedFields()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.MakeVerifier()
            .WithSources("ExportedFields.cs", "MoreExportedFields.cs")
            .WithGeneratedSources("ExportedFields_ScriptProperties.generated.cs")
            .RunAsync();
    }

    [Fact]
    public async void ExportedProperties()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "ExportedProperties.cs",
            "ExportedProperties_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void OneWayPropertiesAllReadOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "AllReadOnly.cs",
            "AllReadOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void OneWayPropertiesAllWriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "AllWriteOnly.cs",
            "AllWriteOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void OneWayPropertiesMixedReadonlyWriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "MixedReadOnlyWriteOnly.cs",
            "MixedReadOnlyWriteOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.MakeVerifier()
            .WithSources("ScriptBoilerplate.cs")
            .WithGeneratedSources(
                "ScriptBoilerplate_ScriptProperties.generated.cs",
                "OuterClass.NestedClass_ScriptProperties.generated.cs"
            )
            .RunAsync();
    }
}
