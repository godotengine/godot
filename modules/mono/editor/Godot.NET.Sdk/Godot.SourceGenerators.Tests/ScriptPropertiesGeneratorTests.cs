using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPropertiesGeneratorTests
{
    [Fact]
    public async void ExportedFields()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            new string[] { "ExportedFields.cs", "MoreExportedFields.cs" },
            new string[] { "ExportedFields_ScriptProperties.generated.cs" }
        );
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
    public async void OneWayPropertiesMixedReadOnlyWriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "MixedReadOnlyWriteOnly.cs",
            "MixedReadOnlyWriteOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "ScriptBoilerplate.cs",
            "ScriptBoilerplate_ScriptProperties.generated.cs", "OuterClass.NestedClass_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void AbstractGenericNode()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "AbstractGenericNode.cs",
            "AbstractGenericNode_T_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void Generic()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            "Generic.cs",
            "Generic_T_ScriptProperties.generated.cs"
        );
    }
}
