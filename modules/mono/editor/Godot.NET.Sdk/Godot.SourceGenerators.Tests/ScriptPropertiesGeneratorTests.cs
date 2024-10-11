using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPropertiesGeneratorTests
{
    [Fact]
    public async void ExportedFields()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            new string[] { "ExportedFields.cs", "MoreExportedFields.cs" },
            new string[] { "ExportedFields_ScriptProperties.generated.cs" }
        );
    }

    [Fact]
    public async void ExportedProperties()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            "ExportedProperties.cs",
            "ExportedProperties_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void OneWayPropertiesAllReadOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            "AllReadOnly.cs",
            "AllReadOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void OneWayPropertiesAllWriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            "AllWriteOnly.cs",
            "AllWriteOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void OneWayPropertiesMixedReadOnlyWriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            "MixedReadOnlyWriteOnly.cs",
            "MixedReadOnlyWriteOnly_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void ScriptBoilerplate()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            "ScriptBoilerplate.cs",
            "ScriptBoilerplate_ScriptProperties.generated.cs", "OuterClass.NestedClass_ScriptProperties.generated.cs"
        );
    }

    [Fact]
    public async void AbstractGenericNode()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGeneratorRunner>.Verify(
            "AbstractGenericNode.cs",
            "AbstractGenericNode(Of T)_ScriptProperties.generated.cs"
        );
    }
}
