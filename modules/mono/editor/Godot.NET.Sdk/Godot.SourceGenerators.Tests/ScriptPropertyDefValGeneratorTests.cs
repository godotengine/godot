using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPropertyDefValGeneratorTests
{
    [Fact]
    public async void ExportedFields()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGeneratorRunner>.Verify(
            new string[] { "ExportedFields.cs", "MoreExportedFields.cs" },
            new string[] { "ExportedFields_ScriptPropertyDefVal.generated.cs" }
        );
    }

    [Fact]
    public async void ExportedProperties()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGeneratorRunner>.Verify(
            "ExportedProperties.cs",
            "ExportedProperties_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async void ExportedComplexStrings()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGeneratorRunner>.Verify(
            "ExportedComplexStrings.cs",
            "ExportedComplexStrings_ScriptPropertyDefVal.generated.cs"
        );
    }
}
