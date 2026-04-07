using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptPropertyDefValGeneratorTests
{
    [Fact]
    public async Task ExportedFields()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            new string[] { "ExportedFields.cs", "MoreExportedFields.cs" },
            new string[] { "ExportedFields_ScriptPropertyDefVal.generated.cs" }
        );
    }

    [Fact]
    public async Task ExportedProperties()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportedProperties.cs",
            "ExportedProperties_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async Task ExportedProperties2()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportedProperties2.cs", "ExportedProperties2_ScriptPropertyDefVal.generated.cs");
    }

    [Fact]
    public async Task ExportedComplexStrings()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportedComplexStrings.cs",
            "ExportedComplexStrings_ScriptPropertyDefVal.generated.cs"
        );
    }
}
