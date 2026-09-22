using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ExportDiagnosticsTests
{
    [Fact]
    public async Task StaticMembers()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics_GD0101.cs",
            "ExportDiagnostics_GD0101_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async Task TypeIsNotSupported()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics_GD0102.cs",
            "ExportDiagnostics_GD0102_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async Task ReadOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics_GD0103.cs",
            "ExportDiagnostics_GD0103_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async Task WriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics_GD0104.cs",
            "ExportDiagnostics_GD0104_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async Task Indexer()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics_GD0105.cs",
            "ExportDiagnostics_GD0105_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async Task ExplicitInterfaceImplementation()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            new string[] { "ExportDiagnostics_GD0106.cs" },
            new string[]
            {
                "ExportDiagnostics_GD0106_OK_ScriptPropertyDefVal.generated.cs",
                "ExportDiagnostics_GD0106_KO_ScriptPropertyDefVal.generated.cs",
            }
        );
    }

    [Fact]
    public async Task NodeExports()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            new string[] { "ExportDiagnostics_GD0107.cs" },
            new string[]
            {
                "ExportDiagnostics_GD0107_OK_ScriptPropertyDefVal.generated.cs",
                "ExportDiagnostics_GD0107_KO_ScriptPropertyDefVal.generated.cs",
            }
        );
    }

    [Fact]
    public async Task ExportToolButtonInNonToolClass()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            new string[] { "ExportDiagnostics_GD0108.cs" },
            new string[] { "ExportDiagnostics_GD0108_ScriptProperties.generated.cs" }
        );
    }

    [Fact]
    public async Task ExportAndExportToolButtonOnSameMember()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            new string[] { "ExportDiagnostics_GD0109.cs" },
            new string[] { "ExportDiagnostics_GD0109_ScriptProperties.generated.cs" }
        );
    }

    [Fact]
    public async Task ExportToolButtonOnNonCallable()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            new string[] { "ExportDiagnostics_GD0110.cs" },
            new string[] { "ExportDiagnostics_GD0110_ScriptProperties.generated.cs" }
        );
    }

    [Fact]
    public async Task ExportToolButtonStoringCallable()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertiesGenerator>.Verify(
            new string[] { "ExportDiagnostics_GD0111.cs" },
            new string[] { "ExportDiagnostics_GD0111_ScriptProperties.generated.cs" }
        );
    }
}
