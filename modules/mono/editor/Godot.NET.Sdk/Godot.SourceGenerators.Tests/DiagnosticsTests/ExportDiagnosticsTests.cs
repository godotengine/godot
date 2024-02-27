using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ExportDiagnosticsTests
{
    [Fact]
    public async void StaticMembers()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics.GD0101.cs",
            "ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async void TypeIsNotSupported()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics.GD0102.cs",
            "ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async void ReadOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics.GD0103.cs",
            "ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async void WriteOnly()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.Verify(
            "ExportDiagnostics.GD0104.cs",
            "ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs"
        );
    }

    [Fact]
    public async void Indexer()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.MakeVerifier()
            .WithSources("ExportDiagnostics.GD0105.cs")
            .WithGeneratedSources("ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs")
            .RunAsync();
    }

    [Fact]
    public async void ExplicitInterfaceImplementation()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.MakeVerifier()
            .WithSources("ExportDiagnostics.GD0106.cs")
            .WithGeneratedSources(
                "ExportDiagnosticsInterface_ScriptPropertyDefVal.generated.cs",
                "ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs"
            )
            .RunAsync();
    }

    [Fact]
    public async void NodeExports()
    {
        await CSharpSourceGeneratorVerifier<ScriptPropertyDefValGenerator>.MakeVerifier()
            .WithSources("ExportDiagnostics.GD0107.cs")
            .WithGeneratedSources(
                "ExportDiagnosticsNodes_ScriptPropertyDefVal.generated.cs",
                "ExportDiagnosticsEmpty_ScriptPropertyDefVal.generated.cs"
            )
            .RunAsync();
    }
}
