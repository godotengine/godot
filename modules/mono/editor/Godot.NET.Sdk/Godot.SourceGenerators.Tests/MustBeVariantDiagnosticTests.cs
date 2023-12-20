using Xunit;

namespace Godot.SourceGenerators.Tests;

public class MustBeVariantDiagnosticTests
{
    [Fact]
    public async void MustBeVariant()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.Verify(
            "MustBeVariant.cs"
        );
    }

    [Fact]
    public async void MustBeVariantDiagnostics()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.Verify(
            "MustBeVariant.Diagnostics.cs"
        );
    }

}
