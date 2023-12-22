using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Testing;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class MustBeVariantAnalyzerTests
{
    [Fact]
    public async void GenericTypeArgumentMustBeVariantTest()
    {
        const string MustBeVariantGD0301 = "MustBeVariant.GD0301.cs";

        await CSharpAnalyzerVerifier<MustBeVariantAnalyzer>.Verify(MustBeVariantGD0301,
            new DiagnosticResult("GD0301", DiagnosticSeverity.Error)
                .WithSpan(MustBeVariantGD0301, 11, 16, 11, 22)
                .WithArguments(MustBeVariantGD0301)
        );
    }

    [Fact]
    public async void GenericTypeParameterMustBeVariantAnnotatedTest()
    {
        const string MustBeVariantGD0302 = "MustBeVariant.GD0302.cs";

        await CSharpAnalyzerVerifier<MustBeVariantAnalyzer>.Verify(MustBeVariantGD0302,
            new DiagnosticResult("GD0302", DiagnosticSeverity.Error)
                .WithSpan(MustBeVariantGD0302, 15, 26, 15, 27)
                .WithArguments(MustBeVariantGD0302),
            new DiagnosticResult("GD0302", DiagnosticSeverity.Error)
                .WithSpan(MustBeVariantGD0302, 16, 16, 16, 17)
                .WithArguments(MustBeVariantGD0302)
        );
    }
}
