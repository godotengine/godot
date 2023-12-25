using Xunit;

namespace Godot.SourceGenerators.Tests;

public class MustBeVariantAnalyzerTests
{
    [Fact]
    public async void GenericTypeArgumentMustBeVariantTest()
    {
        const string MustBeVariantGD0301 = "MustBeVariant.GD0301.cs";
        await CSharpAnalyzerVerifier<MustBeVariantAnalyzer>.Verify(MustBeVariantGD0301);
    }

    [Fact]
    public async void GenericTypeParameterMustBeVariantAnnotatedTest()
    {
        const string MustBeVariantGD0302 = "MustBeVariant.GD0302.cs";
        await CSharpAnalyzerVerifier<MustBeVariantAnalyzer>.Verify(MustBeVariantGD0302);
    }
}
