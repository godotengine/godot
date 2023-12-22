using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.Testing;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class GlobalClassAnalyzerTests
{
    [Fact]
    public async void GlobalClassMustDeriveFromGodotObjectTest()
    {
        const string GlobalClassGD0401 = "GlobalClass.GD0401.cs";

        await CSharpAnalyzerVerifier<GlobalClassAnalyzer>.Verify(GlobalClassGD0401,
            new DiagnosticResult("GD0401", DiagnosticSeverity.Error)
                .WithSpan(GlobalClassGD0401, 15, 1, 19, 2)
                .WithArguments(GlobalClassGD0401)
        );
    }

    [Fact]
    public async void GlobalClassMustNotBeGenericTest()
    {
        const string GlobalClassGD0402 = "GlobalClass.GD0402.cs";

        await CSharpAnalyzerVerifier<GlobalClassAnalyzer>.Verify(GlobalClassGD0402,
            new DiagnosticResult("GD0402", DiagnosticSeverity.Error)
                .WithSpan(GlobalClassGD0402, 9, 1, 13, 2)
                .WithArguments(GlobalClassGD0402)
        );
    }

}
