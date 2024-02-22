using Xunit;

namespace Godot.SourceGenerators.Tests;

public class GlobalClassAnalyzerTests
{
    [Fact]
    public async void GlobalClassMustDeriveFromGodotObjectTest()
    {
        const string GlobalClassGD0401 = "GlobalClass.GD0401.cs";
        await CSharpAnalyzerVerifier<GlobalClassAnalyzer>.Verify(GlobalClassGD0401);
    }

    [Fact]
    public async void GlobalClassMustNotBeGenericTest()
    {
        const string GlobalClassGD0402 = "GlobalClass.GD0402.cs";
        await CSharpAnalyzerVerifier<GlobalClassAnalyzer>.Verify(GlobalClassGD0402);
    }
}
