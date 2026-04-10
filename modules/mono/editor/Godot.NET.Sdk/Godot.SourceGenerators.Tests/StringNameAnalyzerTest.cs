using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class StringNameAnalyzerTest
{
    [Fact]
    public async Task StringNameShouldNotBeUsedInLoopCodeFixTest()
    {
        await CSharpCodeFixVerifier<StringNameCodeFixProvider, StringNameAnalyzer>
            .Verify("StringName.GD0501.cs", "StringName.GD0501.fixed.cs");
    }

    [Fact]
    public async void StringNameShouldNotBeUsedInLoopAnalyzerTest()
    {
        await CSharpAnalyzerVerifier<StringNameAnalyzer>.Verify(
            "StringName.GD0501.cs");
    }
}
