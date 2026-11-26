using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ClassPartialModifierTest
{
    [Fact]
    public async Task ClassPartialModifierCodeFixTest()
    {
        await CSharpCodeFixVerifier<ClassPartialModifierCodeFixProvider, ClassPartialModifierAnalyzer>
            .Verify("ClassPartialModifier.GD0001.cs", "ClassPartialModifier.GD0001.fixed.cs");
    }

    [Fact]
    public async Task OuterClassPartialModifierAnalyzerTest()
    {
        await CSharpAnalyzerVerifier<ClassPartialModifierAnalyzer>.Verify("OuterClassPartialModifierAnalyzer.GD0002.cs");
    }
}
