using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ClassParameterlessConstructorTest
{
    [Fact]
    public async Task ClassParameterlessConstructorAnalyzerTest()
    {
        await CSharpAnalyzerVerifier<ClassParameterlessConstructorAnalyzer>.Verify("ClassParameterlessConstructorAnalyzer.GD0004.cs");
    }

    [Fact]
    public async Task ClassParameterlessConstructorCodeFixTest()
    {
        await CSharpCodeFixVerifier<ClassParameterlessConstructorCodeFixProvider, ClassParameterlessConstructorAnalyzer>
            .Verify("ClassParameterlessConstructor.GD0004.cs", "ClassParameterlessConstructor.GD0004.fixed.cs");
    }
}
