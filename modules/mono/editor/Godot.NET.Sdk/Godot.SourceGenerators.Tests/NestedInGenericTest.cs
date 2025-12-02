using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class NestedInGenericTest
{
    [Fact]
    public async Task GenerateScriptMethodsTest()
    {
        await CSharpSourceGeneratorVerifier<ScriptMethodsGenerator>.Verify(
            "NestedInGeneric.cs",
            "GenericClass(Of T).NestedClass_ScriptMethods.generated.cs"
        );
    }
}
