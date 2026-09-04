using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class NoScriptFileAssociationAnalyzerTests
{
    [Fact]
    public async Task NoScriptFileAssociationAnalyzer_GD0010()
    {
        const string NoScriptFileAssociationAnalyzerGD0010 = "NoScriptFileAssociationAnalyzer.GD0010.cs";
        await CSharpAnalyzerVerifier<NoScriptFileAssociationAnalyzer>.Verify(NoScriptFileAssociationAnalyzerGD0010);
    }
    [Fact]
    public async Task NoScriptFileAssociationAnalyzer_GD0011()
    {
        const string NoScriptFileAssociationAnalyzerGD0011 = "NoScriptFileAssociationAnalyzer.GD0011.cs";
        await CSharpAnalyzerVerifier<NoScriptFileAssociationAnalyzer>.Verify(NoScriptFileAssociationAnalyzerGD0011);
    }
}
