using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class GenericScriptTypeMetaProviderAnalyzerTests
{
    [Fact]
    public async Task GenericScriptTypeMetaProvider_GD0004()
    {
        const string GenericScriptTypeMetaProviderGD0004 = "GenericScriptTypeMetaProvider.GD0004.cs";
        await CSharpAnalyzerVerifier<GenericScriptTypeMetaProviderAnalyzer>.Verify(GenericScriptTypeMetaProviderGD0004);
    }
    [Fact]
    public async Task GenericScriptTypeMetaProvider_GD0005()
    {
        const string GenericScriptTypeMetaProviderGD0005 = "GenericScriptTypeMetaProvider.GD0005.cs";
        await CSharpAnalyzerVerifier<GenericScriptTypeMetaProviderAnalyzer>.Verify(GenericScriptTypeMetaProviderGD0005);
    }
    [Fact]
    public async Task GenericScriptTypeMetaProvider_GD0006()
    {
        const string GenericScriptTypeMetaProviderGD0006 = "GenericScriptTypeMetaProvider.GD0006.cs";
        await CSharpAnalyzerVerifier<GenericScriptTypeMetaProviderAnalyzer>.Verify(GenericScriptTypeMetaProviderGD0006);
    }
    [Fact]
    public async Task GenericScriptTypeMetaProvider_GD0007()
    {
        const string GenericScriptTypeMetaProviderGD0007 = "GenericScriptTypeMetaProvider.GD0007.cs";
        await CSharpAnalyzerVerifier<GenericScriptTypeMetaProviderAnalyzer>.Verify(GenericScriptTypeMetaProviderGD0007);
    }
    [Fact]
    public async Task GenericScriptTypeMetaProvider_GD0008()
    {
        const string GenericScriptTypeMetaProviderGD0008 = "GenericScriptTypeMetaProvider.GD0008.cs";
        await CSharpAnalyzerVerifier<GenericScriptTypeMetaProviderAnalyzer>.Verify(GenericScriptTypeMetaProviderGD0008);
    }
    [Fact]
    public async Task GenericScriptTypeMetaProvider_GD0009()
    {
        const string GenericScriptTypeMetaProviderGD0009 = "GenericScriptTypeMetaProvider.GD0009.cs";
        await CSharpAnalyzerVerifier<GenericScriptTypeMetaProviderAnalyzer>.Verify(GenericScriptTypeMetaProviderGD0009);
    }
}
