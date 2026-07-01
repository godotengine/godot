using System.Threading.Tasks;
using Xunit;

namespace Godot.SourceGenerators.Tests;

public class UnloadabilityAnalyzerTests
{
    [Fact]
    public async Task SubscriptionToExternalStaticEventTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0001.cs");
    }

    [Fact]
    public async Task GCHandleAllocTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0002.cs");
    }

    [Fact]
    public async Task ThreadPoolRegisterWaitForSingleObjectTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0003.cs");
    }

    [Fact]
    public async Task NewtonsoftJsonSerializationTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0005.cs");
    }

    [Fact]
    public async Task TypeDescriptorModificationTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0006.cs");
    }

    [Fact]
    public async Task ThreadCreationTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0007.cs");
    }

    [Fact]
    public async Task TimerCreationTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0008.cs");
    }

    [Fact]
    public async Task EncodingRegisterProviderTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0009.cs");
    }

    [Fact]
    public async Task TaskRunTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0010.cs");
    }

    [Fact]
    public async Task ThreadPoolQueueUserWorkItemTest()
    {
        await CSharpAnalyzerVerifier<UnloadabilityAnalyzer>.Verify("Unloadability.GDU0011.cs");
    }
}
