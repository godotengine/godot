using Xunit;

namespace Godot.SourceGenerators.Implementation.Tests;

public class GeneratorInvokerTests
{
    [Fact]
    public void GetGeneratorInstances_BehavesProperly()
    {
        var instances = GeneratorInvoker.GetGeneratorInstances();
        Assert.NotEmpty(instances);
    }
}
