using Xunit;

namespace Godot.SourceGenerators.Implementation.Tests;

public class GodotGeneratorsTests
{
    [Fact]
    public void GetGeneratorInstances_BehavesProperly()
    {
        var instances = GodotGenerators.CreateInstances();
        Assert.NotEmpty(instances);
    }
}
