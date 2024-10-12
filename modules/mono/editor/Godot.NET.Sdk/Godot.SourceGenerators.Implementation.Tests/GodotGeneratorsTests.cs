using Xunit;

namespace Godot.SourceGenerators.Implementation.Tests;

public class GodotGeneratorsTests
{
    [Fact]
    public void GetGeneratorInstances_BehavesProperly()
    {
        var constructors = GodotGenerators.GetConstructors();
        Assert.NotEmpty(constructors);
    }
}
