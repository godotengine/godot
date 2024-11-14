using Xunit;

namespace Godot.SourceGenerators.Tests;

public class OnReadyGeneratorTests
{
    [Fact]
    public async void OneOnReadyProperties()
    {
        await CSharpSourceGeneratorVerifier<OnReadyGenerator>.Verify(
            "OnReadyPropertiesOne.cs",
            "OnReadyPropertiesOne_OnReady.generated.cs"
        );
    }

    [Fact]
    public async void TwoOnReadyProperties()
    {
        await CSharpSourceGeneratorVerifier<OnReadyGenerator>.Verify(
            "OnReadyPropertiesTwo.cs",
            "OnReadyPropertiesTwo_OnReady.generated.cs"
        );
    }
}
