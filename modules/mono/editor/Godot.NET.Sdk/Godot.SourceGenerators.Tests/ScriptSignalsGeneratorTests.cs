using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptSignalsGeneratorTests
{
    [Fact]
    public async void EventSignals()
    {
        await CSharpSourceGeneratorVerifier<ScriptSignalsGenerator>.Verify(
            "EventSignals.cs",
            "EventSignals_ScriptSignals.generated.cs"
        );
    }

    [Fact]
    public async void Generic()
    {
        await CSharpSourceGeneratorVerifier<ScriptSignalsGenerator>.Verify(
            "Generic.cs",
            "Generic_T_ScriptSignals.generated.cs"
        );
    }
}
