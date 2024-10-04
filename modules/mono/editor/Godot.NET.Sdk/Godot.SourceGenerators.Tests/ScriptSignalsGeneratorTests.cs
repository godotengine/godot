using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptSignalsGeneratorImplementationTests
{
    [Fact]
    public async void EventSignals()
    {
        await CSharpSourceGeneratorVerifier<ScriptSignalsGenerator>.Verify(
            "EventSignals.cs",
            "EventSignals_ScriptSignals.generated.cs"
        );
    }
}
