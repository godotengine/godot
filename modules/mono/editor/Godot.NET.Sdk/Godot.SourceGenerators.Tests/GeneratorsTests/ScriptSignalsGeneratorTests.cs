using Xunit;

namespace Godot.SourceGenerators.Tests;

public class ScriptSignalsGeneratorTests
{
    [Fact]
    public async void DisableGenerator()
    {
        await CSharpSourceGeneratorVerifier<ScriptSignalsGenerator>.MakeVerifier(new VerifierConfiguration()
        {
            Sources = new string[] { "EventSignals.cs" },
            DisabledGenerators = new string[] { "ScriptSignals" },
        }).RunAsync();
    }

    [Fact]
    public async void EventSignals()
    {
        await CSharpSourceGeneratorVerifier<ScriptSignalsGenerator>.Verify(
            "EventSignals.cs",
            "EventSignals_ScriptSignals.generated.cs"
        );
    }

    [Fact]
    // ReSharper disable once InconsistentNaming
    public async void EventSignalsGD020X()
    {
        await CSharpSourceGeneratorVerifier<ScriptSignalsGenerator>.MakeVerifier()
            .WithSources("EventSignals.GD020X.cs")
            .WithGeneratedSources("EventSignals_ScriptSignals.generated.cs")
            .RunAsync();
    }
}
