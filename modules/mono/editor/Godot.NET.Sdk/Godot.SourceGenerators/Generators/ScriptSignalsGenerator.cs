using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators;

[Generator]
public class ScriptSignalsGenerator : ISourceGenerator
{
    private readonly ScriptSignalsGeneratorImplementation _implementation = new();

    public void Initialize(GeneratorInitializationContext context)
    {
    }

    public void Execute(Microsoft.CodeAnalysis.GeneratorExecutionContext context)
    {
        _implementation.Execute(new GeneratorExecutionContext(context));
    }
}
