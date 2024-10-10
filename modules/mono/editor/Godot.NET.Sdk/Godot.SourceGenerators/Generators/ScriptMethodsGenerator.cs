using Microsoft.CodeAnalysis;

namespace Godot.SourceGenerators;

[Generator]
public class ScriptMethodsGenerator : ISourceGenerator
{
    private readonly ScriptMethodsGeneratorImplementation _implementation = new();

    public void Initialize(GeneratorInitializationContext context)
    {
    }

    public void Execute(Microsoft.CodeAnalysis.GeneratorExecutionContext context)
    {
        _implementation.Execute(new GeneratorExecutionContext(context));
    }
}
