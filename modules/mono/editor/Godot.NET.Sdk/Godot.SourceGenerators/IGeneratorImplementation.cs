namespace Godot.SourceGenerators;

public interface IGeneratorImplementation
{
    void Execute(IGeneratorExecutionContext context);
}
