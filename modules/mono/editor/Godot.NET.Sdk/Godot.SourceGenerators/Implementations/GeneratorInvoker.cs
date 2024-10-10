using System;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace Godot.SourceGenerators.Implementations;

/// <summary>
/// Allows chaining source generators.
/// </summary>
public static class GeneratorInvoker
{
    public static void RunAll(IGeneratorExecutionContext context, CancellationToken cancellationToken)
    {
        Type[] types;
        try
        {
            types = typeof(GeneratorInvoker).Assembly.GetTypes();
        }
        catch (ReflectionTypeLoadException e)
        {
            types = e.Types.Where(t => t is not null).ToArray();
        }

        var implementations = types
            .Where(t => typeof(IGeneratorImplementation).IsAssignableFrom(t));
        var iGeneratorImplementations = implementations
            .Where(t => t is { IsAbstract: false, IsInterface: false })
            .Select(t => (IGeneratorImplementation)t.GetConstructor(Array.Empty<Type>())!.Invoke(Array.Empty<object>()));
        var generators = iGeneratorImplementations
            .ToArray();

        if (generators.Length == 0)
        {
            throw new InvalidOperationException("Did not find any generators to run.");
        }

        Parallel.Invoke(
            new ParallelOptions
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount - 1, CancellationToken = cancellationToken
            },
            generators.Select(g => (Action)(() => g.Execute(context))).ToArray());
    }
}
