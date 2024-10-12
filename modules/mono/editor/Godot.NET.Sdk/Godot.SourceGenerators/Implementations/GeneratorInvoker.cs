using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace Godot.SourceGenerators.Implementation;

/// <summary>
/// Allows chaining source generators.
/// </summary>
public static class GeneratorInvoker
{
    public static void RunAll(IGeneratorExecutionContext context)
    {
        foreach (var generator in GetGeneratorInstances())
        {
            generator.Execute(context);
        }
    }

    public static IEnumerable<IGeneratorImplementation> GetGeneratorInstances()
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
        var generators = implementations
            .Where(t => t is { IsAbstract: false, IsInterface: false })
            .Select(t => (IGeneratorImplementation)t.GetConstructor(Array.Empty<Type>())!.Invoke(Array.Empty<object>()))
            .ToArray();
        return generators;
    }
}
