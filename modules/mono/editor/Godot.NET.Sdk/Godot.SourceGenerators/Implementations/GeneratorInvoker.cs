using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Godot.SourceGenerators.Implementation;

/// <summary>
/// Allows chaining source generators.
/// </summary>
public static class GeneratorInvoker
{
    public static void RunAll(IGeneratorExecutionContext context)
    {
        foreach (var generator in CreateInstances())
        {
            generator.Execute(context);
        }
    }

    public static IEnumerable<IGeneratorImplementation> CreateInstances() => _generatorsConstructor();

    private static readonly Func<IEnumerable<IGeneratorImplementation>> _generatorsConstructor =
        GetGeneratorsConstructor();

    private static Func<IEnumerable<IGeneratorImplementation>> GetGeneratorsConstructor()
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
        var constructors = implementations
            .Where(t => t is { IsAbstract: false, IsInterface: false })
            .Select(t => t.GetConstructor(Array.Empty<Type>()) ?? throw new InvalidOperationException())
            .ToArray();

        return Expression.Lambda<Func<IEnumerable<IGeneratorImplementation>>>(
                Expression.NewArrayInit(typeof(IGeneratorImplementation),
                    constructors.Select(Expression.New)))
            .Compile();
    }
}
