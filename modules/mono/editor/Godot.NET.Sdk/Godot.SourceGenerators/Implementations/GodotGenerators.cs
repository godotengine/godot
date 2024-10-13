using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;

namespace Godot.SourceGenerators.Implementation;

/// <summary>
/// Allows chaining source generators.
/// </summary>
public static class GodotGenerators
{
    /// <summary>
    /// The recommended way to apply Godot source generators.
    /// </summary>
    public static void RunAll(IGeneratorExecutionContext generatorExecutionContext)
    {
        foreach (var c in GetConstructors())
        {
            c().Execute(generatorExecutionContext);
        }
    }

    /// <summary>
    /// Allows more control over how the generators are executed e.g. in parallel.
    /// Parallel execution can cause cryptic exceptions.
    /// <see cref="RunAll"/> is the recommended way to apply Godot generators.
    /// </summary>
    public static IEnumerable<Func<IGeneratorImplementation>> GetConstructors() => _generatorsConstructor;

    private static readonly Func<IGeneratorImplementation>[] _generatorsConstructor = GetGeneratorConstructors();

    private static Func<IGeneratorImplementation>[] GetGeneratorConstructors()
    {
        Type[] types;
        try
        {
            types = typeof(GodotGenerators).Assembly.GetTypes();
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

        return constructors.Select(c => Expression.Lambda<Func<IGeneratorImplementation>>(
                Expression.New(c))
            .Compile()).ToArray();
    }
}
