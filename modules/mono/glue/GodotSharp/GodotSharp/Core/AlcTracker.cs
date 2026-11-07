#nullable enable

using System;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;

namespace Godot;

/// <summary>
/// Keeps track of all unloading AssemblyLoadContext used by Godot.
/// </summary>
public static class AlcTracker
{
    // ConditionalWeakTable uses DependentHandle, so it stores weak references.
    // Having the assembly load context as key won't prevent it from unloading.
    private static ConditionalWeakTable<AssemblyLoadContext, object?> _alcsBeingUnloaded = new();
    // Forward registered AssemblyLoadContext to other places.
    public static event Action<AssemblyLoadContext>? AlcBeginUnloading;

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static bool IsAlcBeingUnloaded(AssemblyLoadContext alc) => _alcsBeingUnloaded.TryGetValue(alc, out _);

    [MethodImpl(MethodImplOptions.NoInlining)]
    internal static bool IsAssemblyBeingUnloaded(Assembly assembly)
    {
        var alc = AssemblyLoadContext.GetLoadContext(assembly);
        return alc is not null && IsAlcBeingUnloaded(alc);
    }

    public static void RegisterUnloadingAlc(AssemblyLoadContext alc)
    {
        if (_alcsBeingUnloaded.TryAdd(alc, null))
        {
            AlcBeginUnloading?.Invoke(alc);
        }
    }
}
