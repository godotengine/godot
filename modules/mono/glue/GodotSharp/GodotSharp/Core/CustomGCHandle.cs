#nullable enable

using System;
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Godot.Bridge;

namespace Godot;

/// <summary>
/// Provides a GCHandle that becomes weak when unloading the assembly load context, without having
/// to manually replace the GCHandle. This hides all the complexity of releasing strong GC handles
/// to allow the assembly load context to unload properly.
///
/// Internally, a strong CustomGCHandle actually contains a weak GCHandle, while the actual strong
/// reference is stored in a static table.
/// </summary>
public static class CustomGCHandle
{
    // ConditionalWeakTable uses DependentHandle, so it stores weak references.
    // Having the assembly load context as key won't prevent it from unloading.
    private static ConditionalWeakTable<AssemblyLoadContext, object?> _alcsBeingUnloaded = new();

    /// <summary>
    /// Determines if the provided <see cref="AssemblyLoadContext"/> is being unloaded.
    /// </summary>
    /// <param name="alc">The context to check.</param>
    /// <returns><see langword="true"/> if the context is currently being unloaded;
    /// otherwise, <see langword="false"/>.</returns>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static bool IsAlcBeingUnloaded(AssemblyLoadContext alc) => _alcsBeingUnloaded.TryGetValue(alc, out _);

    // ReSharper disable once RedundantNameQualifier
    private static ConcurrentDictionary<
        AssemblyLoadContext,
        ConcurrentDictionary<GCHandle, object>
    > _strongReferencesByAlc = new();

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void OnAlcUnloading(AssemblyLoadContext alc)
    {
        _alcsBeingUnloaded.Add(alc, null);

        if (_strongReferencesByAlc.TryRemove(alc, out var strongReferences))
        {
            strongReferences.Clear();
        }
    }

    /// <summary>
    /// Initializes a strong <see cref="GCHandle"/> using the provided <paramref name="value"/>
    /// </summary>
    /// <param name="value">The value to assign a strong handle.</param>
    /// <returns>A strong handle for the provided value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GCHandle AllocStrong(object value)
        => AllocStrong(value, value.GetType());

    /// <summary>
    /// Initializes a strong <see cref="GCHandle"/> using the provided <paramref name="value"/>
    /// with the given <paramref name="valueType"/>
    /// </summary>
    /// <param name="value">The value to assign a strong handle.</param>
    /// <param name="valueType">The type of the object.</param>
    /// <returns>A strong handle for the provided value and type.</returns>
    public static GCHandle AllocStrong(object value, Type valueType)
    {
        if (AlcReloadCfg.IsAlcReloadingEnabled)
        {
            var alc = AssemblyLoadContext.GetLoadContext(valueType.Assembly);

            if (alc != null)
            {
                var weakHandle = GCHandle.Alloc(value, GCHandleType.Weak);

                if (!IsAlcBeingUnloaded(alc))
                {
                    var strongReferences = _strongReferencesByAlc.GetOrAdd(alc,
                        static alc =>
                        {
                            alc.Unloading += OnAlcUnloading;
                            return new();
                        });
                    strongReferences.TryAdd(weakHandle, value);
                }

                return weakHandle;
            }
        }

        return GCHandle.Alloc(value, GCHandleType.Normal);
    }

    /// <summary>
    /// Initializes a weak <see cref="GCHandle"/> using the provided <paramref name="value"/>
    /// </summary>
    /// <param name="value">The value to assign a weak handle.</param>
    /// <returns>A weak handle for the provided value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GCHandle AllocWeak(object value) => GCHandle.Alloc(value, GCHandleType.Weak);

    /// <summary>
    /// Releases a <see cref="GCHandle"/>.
    /// </summary>
    /// <param name="handle">The <see cref="GCHandle"/> to release.</param>
    public static void Free(GCHandle handle)
    {
        if (AlcReloadCfg.IsAlcReloadingEnabled)
        {
            var target = handle.Target;

            if (target != null)
            {
                var alc = AssemblyLoadContext.GetLoadContext(target.GetType().Assembly);

                if (alc != null && _strongReferencesByAlc.TryGetValue(alc, out var strongReferences))
                    _ = strongReferences.TryRemove(handle, out _);
            }
        }

        handle.Free();
    }
}
