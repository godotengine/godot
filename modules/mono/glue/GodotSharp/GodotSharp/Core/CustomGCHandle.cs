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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GCHandle AllocStrong(object value)
        => AllocStrong(value, value.GetType());

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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static GCHandle AllocWeak(object value) => GCHandle.Alloc(value, GCHandleType.Weak);

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
