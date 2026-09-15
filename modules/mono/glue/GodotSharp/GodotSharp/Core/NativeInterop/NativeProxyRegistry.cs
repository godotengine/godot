using System;
using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;

namespace Godot.NativeInterop;

#nullable enable

internal record NativeProxyMeta(Type ProxyType, StringName NativeName);

internal static class NativeProxyRegistry
{
    private static readonly ConcurrentDictionary<Type, NativeProxyMeta> _typeNativeProxyMetaMap = new();

    internal static NativeProxyMeta GetNativeProxyMeta(Type proxyType) => _typeNativeProxyMetaMap[proxyType];

    internal static NativeProxyMeta? GetNativeProxyMetaOrNull(Type proxyType)
        => _typeNativeProxyMetaMap.TryGetValue(proxyType, out var nativeProxyMeta) ? nativeProxyMeta : null;

    internal static bool TryGetNativeProxyMeta(Type proxyType,
        [MaybeNullWhen(false)] out NativeProxyMeta nativeProxyMeta)
        => _typeNativeProxyMetaMap.TryGetValue(proxyType, out nativeProxyMeta);
}
