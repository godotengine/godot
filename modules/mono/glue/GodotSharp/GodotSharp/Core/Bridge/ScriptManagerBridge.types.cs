using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Godot.Bridge;

#nullable enable

public static partial class ScriptManagerBridge
{
    [SuppressMessage("Design", "CA1001", MessageId = "Types that own disposable fields should be disposable",
            Justification = "Not applicable. The class functions as a persistent singleton.")]
    private class ScriptTypeBiMap
    {
        public readonly ReaderWriterLockSlim ReadWriteLock = new(LockRecursionPolicy.SupportsRecursion);
        private System.Collections.Generic.Dictionary<IntPtr, Type> _scriptTypeMap = new();
        private System.Collections.Generic.Dictionary<Type, IntPtr> _typeScriptMap = new();

        // 2dog: accessors restore trimming annotations lost through generic dictionary storage.
        public void Add(IntPtr scriptPtr, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)] Type scriptType)
        {
            // TODO: What if this is called while unloading a load context, but after we already did cleanup in preparation for unloading?

            Debug.Assert(!scriptType.IsGenericTypeDefinition, $"A generic type definition must never be added to the script type map. Type: {scriptType}.");

            _scriptTypeMap.Add(scriptPtr, scriptType);
            _typeScriptMap.Add(scriptType, scriptPtr);

            if (AlcReloadCfg.IsAlcReloadingEnabled)
            {
                AddTypeForAlcReloading(scriptType);
            }
        }

        public void Remove(IntPtr scriptPtr)
        {
            if (_scriptTypeMap.Remove(scriptPtr, out Type? scriptType))
                _ = _typeScriptMap.Remove(scriptType);
        }

        public bool RemoveByScriptType(Type scriptType, out IntPtr scriptPtr)
        {
            if (_typeScriptMap.Remove(scriptType, out scriptPtr))
                return _scriptTypeMap.Remove(scriptPtr);
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [UnconditionalSuppressMessage("Trimming", "IL2073",
            Justification = "Every type in the map was annotated at Add(); the dictionary drops the annotation. " + TrimJustifications.ScriptTypesAreRooted)]
        [return: DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)]
        public Type GetScriptType(IntPtr scriptPtr) => _scriptTypeMap[scriptPtr];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [UnconditionalSuppressMessage("Trimming", "IL2067",
            Justification = "Every type in the map was annotated at Add(); the dictionary drops the annotation. " + TrimJustifications.ScriptTypesAreRooted)]
        public bool TryGetScriptType(IntPtr scriptPtr,
            [MaybeNullWhen(false)] [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)] out Type scriptType) =>
            _scriptTypeMap.TryGetValue(scriptPtr, out scriptType);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetScriptPtr(Type scriptType, out IntPtr scriptPtr) =>
            _typeScriptMap.TryGetValue(scriptType, out scriptPtr);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsScriptRegistered(IntPtr scriptPtr) => _scriptTypeMap.ContainsKey(scriptPtr);
    }

    private class PathScriptTypeBiMap
    {
        private System.Collections.Generic.Dictionary<string, Type> _pathTypeMap = new();
        private System.Collections.Generic.Dictionary<Type, string> _typePathMap = new();

        public System.Collections.Generic.Dictionary<string, Type>.KeyCollection Paths => _pathTypeMap.Keys;

        // 2dog: use the same annotation contract as ScriptTypeBiMap.
        public void Add(string scriptPath, [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)] Type scriptType)
        {
            _pathTypeMap.Add(scriptPath, scriptType);

            // Due to partial classes, more than one file can point to the same type, so
            // there could be duplicate keys in this case. We only add a type as key once.
            _typePathMap.TryAdd(scriptType, scriptPath);
        }

        public void RemoveByScriptType(Type scriptType)
        {
            foreach (var pair in _pathTypeMap
                         .Where(p => p.Value == scriptType).ToArray())
            {
                _pathTypeMap.Remove(pair.Key);
            }

            _typePathMap.Remove(scriptType);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        [UnconditionalSuppressMessage("Trimming", "IL2067",
            Justification = "Every type in the map was annotated at Add(); the dictionary drops the annotation. " + TrimJustifications.ScriptTypesAreRooted)]
        public bool TryGetScriptType(string scriptPath,
            [MaybeNullWhen(false)] [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.All)] out Type scriptType) =>
            // This must never return true for a generic type definition, we only consider script types
            // the types that can be attached to a Node/Resource (non-generic or constructed generic types).
            _pathTypeMap.TryGetValue(scriptPath, out scriptType) && !scriptType.IsGenericTypeDefinition;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetScriptPath(Type scriptType, [MaybeNullWhen(false)] out string scriptPath)
        {
            if (scriptType.IsGenericTypeDefinition)
            {
                // This must never return true for a generic type definition, we only consider script types
                // the types that can be attached to a Node/Resource (non-generic or constructed generic types).
                scriptPath = null;
                return false;
            }

            return _typePathMap.TryGetValue(scriptType, out scriptPath);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetGenericTypeDefinitionPath(Type genericTypeDefinition, [MaybeNullWhen(false)] out string scriptPath)
        {
            Debug.Assert(genericTypeDefinition.IsGenericTypeDefinition);
            return _typePathMap.TryGetValue(genericTypeDefinition, out scriptPath);
        }
    }
}
