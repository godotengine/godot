using Godot.NativeInterop;
using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Godot.Bridge
{
    public sealed class ScriptMethodRegistry<T> where T : GodotObject
    {
        internal Dictionary<MethodKey, ScriptMethod<GodotObject>> BuilderMethodsByNameAndArgc = new();

        internal FrozenDictionary<MethodKey, ScriptMethod<GodotObject>> MethodsByNameAndArgc;

        internal Dictionary<MethodKey, StringName> Aliases { get; } = new();

        private readonly HashSet<IntPtr> _knownMethodNames = new();

        public ScriptMethodRegistry<T> AddAlias(StringName methodName, int argumentCount, StringName alias)
        {
            Aliases[new MethodKey(methodName, argumentCount)] = alias;
            return this;
        }

        public ScriptMethodRegistry<T> Register(StringName methodName, int argumentCount, ScriptMethod<GodotObject> method)
        {
            BuilderMethodsByNameAndArgc[new MethodKey(methodName, argumentCount)] = method;
            _knownMethodNames.Add(methodName.NativeValue._data);
            return this;
        }

        public ScriptMethodRegistry<T> Build()
        {
            int aliasesRegistered = 0;
            foreach (var (methodKey, alias) in Aliases)
            {
                if (BuilderMethodsByNameAndArgc.TryGetValue(methodKey, out var scriptMethod))
                {
                    // don't apply aliases when we have an actual method for the alias already
                    if (!BuilderMethodsByNameAndArgc.ContainsKey(new MethodKey(alias, methodKey.Argc)))
                    {
                        Register(alias, methodKey.Argc, scriptMethod);
                        aliasesRegistered++;
                    }
                }
            }

            MethodsByNameAndArgc = BuilderMethodsByNameAndArgc.ToFrozenDictionary();

            var methods = MethodsByNameAndArgc
                    .Select(x => (x.Key.Name, x.Key.Argc, x.Value))
                    .ToArray();
            ScriptMethodCache<T>.Initialize(methods);

            return this;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool ContainsMethod(scoped in godot_string_name name) => _knownMethodNames.Contains(name._data);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool TryGetMethod(scoped in godot_string_name name, int argumentCount, out ScriptMethod<T> method) =>
            ScriptMethodCache<T>.TryGet(name._data, argumentCount, out method);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ref readonly ScriptMethod<GodotObject> TryGetMethodFast(scoped in godot_string_name name, int argumentCount)
        {
            return ref ScriptMethodCache<T>.TryGetFast(name._data, argumentCount);
        }
    }

    public static class ScriptMethodRegistryExtensions
    {
        // This is an extension method because C# does not allow additional type constraints for an already existing T
        public static ScriptMethodRegistry<T> Register<T, TBase>(this ScriptMethodRegistry<T> registry, ScriptMethodRegistry<TBase> baseTypeRegistry)
            where TBase : GodotObject
            where T : GodotObject, TBase
        {
            foreach (var (methodKey, alias) in baseTypeRegistry.Aliases)
            {
                registry.AddAlias(methodKey.Name, methodKey.Argc, alias);
            }

            foreach (var (methodKey, value) in baseTypeRegistry.MethodsByNameAndArgc)
            {
                registry.Register(methodKey.Name, methodKey.Argc, value);
            }

            return registry;
        }
    }
}
