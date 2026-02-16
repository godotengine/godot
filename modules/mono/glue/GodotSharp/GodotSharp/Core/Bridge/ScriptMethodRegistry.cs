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

        internal Dictionary<MethodKey, IntPtr> Aliases { get; } = new();

        private readonly HashSet<IntPtr> _knownMethodNames = new();

        public ScriptMethodRegistry<T> AddAlias(StringName methodName, int argumentCount, StringName alias) =>
            AddAlias(methodName.NativeValue._data, argumentCount, alias.NativeValue._data);

        public ScriptMethodRegistry<T> Register(StringName methodName, int argumentCount, ScriptMethod<GodotObject> method) =>
            Register(methodName.NativeValue._data, argumentCount, method);

        internal ScriptMethodRegistry<T> AddAlias(IntPtr methodName, int argumentCount, IntPtr alias)
        {
            Aliases[new MethodKey(methodName, argumentCount)] = alias;
            return this;
        }

        internal ScriptMethodRegistry<T> Register(IntPtr methodName, int argumentCount, ScriptMethod<GodotObject> method)
        {
            BuilderMethodsByNameAndArgc[new MethodKey(methodName, argumentCount)] = method;
            _knownMethodNames.Add(methodName);
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
        public bool ContainsMethod(in godot_string_name name) => _knownMethodNames.Contains(name._data);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool TryGetMethod(in godot_string_name name, int argumentCount, out ScriptMethod<T> method) =>
            ScriptMethodCache<T>.TryGet(name._data, argumentCount, out method);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ref readonly ScriptMethod<GodotObject> TryGetMethodFast(in godot_string_name name, int argumentCount)
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
