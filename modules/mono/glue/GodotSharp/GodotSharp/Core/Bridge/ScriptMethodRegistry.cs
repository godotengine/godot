using Godot.NativeInterop;
using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Godot.Bridge
{
    public sealed class ScriptMethodRegistry<T> :
        ScriptRegistry<T, ScriptMethod<GodotObject>, ScriptCache<T, ScriptMethod<GodotObject>>,
            ScriptMethodRegistry<T>>
        where T : GodotObject
    {

    }

    public sealed class ScriptPropertyRegistry<T> :
        ScriptRegistry<T, PropertyAccessMethod<GodotObject>, ScriptCache<T, PropertyAccessMethod<GodotObject>>,
            ScriptPropertyRegistry<T>>
        where T : GodotObject
    {

    }
    public abstract class ScriptRegistry<T, TMethod, TCache, TSelf>
        where T : GodotObject
        where TCache : ScriptCache<T, TMethod>
        where TSelf : ScriptRegistry<T, TMethod, TCache, TSelf>
    {
        internal Dictionary<MethodKey, TMethod> BuilderMethodsByNameAndArgc = new();

        internal FrozenDictionary<MethodKey, TMethod> MethodsByNameAndArgc;

        internal Dictionary<MethodKey, StringName> Aliases { get; } = new();

        private readonly HashSet<IntPtr> _knownMethodNames = new();

        public TSelf AddAlias(StringName methodName, int argumentCount, StringName alias)
        {
            Aliases[new MethodKey(methodName, argumentCount)] = alias;
            return (TSelf)this;
        }

        public TSelf Register(StringName methodName, int argumentCount, TMethod method)
        {
            BuilderMethodsByNameAndArgc[new MethodKey(methodName, argumentCount)] = method;
            _knownMethodNames.Add(methodName.NativeValue._data);
            return (TSelf)this;
        }

        public TSelf Build()
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
            ScriptCache<T, TMethod>.Initialize(methods);

            return (TSelf)this;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool ContainsMethod(scoped in godot_string_name name) => _knownMethodNames.Contains(name._data);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ref readonly TMethod GetMethodOrNullRef(scoped in godot_string_name name, int argumentCount)
        {
            return ref ScriptCache<T, TMethod>.GetOrNullRef(name._data, argumentCount);
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

    public static class ScriptPropertyRegistryExtensions
    {
        // This is an extension method because C# does not allow additional type constraints for an already existing T
        public static ScriptPropertyRegistry<T> Register<T, TBase>(this ScriptPropertyRegistry<T> registry, ScriptPropertyRegistry<TBase> baseTypeRegistry)
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
