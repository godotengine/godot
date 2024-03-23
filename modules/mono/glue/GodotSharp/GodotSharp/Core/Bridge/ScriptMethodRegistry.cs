using System;
using System.Collections.Generic;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate void ScriptMethod<in T>(T scriptInstance, NativeVariantPtrArgs args, out godot_variant ret)
        where T : GodotObject;

    public sealed class ScriptMethodRegistry<T> where T : GodotObject
    {
        internal Dictionary<(int argc, IntPtr methodName), ScriptMethod<T>> MethodsByNameAndArgc { get; } = new();
        internal Dictionary<(int argc, IntPtr methodName), IntPtr> Aliases { get; } = new();

        private readonly HashSet<IntPtr> _knownMethodNames = new();

        public ScriptMethodRegistry<T> AddAlias(StringName methodName, int argumentCount, StringName alias) =>
            AddAlias(methodName.NativeValue._data, argumentCount, alias.NativeValue._data);

        public ScriptMethodRegistry<T> Register(StringName methodName, int argumentCount, ScriptMethod<T> method) =>
            Register(methodName.NativeValue._data, argumentCount, method);

        internal ScriptMethodRegistry<T> AddAlias(IntPtr methodName, int argumentCount, IntPtr alias)
        {
            Aliases[(argumentCount, methodName)] = alias;
            return this;
        }

        internal ScriptMethodRegistry<T> Register(IntPtr methodName, int argumentCount, ScriptMethod<T> method)
        {
            MethodsByNameAndArgc[(argumentCount, methodName)] = method;
            _knownMethodNames.Add(methodName);
            return this;
        }

        public ScriptMethodRegistry<T> Compile()
        {
            foreach (var (source, alias) in Aliases)
            {
                if (MethodsByNameAndArgc.TryGetValue(source, out var scriptMethod))
                {
                    // don't apply aliases when we have an actual method for the alias already
                    if (!MethodsByNameAndArgc.ContainsKey((source.argc, alias)))
                    {
                        Register(alias, source.argc, scriptMethod);
                    }
                }
            }

            GD.Print($"Script method registry compiled for {typeof(T)}: size={MethodsByNameAndArgc.Count}, alias_size={Aliases.Count}");
            // TODO: I would like to discard _aliases now to free up memory, but the hierarchy above it still needs it
            //       There are probably lots of aliases, we could at least not copy them and recursively walk our parent
            //       hierarchy as it's only done once (here). Ideas are appreciated
            return this;
        }

        public bool ContainsMethod(in godot_string_name name) => _knownMethodNames.Contains(name._data);

        public bool TryGetMethod(in godot_string_name name, int argumentCount, out ScriptMethod<T> method) =>
            MethodsByNameAndArgc.TryGetValue((argumentCount, name._data), out method);
    }

    public static class ScriptMethodRegistryExtensions
    {
        // This is an extension method because C# does not allow additional type constraints for an already existing T
        public static ScriptMethodRegistry<T> Register<T, V>(this ScriptMethodRegistry<T> registry, ScriptMethodRegistry<V> baseTypeRegistry)
            where T : V where V : GodotObject
        {
            foreach (var ((argc, method), alias) in baseTypeRegistry.Aliases)
            {
                registry.AddAlias(method, argc, alias);
            }

            foreach (var ((argc, method), value) in baseTypeRegistry.MethodsByNameAndArgc)
            {
                registry.Register(method, argc, value);
            }

            return registry;
        }
    }
}
