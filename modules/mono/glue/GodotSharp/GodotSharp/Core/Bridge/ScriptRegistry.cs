using Godot.NativeInterop;
using System;
using System.Collections.Frozen;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Godot.Bridge
{
    public abstract class ScriptRegistry<T, TMethod, TCache, TSelf>
        where T : GodotObject
        where TCache : ScriptCache<TMethod>
        where TSelf : ScriptRegistry<T, TMethod, TCache, TSelf>
    {
        private static ScriptCache<TMethod> _cache;

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
            _knownMethodNames.Add(methodName.NativeValue.GetIntPtr());
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
                .Select(x => (x.Key, x.Value))
                .ToArray();

            _cache = InitializeCache(methods);

            return (TSelf)this;
        }

        protected abstract ScriptCache<TMethod> InitializeCache((MethodKey, TMethod)[] methods);

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public bool ContainsName(scoped in godot_string_name name) => _knownMethodNames.Contains(name.GetIntPtr());

        /// <summary>
        /// Get the script method.
        /// </summary>
        /// <param name="name">Name of the script method</param>
        /// <param name="argumentCount">Argument count</param>
        /// <returns>Reference to script method</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public ref readonly TMethod GetMethodOrNullRef(scoped in godot_string_name name, int argumentCount)
        {
            return ref _cache.GetOrNullRef(name.GetIntPtr(), argumentCount);
        }
    }
}
