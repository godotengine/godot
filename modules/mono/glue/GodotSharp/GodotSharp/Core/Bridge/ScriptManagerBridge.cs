#nullable enable

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using Godot.NativeInterop;
using Godot.NativeInterop.UnsafeCollections;
using JetBrains.Annotations;

namespace Godot.Bridge
{
    /// <summary>
    /// Manages the mapping between Godot script resources and C# types, as well as other related metadata.
    /// </summary>
    public static partial class ScriptManagerBridge
    {
        private static readonly ConcurrentDictionary<AssemblyLoadContext, ConcurrentDictionary<Type, byte>>
            _alcData = new();

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void OnAlcUnloading(AssemblyLoadContext alc)
        {
            if (_alcData.TryRemove(alc, out var typesInAlc))
            {
                foreach (var type in typesInAlc.Keys)
                {
                    if (_scriptTypeBiMap.RemoveByScriptType(type, out IntPtr scriptPtr) &&
                        (!_pathTypeBiMap.TryGetScriptPath(type, out string? scriptPath) ||
                         scriptPath.StartsWith("csharp://", StringComparison.Ordinal)))
                    {
                        // For scripts without a path, we need to keep the class qualified name for reloading
                        _scriptDataForReload.TryAdd(scriptPtr,
                            (type.Assembly.GetName().Name, type.FullName ?? type.ToString()));
                    }

                    _pathTypeBiMap.RemoveByScriptType(type);
                    _scriptTypeMetaMap.TryRemove(type, out _);
                }
            }
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void AddTypeForAlcReloading(Type type)
        {
            var alc = AssemblyLoadContext.GetLoadContext(type.Assembly);
            if (alc == null)
                return;

            var typesInAlc = _alcData.GetOrAdd(alc,
                static alc =>
                {
                    alc.Unloading += OnAlcUnloading;
                    return new();
                });
            typesInAlc.TryAdd(type, 0);
        }

        /// <summary>
        /// Tracks the given AssemblyLoadContext for unloading, so that when it's unloaded we can
        /// do the necessary cleanup of the types that we keep track of. This is needed to prevent
        /// the references to the types from keeping the ALC alive and thus preventing unloading.
        /// </summary>
        /// <param name="alc">The AssemblyLoadContext to track for unloading.</param>
        [PublicAPI]
        [MethodImpl(MethodImplOptions.NoInlining)]
        public static void TrackAlcForUnloading(AssemblyLoadContext alc)
        {
            _ = _alcData.GetOrAdd(alc,
                static alc =>
                {
                    alc.Unloading += OnAlcUnloading;
                    return new();
                });
        }

        private static readonly ScriptTypeBiMap _scriptTypeBiMap = new();
        private static readonly PathScriptTypeBiMap _pathTypeBiMap = new();

        private static readonly ConcurrentDictionary<IntPtr, (string? assemblyName, string classFullName)>
            _scriptDataForReload = new();

        private delegate bool LegacyScriptTypeMetaResolver(Type scriptType,
            [MaybeNullWhen(false)] out ScriptTypeMeta scriptTypeMeta);

        private static List<LegacyScriptTypeMetaResolver>? _legacyScriptTypeMetaResolvers;

        private static void AddLegacyScriptTypeMetaLookupDelegate(LegacyScriptTypeMetaResolver legacyResolver) =>
            (_legacyScriptTypeMetaResolvers ??= []).Add(legacyResolver);

        private static readonly ConcurrentDictionary<Type, ScriptTypeMeta> _scriptTypeMetaMap = new();

        private static ScriptTypeMeta GetOrAddScriptTypeMeta(ScriptTypeMeta scriptTypeMeta)
        {
            // No worth locking, but need to use TryAdd in case another thread finishes first.

            if (!_scriptTypeMetaMap.TryAdd(scriptTypeMeta.Type, scriptTypeMeta))
                // We're not the ones who added it, so no need to call AddTypeForAlcReloading.
                return _scriptTypeMetaMap[scriptTypeMeta.Type];

            // Successfully added by us, so we need to call AddTypeForAlcReloading.
            if (AlcReloadCfg.IsAlcReloadingEnabled)
                AddTypeForAlcReloading(scriptTypeMeta.Type);

            return scriptTypeMeta;
        }

        private static ScriptTypeMeta GetOrResolveScriptTypeMeta(Type scriptType)
        {
            if (_scriptTypeMetaMap.TryGetValue(scriptType, out var scriptTypeMeta))
                return scriptTypeMeta;

            // No worth locking, but need to use GetOrAddScriptTypeMeta in case another thread finishes first.

            if (scriptType.GetCustomAttribute<ScriptTypeMetaProviderBaseAttribute>() is { } providerAttr)
                return GetOrAddScriptTypeMeta(providerAttr.GetGodotClassScriptMeta(scriptType));

            if (_legacyScriptTypeMetaResolvers != null)
            {
                foreach (var legacyResolver in _legacyScriptTypeMetaResolvers)
                {
                    if (legacyResolver(scriptType, out scriptTypeMeta))
                        return GetOrAddScriptTypeMeta(scriptTypeMeta);
                }
            }

            throw new ArgumentException(
                $"The type '{scriptType.FullName}' does not have a ScriptTypeMetaProviderBaseAttribute.",
                nameof(scriptType));
        }

        private static bool TryGetOrResolveScriptTypeMeta(Type scriptType,
            [MaybeNullWhen(false)] out ScriptTypeMeta scriptTypeMeta)
        {
            if (_scriptTypeMetaMap.TryGetValue(scriptType, out scriptTypeMeta))
                return true;

            // No worth locking, but need to use GetOrAddScriptTypeMeta in case another thread finishes first.

            if (scriptType.GetCustomAttribute<ScriptTypeMetaProviderBaseAttribute>() is { } providerAttr)
            {
                scriptTypeMeta = GetOrAddScriptTypeMeta(providerAttr.GetGodotClassScriptMeta(scriptType));
                return true;
            }

            if (_legacyScriptTypeMetaResolvers != null)
            {
                foreach (var legacyResolver in _legacyScriptTypeMetaResolvers)
                {
                    if (legacyResolver(scriptType, out var resolvedScriptTypeMeta))
                    {
                        scriptTypeMeta = GetOrAddScriptTypeMeta(resolvedScriptTypeMeta);
                        return true;
                    }
                }
            }

            scriptTypeMeta = null;
            return false;
        }

        internal static ScriptTypeMeta? GetOrResolveScriptTypeMetaOrNull(Type type)
        {
            // ReSharper disable once CanSimplifyDictionaryTryGetValueWithGetValueOrDefault
            return TryGetOrResolveScriptTypeMeta(type, out var scriptTypeMeta)
                ? scriptTypeMeta
                : null;
        }

        [UnmanagedCallersOnly]
        internal static void FrameCallback()
        {
            try
            {
                Dispatcher.DefaultGodotTaskScheduler?.Activate();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe IntPtr CreateManagedForGodotObjectBinding(godot_string_name* nativeTypeName,
            IntPtr godotObject)
        {
            try
            {
                using var stringName = StringName.CreateConsuming(
                    NativeFuncs.godotsharp_string_name_new_copy(CustomUnsafe.AsRef(nativeTypeName)));
                string nativeTypeNameStr = stringName.ToString();

                var instance = Constructors.Invoke(nativeTypeNameStr, godotObject);

                return GCHandle.ToIntPtr(CustomGCHandle.AllocStrong(instance));
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return IntPtr.Zero;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CreateManagedForGodotObjectScriptInstanceWithTrampoline(
            ConstructorTrampolineDelegate constructorTrampoline, IntPtr godotObjectPtr,
            godot_variant** args, int argCount)
        {
            try
            {
                _ = constructorTrampoline(godotObjectPtr, new NativeVariantPtrArgs(args, argCount));
                return godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetGlobalClassName(godot_string* scriptPath, godot_string* outBaseType,
            godot_string* outIconPath, godot_bool* outIsAbstract, godot_bool* outIsTool, godot_string* outClassName)
        {
            // This method must always return the outBaseType for every script, even if the script is
            // not a global class. But if the script is not a global class it must return an empty
            // outClassName string since it should not have a name.
            string scriptPathStr = Marshaling.ConvertStringToManaged(*scriptPath);
            Debug.Assert(!string.IsNullOrEmpty(scriptPathStr), "Script path can't be empty.");

            if (!(_pathTypeBiMap.TryGetScriptType(scriptPathStr, out Type? scriptType)
                  && TryGetOrResolveScriptTypeMeta(scriptType, out var scriptTypeMeta)))
            {
                // Script at the given path does not exist, or it's not a C# type.
                // This is fine, it may be a path to a generic script and those can't be global classes.
                *outClassName = default;
                return;
            }

            if (outIconPath != null)
            {
                IconAttribute? iconAttr = scriptType.GetCustomAttributes(inherit: false)
                    .OfType<IconAttribute>()
                    .FirstOrDefault();

                if (!string.IsNullOrEmpty(iconAttr?.Path))
                {
                    string iconPath = iconAttr.Path.IsAbsolutePath()
                        ? iconAttr.Path.SimplifyPath()
                        : scriptPathStr.GetBaseDir().PathJoin(iconAttr.Path).SimplifyPath();
                    *outIconPath = Marshaling.ConvertStringToNative(iconPath);
                }
            }

            if (outBaseType != null)
            {
                bool foundGlobalBaseScript = false;

                Type native = scriptTypeMeta.NativeType;
                Type? top = scriptType.BaseType;

                while (top != null && top != native)
                {
                    if (IsGlobalClass(top))
                    {
                        *outBaseType = Marshaling.ConvertStringToNative(top.Name);
                        foundGlobalBaseScript = true;
                        break;
                    }

                    top = top.BaseType;
                }

                if (!foundGlobalBaseScript)
                {
                    string nativeName = native.GetCustomAttribute<GodotClassNameAttribute>(false)?.Name ?? native.Name;
                    *outBaseType = Marshaling.ConvertStringToNative(nativeName);
                }
            }

            if (outIsAbstract != null)
            {
                *outIsAbstract = scriptType.IsAbstract.ToGodotBool();
            }

            if (outIsTool != null)
            {
                *outIsTool = Attribute.IsDefined(scriptType, typeof(ToolAttribute)).ToGodotBool();
            }

            if (!IsGlobalClass(scriptType))
            {
                // Scripts that are not global classes should not have a name.
                // Return an empty string to prevent the class from being registered
                // as a global class in the editor.
                *outClassName = default;
                return;
            }

            *outClassName = Marshaling.ConvertStringToNative(scriptType.Name);

            static bool IsGlobalClass(Type scriptType) =>
                scriptType.IsDefined(typeof(GlobalClassAttribute), inherit: false);
        }

        [UnmanagedCallersOnly]
        internal static void SetGodotObjectPtr(IntPtr gcHandlePtr, IntPtr newPtr)
        {
            try
            {
                var target = (GodotObject?)GCHandle.FromIntPtr(gcHandlePtr).Target;
                if (target != null)
                    target.NativePtr = newPtr;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        /// <summary>
        /// Registers a C# type as a Godot script type, associating it with the given script path.
        /// </summary>
        /// <param name="type">The type to associate with the script path.</param>
        /// <param name="scriptPath">The script path to associate with the type.</param>
        /// <exception cref="ArgumentException">Thrown when the type is already registered.</exception>
        /// <remarks>
        /// <para>This allows the type to be used as a script in Godot. The script path is used to associate
        /// the type with a specific script resource in Godot, and must be unique for each type.</para>
        /// <para>It's expected that this method will be called during initialization or after a reload.</para>
        /// </remarks>
        [PublicAPI]
        public static void RegisterScriptPathForType(Type type, string scriptPath)
        {
            _pathTypeBiMap.Add(scriptPath, type);

            if (AlcReloadCfg.IsAlcReloadingEnabled)
                AddTypeForAlcReloading(type);

            // This method may be called before initialization.
            if (NativeFuncs.godotsharp_dotnet_module_is_initialized().ToBool() && Engine.IsEditorHint())
            {
                // This is necessary after reloading to ensure that new scripts
                // are included in the global class list by updating EFS for them.
                QueueScriptPathAssociationEfsUpdate();
            }
        }

        /// <summary>
        /// Registers a generic type definition as a Godot script type, associating it with the given script path.
        /// </summary>
        /// <param name="genericTypeDefinition">The generic type definition to register.</param>
        /// <param name="scriptPath">The script path to associate with the type.</param>
        /// <exception cref="ArgumentException">
        /// Thrown when the type is already registered or is not a generic type definition.
        /// </exception>
        /// <remarks>
        /// <para>While generic type definitions can't be instantiated as scripts in Godot,
        /// their constructed generic types can be, and they will look for the script path
        /// of their generic type definition to find the script resource in Godot.</para>
        /// <para>Generic types must register both their generic type definition and their
        /// constructed generic types with <see cref="RegisterScriptPathForGenericTypeDefinition"/>
        /// and <see cref="ScriptTypeMetaProviderAttribute{T}"/> respectively.</para>
        /// </remarks>
        [PublicAPI]
        public static void RegisterScriptPathForGenericTypeDefinition(Type genericTypeDefinition, string scriptPath)
        {
            if (!genericTypeDefinition.IsGenericTypeDefinition)
                throw new ArgumentException(
                    $"Argument {nameof(genericTypeDefinition)} is not a generic type definition.");

            _pathTypeBiMap.Add(scriptPath, genericTypeDefinition);

            if (AlcReloadCfg.IsAlcReloadingEnabled)
                AddTypeForAlcReloading(genericTypeDefinition);

            // This method may be called before initialization.
            if (NativeFuncs.godotsharp_dotnet_module_is_initialized().ToBool() && Engine.IsEditorHint())
            {
                // This is necessary after reloading to ensure that new scripts
                // are included in the global class list by updating EFS for them.
                QueueScriptPathAssociationEfsUpdate();
            }
        }

        private static bool _isScriptPathAssociationEfsUpdateQueued;
        private static readonly object _queueScriptPathAssociationEfsUpdateLock = new();

        /// <summary>
        /// Queues an editor file system update for all registered script paths.
        /// </summary>
        /// <remarks>
        /// <para>This is necessary after reloading to ensure that new scripts are included
        /// in the global class list by updating the editor file system for them.</para>
        /// <para>It's expected that the editor file system is updated only once for each assembly load context,
        /// as it's a very expensive operation. Our source generators comply with this, as all associations
        /// between types and script paths are only registered once in batch from a single method.</para>
        /// </remarks>
        private static void QueueScriptPathAssociationEfsUpdate()
        {
            // Quick check to avoid locking. If used correctly (by doing batch registration),
            // this will be the exit for all calls during except for the first one.
            if (_isScriptPathAssociationEfsUpdateQueued)
                return;

            // Locking shouldn't be needed as associations between types and script paths are only
            // registered once in batch from a single method by our source generators, but we lock
            // any way just to be safe, since public APIs can be misused. When misused, the extra
            // cost of locking is insignificant next to the cost of updating the editor file system.
            lock (_queueScriptPathAssociationEfsUpdateLock)
            {
                // Between the previous check and the locking, another thread might have locked, queued and unlocked.
                if (_isScriptPathAssociationEfsUpdateQueued)
                    return;

                _isScriptPathAssociationEfsUpdateQueued = true;

                Callable.From(() =>
                {
                    _isScriptPathAssociationEfsUpdateQueued = false;

                    // Since we assume this will only be queued once during initialization,
                    // we don't need to worry about which script paths to update. Just update
                    // them all. If this is queued more than once, that's considered misuse,
                    // and in such case the cost of file system updating is already expensive.
                    string[] scriptPaths = _pathTypeBiMap.Paths.ToArray();
                    using godot_packed_string_array scriptPathsNative =
                        Marshaling.ConvertSystemArrayToNativePackedStringArray(scriptPaths);
                    NativeFuncs.godotsharp_internal_editor_file_system_update_files(scriptPathsNative);
                }).CallDeferred();
            }
        }

        /// <summary>
        /// Initializes the registry of constructors for native type C# classes.
        /// </summary>
        /// <remarks>
        /// This is necessary to ensure that the constructors are available when needed
        /// for creating instances of native type C# classes from C++.
        /// </remarks>
        [PublicAPI]
        public static void InitializeNativeClassConstructors()
        {
            Constructors.Initialize();
        }

        [UnmanagedCallersOnly]
        internal static unsafe void LegacyRaiseEventSignal(IntPtr ownerGCHandlePtr,
            godot_string_name* eventSignalName, godot_variant** args, int argCount, godot_bool* refOwnerIsNull)
        {
            try
            {
                var owner = (GodotObject?)GCHandle.FromIntPtr(ownerGCHandlePtr).Target;

                if (owner == null)
                {
                    *refOwnerIsNull = godot_bool.True;
                    return;
                }

                *refOwnerIsNull = godot_bool.False;

                owner.RaiseGodotClassSignalCallbacks(CustomUnsafe.AsRef(eventSignalName),
                    new NativeVariantPtrArgs(args, argCount));
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *refOwnerIsNull = godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void RaiseEventSignalViaTrampoline(
            RaiseSignalTrampolineDelegate raiseSignalTrampoline,
            IntPtr ownerGCHandlePtr, godot_variant** args, int argCount, godot_bool* refOwnerIsNull)
        {
            try
            {
                object? owner = GCHandle.FromIntPtr(ownerGCHandlePtr).Target;

                if (owner == null)
                {
                    *refOwnerIsNull = godot_bool.True;
                    return;
                }

                *refOwnerIsNull = godot_bool.False;

                raiseSignalTrampoline(owner, new NativeVariantPtrArgs(args, argCount));
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *refOwnerIsNull = godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static godot_bool ScriptIsOrInherits(IntPtr scriptPtr, IntPtr scriptPtrMaybeBase)
        {
            try
            {
                if (!_scriptTypeBiMap.TryGetScriptType(scriptPtr, out Type? scriptType))
                    return godot_bool.False;

                if (!_scriptTypeBiMap.TryGetScriptType(scriptPtrMaybeBase, out Type? maybeBaseType))
                    return godot_bool.False;

                return (scriptType == maybeBaseType || maybeBaseType.IsAssignableFrom(scriptType)).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool AddScriptBridge(IntPtr scriptPtr, godot_string* scriptPath)
        {
            try
            {
                string scriptPathStr = Marshaling.ConvertStringToManaged(*scriptPath);
                return AddScriptBridgeCore(scriptPtr, scriptPathStr).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        private static bool AddScriptBridgeCore(IntPtr scriptPtr, string scriptPath)
        {
            _scriptTypeBiMap.ReadWriteLock.EnterUpgradeableReadLock();
            try
            {
                if (!_scriptTypeBiMap.IsScriptRegistered(scriptPtr))
                {
                    if (!_pathTypeBiMap.TryGetScriptType(scriptPath, out Type? scriptType))
                        return false;

                    _scriptTypeBiMap.ReadWriteLock.EnterWriteLock();
                    try
                    {
                        _scriptTypeBiMap.Add(scriptPtr, scriptType);
                    }
                    finally
                    {
                        _scriptTypeBiMap.ReadWriteLock.ExitWriteLock();
                    }
                }
            }
            finally
            {
                _scriptTypeBiMap.ReadWriteLock.ExitUpgradeableReadLock();
            }

            return true;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetOrCreateScriptBridgeForPath(godot_string* scriptPath, godot_ref* outScript)
        {
            string scriptPathStr = Marshaling.ConvertStringToManaged(*scriptPath);

            if (!_pathTypeBiMap.TryGetScriptType(scriptPathStr, out Type? scriptType))
            {
                NativeFuncs.godotsharp_internal_new_csharp_script(outScript);
                return;
            }

            Debug.Assert(!scriptType.IsGenericTypeDefinition,
                $"Cannot get or create script for a generic type definition '{scriptType.FullName}'. Path: '{scriptPathStr}'.");

            GetOrCreateScriptBridgeForType(scriptType, outScript);
        }

        private static unsafe void GetOrCreateScriptBridgeForType(Type scriptType, godot_ref* outScript)
        {
            _scriptTypeBiMap.ReadWriteLock.EnterUpgradeableReadLock();
            try
            {
                if (_scriptTypeBiMap.TryGetScriptPtr(scriptType, out IntPtr scriptPtr))
                {
                    // Use existing
                    NativeFuncs.godotsharp_ref_new_from_ref_counted_ptr(out *outScript, scriptPtr);
                    return;
                }

                // This path is slower, but it's only executed for the first instantiation of the type
                CreateScriptBridgeForType(scriptType, outScript);
            }
            finally
            {
                _scriptTypeBiMap.ReadWriteLock.ExitUpgradeableReadLock();
            }

            NativeFuncs.godotsharp_internal_reload_registered_script(outScript->Reference);
        }

        internal static unsafe void GetOrLoadOrCreateScriptForType(Type scriptType, godot_ref* outScript)
        {
            static bool GetPathOtherwiseGetOrCreateScript(Type scriptType, godot_ref* outScript,
                [MaybeNullWhen(false)] out string scriptPath)
            {
                _scriptTypeBiMap.ReadWriteLock.EnterUpgradeableReadLock();
                try
                {
                    if (_scriptTypeBiMap.TryGetScriptPtr(scriptType, out IntPtr scriptPtr))
                    {
                        // Use existing
                        NativeFuncs.godotsharp_ref_new_from_ref_counted_ptr(out *outScript, scriptPtr);
                        scriptPath = null;
                        return false;
                    }

                    // This path is slower, but it's only executed for the first instantiation of the type

                    if (_pathTypeBiMap.TryGetScriptPath(scriptType, out scriptPath))
                        return true;

                    if (scriptType.IsConstructedGenericType)
                    {
                        // If the script type is generic, also try looking for the path of the generic type definition
                        // since we can use it to create the script.
                        Type genericTypeDefinition = scriptType.GetGenericTypeDefinition();
                        if (_pathTypeBiMap.TryGetGenericTypeDefinitionPath(genericTypeDefinition, out scriptPath))
                            return true;
                    }

                    CreateScriptBridgeForType(scriptType, outScript);
                    scriptPath = null;
                }
                finally
                {
                    _scriptTypeBiMap.ReadWriteLock.ExitUpgradeableReadLock();
                }

                NativeFuncs.godotsharp_internal_reload_registered_script(outScript->Reference);
                return false;
            }

            static string GetVirtualConstructedGenericTypeScriptPath(Type scriptType, string scriptPath)
            {
                // Constructed generic types all have the same path which is not allowed by Godot
                // (every Resource must have a unique path). So we create a unique "virtual" path
                // for each type.

                if (!scriptPath.StartsWith("res://", StringComparison.Ordinal))
                {
                    throw new ArgumentException("Script path must start with 'res://'.", nameof(scriptPath));
                }

                scriptPath = scriptPath.Substring("res://".Length);
                return $"csharp://{scriptPath}:{scriptType}.cs";
            }

            if (GetPathOtherwiseGetOrCreateScript(scriptType, outScript, out string? scriptPath))
            {
                // This path is slower, but it's only executed for the first instantiation of the type

                if (scriptType.IsConstructedGenericType &&
                    !scriptPath.StartsWith("csharp://", StringComparison.Ordinal))
                {
                    // If the script type is generic it can't be loaded using the real script path.
                    // Construct a virtual path unique to this constructed generic type and add it
                    // to the path bimap so they can be found later by their virtual path.
                    // IMPORTANT: The virtual path must be added to _pathTypeBiMap before the first
                    // load of the script, otherwise the loaded script won't be added to _scriptTypeBiMap.
                    scriptPath = GetVirtualConstructedGenericTypeScriptPath(scriptType, scriptPath);

                    _scriptTypeBiMap.ReadWriteLock.EnterWriteLock();
                    try
                    {
                        _pathTypeBiMap.Add(scriptPath, scriptType);
                    }
                    finally
                    {
                        _scriptTypeBiMap.ReadWriteLock.ExitWriteLock();
                    }
                }

                // This must be done outside the read-write lock, as the script resource loading can lock it
                using godot_string scriptPathIn = Marshaling.ConvertStringToNative(scriptPath);
                if (!NativeFuncs.godotsharp_internal_script_load(scriptPathIn, outScript).ToBool())
                {
                    GD.PushError($"Cannot load script for type '{scriptType.FullName}'. Path: '{scriptPath}'.");

                    // If loading of the script fails, best we can do create a new script
                    // with no path, as we do for types without an associated script file.
                    GetOrCreateScriptBridgeForType(scriptType, outScript);
                }

                if (scriptType.IsConstructedGenericType)
                {
                    // When reloading generic scripts they won't be added to the script bimap because their
                    // virtual path won't be in the path bimap yet. The current method executes when a derived type
                    // is trying to get or create the script for their base type. The code above has now added
                    // the virtual path to the path bimap and loading the script with that path should retrieve
                    // any existing script, so now we have a chance to make sure it's added to the script bimap.
                    AddScriptBridgeCore(outScript->Reference, scriptPath);
                }
            }
        }

        /// <summary>
        /// WARNING: We need to make sure that after unlocking the bimap, we call godotsharp_internal_reload_registered_script
        /// </summary>
        private static unsafe void CreateScriptBridgeForType(Type scriptType, godot_ref* outScript)
        {
            Debug.Assert(!scriptType.IsGenericTypeDefinition,
                $"Script type must be a constructed generic type or not generic at all. Type: {scriptType}.");

            _scriptTypeBiMap.ReadWriteLock.EnterWriteLock();
            try
            {
                NativeFuncs.godotsharp_internal_new_csharp_script(outScript);
                IntPtr scriptPtr = outScript->Reference;

                _scriptTypeBiMap.Add(scriptPtr, scriptType);
            }
            finally
            {
                _scriptTypeBiMap.ReadWriteLock.ExitWriteLock();
            }
        }

        [UnmanagedCallersOnly]
        internal static void RemoveScriptBridge(IntPtr scriptPtr)
        {
            _scriptTypeBiMap.ReadWriteLock.EnterWriteLock();
            try
            {
                _scriptTypeBiMap.Remove(scriptPtr);
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
            finally
            {
                _scriptTypeBiMap.ReadWriteLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Provides trimmer-safe access to unmanaged callables annotated with <see cref="RequiresUnreferencedCodeAttribute"/>.
        /// </summary>
        /// <remarks>
        /// The trimmer analyzer doesn't warn when unsafely taking the address of a method that's
        /// annotated with <see cref="RequiresUnreferencedCodeAttribute"/>. By wrapping that method
        /// with private access inside this class, we make it impossible to take its address.
        /// Instead, we provide a getter method that's annotated with <see cref="RequiresUnreferencedCodeAttribute"/>.
        /// </remarks>
        internal static class ToolsBuildUnmanagedCallables
        {
            [RequiresUnreferencedCode(
                "This method is for internal use by the Godot editor only. "
                + "It searches for the script type at runtime, which is not compatible with trimming.")]
            [UnmanagedCallersOnly]
            private static godot_bool TryReloadRegisteredScriptWithClass(IntPtr scriptPtr)
            {
                _scriptTypeBiMap.ReadWriteLock.EnterUpgradeableReadLock();
                try
                {
                    if (_scriptTypeBiMap.TryGetScriptType(scriptPtr, out _))
                    {
                        // NOTE:
                        // Currently, we reload all scripts, not only the ones from the unloaded ALC.
                        // As such, we need to handle this case instead of treating it as an error.
                        NativeFuncs.godotsharp_internal_reload_registered_script(scriptPtr);
                        return godot_bool.True;
                    }

                    if (!_scriptDataForReload.TryGetValue(scriptPtr, out var dataForReload))
                    {
                        GD.PushError("Missing class qualified name for reloading script");
                        return godot_bool.False;
                    }

                    _ = _scriptDataForReload.TryRemove(scriptPtr, out _);

                    if (dataForReload.assemblyName == null)
                    {
                        GD.PushError(
                            $"Missing assembly name of class '{dataForReload.classFullName}' for reloading script");
                        return godot_bool.False;
                    }

                    var scriptType = ReflectionUtils.FindTypeInLoadedAssemblies(dataForReload.assemblyName,
                        dataForReload.classFullName);

                    if (scriptType == null)
                    {
                        // The class was removed, can't reload
                        return godot_bool.False;
                    }

                    if (!typeof(GodotObject).IsAssignableFrom(scriptType))
                    {
                        // The class no longer inherits GodotObject, can't reload
                        return godot_bool.False;
                    }

                    _scriptTypeBiMap.ReadWriteLock.EnterWriteLock();
                    try
                    {
                        _scriptTypeBiMap.Add(scriptPtr, scriptType);
                    }
                    finally
                    {
                        _scriptTypeBiMap.ReadWriteLock.ExitWriteLock();
                    }

                    NativeFuncs.godotsharp_internal_reload_registered_script(scriptPtr);

                    return godot_bool.True;
                }
                catch (Exception e)
                {
                    ExceptionUtils.LogException(e);
                    return godot_bool.False;
                }
                finally
                {
                    _scriptTypeBiMap.ReadWriteLock.ExitUpgradeableReadLock();
                }
            }

            [RequiresUnreferencedCode(
                "This method is for internal use by the Godot editor only. "
                + "The returned delegate points to a method that searches for the "
                + "script type at runtime, which is not compatible with trimming.")]
            public static unsafe
                delegate* unmanaged<IntPtr, godot_bool>
                GetAddressOfTryReloadRegisteredScriptWithClass()
                => &TryReloadRegisteredScriptWithClass;
        }

        private static unsafe void GetScriptTypeInfo(ScriptTypeMeta scriptTypeMeta, godot_csharp_type_info* outTypeInfo)
        {
            var scriptType = scriptTypeMeta.Type;

            godot_string className = Marshaling.ConvertStringToNative(ReflectionUtils.ConstructTypeName(scriptType));

            StringName nativeBase = scriptTypeMeta.NativeName;

            bool isTool = scriptType.IsDefined(typeof(ToolAttribute), inherit: false);

            // If the type is nested and the parent type is a tool script,
            // consider the nested type a tool script as well.
            if (!isTool && scriptType.IsNested)
            {
                isTool = scriptType.DeclaringType?.IsDefined(typeof(ToolAttribute), inherit: false) ?? false;
            }

            // Every script in the GodotTools assembly is a tool script.
            if (!isTool && scriptType.Assembly.GetName().Name == "GodotTools")
            {
                isTool = true;
            }

            bool isGlobalClass = scriptType.IsDefined(typeof(GlobalClassAttribute), inherit: false);

            var iconAttr = scriptType.GetCustomAttributes(inherit: false)
                .OfType<IconAttribute>()
                .FirstOrDefault();

            godot_string iconPath = Marshaling.ConvertStringToNative(iconAttr?.Path);

            outTypeInfo->ClassName = className;
            outTypeInfo->NativeBaseName = NativeFuncs.godotsharp_string_name_new_copy(
                (godot_string_name)nativeBase.NativeValue);
            outTypeInfo->IconPath = iconPath;
            outTypeInfo->IsTool = isTool.ToGodotBool();
            outTypeInfo->IsGlobalClass = isGlobalClass.ToGodotBool();
            outTypeInfo->IsAbstract = scriptType.IsAbstract.ToGodotBool();
            outTypeInfo->IsGenericTypeDefinition = scriptType.IsGenericTypeDefinition.ToGodotBool();
            outTypeInfo->IsConstructedGenericType = scriptType.IsConstructedGenericType.ToGodotBool();
        }

        [ThreadStatic] private static TrampolineCollectorPool? _cachedTrampolineCollectorPool;
        [ThreadStatic] private static RpcMethodCollector? _cachedRpcMethodCollector;

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptTrampolines(
            IntPtr scriptPtr, godot_bool* outShouldFallbackToLegacyTrampolines,
            TryAddConstructorTrampolineDelegate tryAddConstructorTrampoline,
            TryAddMethodTrampolineDelegate tryAddMethodTrampoline,
            TryAddPropertyTrampolineDelegate tryAddPropertyTrampoline,
            TryAddRaiseSignalTrampolineDelegate tryAddRaiseSignalTrampoline)
        {
            try
            {
                var scriptTypeMeta = GetOrResolveScriptTypeMeta(_scriptTypeBiMap.GetScriptType(scriptPtr));
                var scriptType = scriptTypeMeta.Type;
                Debug.Assert(!scriptType.IsGenericTypeDefinition,
                    $"Script type must be a constructed generic type or not generic at all. Type: {scriptType}.");

                *outShouldFallbackToLegacyTrampolines = scriptTypeMeta.ShouldFallbackToLegacyTrampolines.ToGodotBool();

                if (scriptTypeMeta.GetGodotClassTrampolines == null)
                    return;

                TrampolineCollectorPool collectorPool;

                if (_cachedTrampolineCollectorPool == null)
                {
                    _cachedTrampolineCollectorPool = new(
                        Collectors: new(
                            new(scriptPtr, tryAddConstructorTrampoline),
                            new(scriptPtr, tryAddMethodTrampoline),
                            new(scriptPtr, tryAddPropertyTrampoline),
                            new(scriptPtr, tryAddRaiseSignalTrampoline)),
                        CollectionOptions: new(IncludeAncestors: true) { CollectConstructors = true });

                    collectorPool = _cachedTrampolineCollectorPool.Value;
                }
                else
                {
                    collectorPool = _cachedTrampolineCollectorPool.Value;
                    collectorPool.Collectors.UpdateCollectors(scriptPtr,
                        tryAddConstructorTrampoline, tryAddMethodTrampoline,
                        tryAddPropertyTrampoline, tryAddRaiseSignalTrampoline);
                    // GetGodotClassTrampolines changes this before calling the ancestor, so set it again.
                    collectorPool.CollectionOptions.CollectConstructors = true;
                }

                scriptTypeMeta.GetGodotClassTrampolines.Invoke(
                    collectorPool.Collectors, collectorPool.CollectionOptions);
            }
            catch (Exception e)
            {
                *outShouldFallbackToLegacyTrampolines = godot_bool.True;

                ExceptionUtils.LogException(e);
            }
        }

        private static class PropertyInfoNameCache
        {
            public static readonly Variant Name = "name";
            public static readonly Variant Type = "type";
            public static readonly Variant Usage = "usage";
            public static readonly Variant ClassName = "class_name";
        }

        private static class MethodInfoNameCache
        {
            public static readonly Variant Name = "name";
            public static readonly Variant ReturnVal = "return_val";
            public static readonly Variant Params = "params";
            public static readonly Variant Flags = "flags";
        }

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptClassInfo(IntPtr scriptPtr, godot_csharp_type_info* outTypeInfo,
            godot_array* outMethodsDest, godot_dictionary* outRpcFunctionsDest, godot_dictionary* outEventSignalsDest,
            godot_ref* outBaseScript)
        {
            try
            {
                // Zero-initialize output parameters, in case an exception is thrown, to know if they've been set.
                *outTypeInfo = default;
                *outMethodsDest = default;
                *outRpcFunctionsDest = default;
                *outEventSignalsDest = default;
                *outBaseScript = default;

                var scriptTypeMeta = GetOrResolveScriptTypeMeta(_scriptTypeBiMap.GetScriptType(scriptPtr));
                var scriptType = scriptTypeMeta.Type;
                Debug.Assert(!scriptType.IsGenericTypeDefinition,
                    $"Script type must be a constructed generic type or not generic at all. Type: {scriptType}.");

                GetScriptTypeInfo(scriptTypeMeta, outTypeInfo);

                Type native = scriptTypeMeta.NativeType;

                // Methods

                *outMethodsDest = NativeFuncs.godotsharp_array_new();
                var methods = UnsafeGodotArray.CreateBorrowing(*outMethodsDest);

                if (scriptType != native)
                {
                    var methodList = scriptTypeMeta.GetGodotMethodList?.Invoke();

                    if (methodList != null)
                    {
                        foreach (var method in methodList)
                        {
                            using var methodInfo = UnsafeGodotDictionary.Create();

                            methodInfo.Add(MethodInfoNameCache.Name, method.Name);

                            using var returnVal = UnsafeGodotDictionary.Create();

                            returnVal.Add(PropertyInfoNameCache.Name, method.ReturnVal.Name);
                            returnVal.Add(PropertyInfoNameCache.Type, (int)method.ReturnVal.Type);
                            returnVal.Add(PropertyInfoNameCache.Usage, (int)method.ReturnVal.Usage);

                            if (method.ReturnVal.ClassName != null)
                            {
                                returnVal[PropertyInfoNameCache.ClassName] = method.ReturnVal.ClassName;
                            }

                            methodInfo.Add(MethodInfoNameCache.ReturnVal, returnVal.BorrowDisposable());

                            using var methodParams = UnsafeGodotArray.Create();

                            if (method.Arguments != null)
                            {
                                foreach (var param in method.Arguments)
                                {
                                    using var pinfo = UnsafeGodotDictionary.Create();

                                    pinfo.Add(PropertyInfoNameCache.Name, param.Name);
                                    pinfo.Add(PropertyInfoNameCache.Type, (int)param.Type);
                                    pinfo.Add(PropertyInfoNameCache.Usage, (int)param.Usage);

                                    if (param.ClassName != null)
                                    {
                                        pinfo[PropertyInfoNameCache.ClassName] = param.ClassName;
                                    }

                                    methodParams.Add(pinfo.BorrowDisposable());
                                }
                            }

                            methodInfo.Add(MethodInfoNameCache.Params, methodParams.BorrowDisposable());

                            methodInfo.Add(MethodInfoNameCache.Flags, (int)method.Flags);

                            methods.Target.Add(methodInfo.BorrowDisposable());
                        }
                    }
                }

                // RPC functions

                if (_cachedRpcMethodCollector == null)
                    _cachedRpcMethodCollector = new RpcMethodCollector();
                else
                    _cachedRpcMethodCollector.Reset();

                scriptTypeMeta.GetGodotRpcMethods?.Invoke(_cachedRpcMethodCollector);

                *outRpcFunctionsDest = _cachedRpcMethodCollector.RpcMethodsDict is { } rpcMethodsDict
                    ? NativeFuncs.godotsharp_dictionary_new_copy((godot_dictionary)rpcMethodsDict.NativeValue)
                    : NativeFuncs.godotsharp_dictionary_new();

                // Event signals

                *outEventSignalsDest = NativeFuncs.godotsharp_dictionary_new();
                var signals = UnsafeGodotDictionary.CreateBorrowing(*outEventSignalsDest);

                if (scriptType != native)
                {
                    var signalList = scriptTypeMeta.GetGodotSignalList?.Invoke();

                    if (signalList != null)
                    {
                        foreach (var signal in signalList)
                        {
                            string signalName = signal.Name;

                            if (signals.Target.ContainsKey(signalName))
                                continue;

                            using var signalParams = UnsafeGodotArray.Create();

                            if (signal.Arguments != null)
                            {
                                foreach (var param in signal.Arguments)
                                {
                                    using var pinfo = UnsafeGodotDictionary.Create();

                                    pinfo.Add(PropertyInfoNameCache.Name, param.Name);
                                    pinfo.Add(PropertyInfoNameCache.Type, (int)param.Type);
                                    pinfo.Add(PropertyInfoNameCache.Usage, (int)param.Usage);

                                    if (param.ClassName != null)
                                    {
                                        pinfo[PropertyInfoNameCache.ClassName] = param.ClassName;
                                    }

                                    signalParams.Add(pinfo.BorrowDisposable());
                                }
                            }

                            signals.Target.Add(signalName, signalParams.BorrowDisposable());
                        }
                    }
                }

                // Base script

                var baseType = scriptType.BaseType;
                if (baseType != null && baseType != native)
                {
                    GetOrLoadOrCreateScriptForType(baseType, outBaseScript);
                }
                else
                {
                    *outBaseScript = default;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                // We zero-initialized them at the start of this method. Replace them only
                // if they haven't been assigned a new instance. Otherwise, it would cause a leak.
                if (outMethodsDest->IsAllocated)
                    *outMethodsDest = NativeFuncs.godotsharp_array_new();
                if (outRpcFunctionsDest->IsAllocated)
                    *outRpcFunctionsDest = NativeFuncs.godotsharp_dictionary_new();
                if (outEventSignalsDest->IsAllocated)
                    *outEventSignalsDest = NativeFuncs.godotsharp_dictionary_new();
            }
        }

        /// <summary>
        /// This is used as a pool to avoid having to allocate multiple instances of the
        /// collectors and argument arrays when updating trampolines for a script.
        /// </summary>
        private record struct TrampolineCollectorPool(
            TrampolineCollectors Collectors,
            TrampolineCollectionOptions CollectionOptions);

#pragma warning disable IDE1006 // Naming rule violation
        // ReSharper disable once InconsistentNaming
        // ReSharper disable once NotAccessedField.Local
        [StructLayout(LayoutKind.Sequential)]
        private ref struct godotsharp_property_info
        {
            // Careful with padding...
            public godot_string_name Name; // Not owned
            public godot_string HintString;
            public int Type;
            public int Hint;
            public int Usage;
            public godot_bool Exported;

            public void Dispose()
            {
                HintString.Dispose();
            }
        }
#pragma warning restore IDE1006

        [UnmanagedCallersOnly]
        internal static unsafe void GetPropertyInfoList(IntPtr scriptPtr,
            delegate* unmanaged<IntPtr, godot_string*, void*, int, void> addPropInfoFunc)
        {
            try
            {
                var scriptTypeMeta = GetOrResolveScriptTypeMeta(_scriptTypeBiMap.GetScriptType(scriptPtr));

                if (scriptTypeMeta.GetGodotPropertyList == null)
                    return;

                var properties = scriptTypeMeta.GetGodotPropertyList.Invoke();

                if (properties is not { Count: > 0 })
                    return;

                int length = properties.Count;

                // There's no recursion here, so it's ok to go with a big enough number for most cases
                // StackMaxSize = StackMaxLength * sizeof(godotsharp_property_info)
                const int StackMaxLength = 32;
                bool useStack = length < StackMaxLength;

                godotsharp_property_info* interopProperties;

                if (useStack)
                {
                    // Weird limitation, hence the need for aux:
                    // "In the case of pointer types, you can use a stackalloc expression only in a local variable declaration to initialize the variable."
                    var aux = stackalloc godotsharp_property_info[StackMaxLength];
                    interopProperties = aux;
                }
                else
                {
                    interopProperties = ((godotsharp_property_info*)NativeMemory.Alloc(
                        (nuint)length, (nuint)sizeof(godotsharp_property_info)))!;
                }

                try
                {
                    for (int i = 0; i < length; i++)
                    {
                        var property = properties[i];

                        godotsharp_property_info interopProperty = new()
                        {
                            Type = (int)property.Type,
                            Name = (godot_string_name)property.Name.NativeValue, // Not owned
                            Hint = (int)property.Hint,
                            HintString = Marshaling.ConvertStringToNative(property.HintString),
                            Usage = (int)property.Usage,
                            Exported = property.Exported.ToGodotBool()
                        };

                        interopProperties[i] = interopProperty;
                    }

                    using godot_string currentClassName = Marshaling.ConvertStringToNative(
                        ReflectionUtils.ConstructTypeName(scriptTypeMeta.Type));

                    addPropInfoFunc(scriptPtr, &currentClassName, interopProperties, length);

                    // We're borrowing the native value of the StringName entries.
                    // The dictionary needs to be kept alive until `addPropInfoFunc` returns.
                    GC.KeepAlive(properties);
                }
                finally
                {
                    for (int i = 0; i < length; i++)
                        interopProperties[i].Dispose();

                    if (!useStack)
                        NativeMemory.Free(interopProperties);
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

#pragma warning disable IDE1006 // Naming rule violation
        // ReSharper disable once InconsistentNaming
        // ReSharper disable once NotAccessedField.Local
        [StructLayout(LayoutKind.Sequential)]
        private ref struct godotsharp_property_def_val_pair
        {
            // Careful with padding...
            public godot_string_name Name; // Not owned
            public godot_variant Value; // Not owned
        }
#pragma warning restore IDE1006

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CallStaticWithTrampoline(MethodTrampolineDelegate methodTrampoline,
            godot_variant** args, int argCount, godot_variant_call_error* refCallError, godot_variant* outRet)
        {
            try
            {
                *outRet = methodTrampoline(null, new NativeVariantPtrArgs(args, argCount), ref *refCallError);
                return godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outRet = default;
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetPropertyDefaultValues(IntPtr scriptPtr,
            delegate* unmanaged<IntPtr, void*, int, void> addDefValFunc)
        {
            try
            {
                var scriptTypeMeta = GetOrResolveScriptTypeMeta(_scriptTypeBiMap.GetScriptType(scriptPtr));

                Dictionary<StringName, Variant>? defaultValues;

                if (scriptTypeMeta.GetGodotPropertyDefaultValues != null)
                {
                    defaultValues = scriptTypeMeta.GetGodotPropertyDefaultValues.Invoke();
                }
                else
                {
                    if (scriptTypeMeta.LegacyGetGodotPropertyDefaultValues == null)
                        return;

                    var defaultValuesLegacy = scriptTypeMeta.LegacyGetGodotPropertyDefaultValues.Invoke();

                    // We have to support this for some time, otherwise this could cause data loss for projects
                    // built with previous releases. Ideally, we should remove this before Godot 4.0 stable.

                    if (defaultValuesLegacy.Count <= 0)
                        return;

                    defaultValues = new();

                    foreach (var pair in defaultValuesLegacy)
                    {
                        defaultValues[pair.Key] = Variant.CreateConsuming(
                            DelegateUtils.RuntimeTypeConversionHelper.ConvertToVariant(pair.Value));
                    }
                }

                if (defaultValues is not { Count: > 0 })
                    return;

                int length = defaultValues.Count;

                // There's no recursion here, so it's ok to go with a big enough number for most cases
                // StackMaxSize = StackMaxLength * sizeof(godotsharp_property_def_val_pair)
                const int StackMaxLength = 32;
                bool useStack = length < StackMaxLength;

                godotsharp_property_def_val_pair* interopDefaultValues;

                if (useStack)
                {
                    // Weird limitation, hence the need for aux:
                    // "In the case of pointer types, you can use a stackalloc expression only in a local variable declaration to initialize the variable."
                    var aux = stackalloc godotsharp_property_def_val_pair[StackMaxLength];
                    interopDefaultValues = aux;
                }
                else
                {
                    interopDefaultValues = ((godotsharp_property_def_val_pair*)NativeMemory.Alloc(
                        (nuint)length, (nuint)sizeof(godotsharp_property_def_val_pair)))!;
                }

                try
                {
                    int i = 0;
                    foreach (var defaultValuePair in defaultValues)
                    {
                        godotsharp_property_def_val_pair interopProperty = new()
                        {
                            Name = (godot_string_name)defaultValuePair.Key.NativeValue, // Not owned
                            Value = (godot_variant)defaultValuePair.Value.NativeVar // Not owned
                        };

                        interopDefaultValues[i] = interopProperty;

                        i++;
                    }

                    addDefValFunc(scriptPtr, interopDefaultValues, length);

                    // We're borrowing the native value of the StringName and Variant entries.
                    // The dictionary needs to be kept alive until `addDefValFunc` returns.
                    GC.KeepAlive(defaultValues);
                }
                finally
                {
                    if (!useStack)
                        NativeMemory.Free(interopDefaultValues);
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool SwapGCHandleForType(IntPtr oldGCHandlePtr, IntPtr* outNewGCHandlePtr,
            godot_bool createWeak)
        {
            try
            {
                var oldGCHandle = GCHandle.FromIntPtr(oldGCHandlePtr);

                object? target = oldGCHandle.Target;

                if (target == null)
                {
                    CustomGCHandle.Free(oldGCHandle);
                    *outNewGCHandlePtr = IntPtr.Zero;
                    return godot_bool.False; // Called after the managed side was collected, so nothing to do here
                }

                // Release the current weak handle and replace it with a strong handle.
                var newGCHandle = createWeak.ToBool()
                    ? CustomGCHandle.AllocWeak(target)
                    : CustomGCHandle.AllocStrong(target);

                CustomGCHandle.Free(oldGCHandle);
                *outNewGCHandlePtr = GCHandle.ToIntPtr(newGCHandle);
                return godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outNewGCHandlePtr = IntPtr.Zero;
                return godot_bool.False;
            }
        }

        /// <summary>
        /// Provides access to otherwise inaccessible members of <see cref="GodotObject"/>
        /// for use in very specific scenarios, such as source generator constructor trampolines.
        /// </summary>
        [PublicAPI]
        public static class Accessors
        {
            /// <summary>
            /// WARNING: This is unsafe and should only be used in very specific scenarios,
            /// such as source generator constructor trampolines.
            /// </summary>
            /// <param name="godotObject">The Godot object to set the native pointer for.</param>
            /// <param name="nativePtr">The native pointer to set.</param>
            /// <exception cref="InvalidOperationException">
            /// Thrown when the Godot object is already initialized with a native pointer.
            /// </exception>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            [PublicAPI("Source generators depend on this for constructor trampolines.")]
            public static void UnsafeSetGodotObjectNativePtr(GodotObject godotObject, IntPtr nativePtr)
            {
                if (godotObject.NativePtr != IntPtr.Zero)
                    ThrowGodotObjectNativePtrAlreadySet();
                godotObject.NativePtr = nativePtr;

                // To avoid inlining the exception throwing code that could make the method less likely to be inlined.
                [MethodImpl(MethodImplOptions.NoInlining)]
                static void ThrowGodotObjectNativePtrAlreadySet()
                    => throw new InvalidOperationException(
                        "The Godot Object was already initialized with a native pointer.");
            }
        }
    }
}
