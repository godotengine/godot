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

namespace Godot.Bridge
{
    using PropertyTrampolines = (PropertyGetterTrampoline getterTramp, PropertySetterTrampoline setterTramp);
    using unsafe TryAddNameToProxyNameMapDelegate = delegate* unmanaged<
        IntPtr, godot_string_name*, int, godot_string_name*, void>;
    using unsafe TryAddMethodTrampolineDelegate = delegate* unmanaged<
        IntPtr, godot_string_name*, int, void*, godot_bool, void>;
    using unsafe TryAddPropertyTrampolineDelegate = delegate* unmanaged<
        IntPtr, godot_string_name*, void*, void*, void>;
    using unsafe TryAddRaiseSignalTrampolineDelegate = delegate* unmanaged<
        IntPtr, godot_string_name*, int, void*, void>;

    // TODO: Make class internal once we replace LookupScriptsInAssembly (the only public member) with source generators
    public static partial class ScriptManagerBridge
    {
        private static ConcurrentDictionary<AssemblyLoadContext, ConcurrentDictionary<Type, byte>>
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

        private static ScriptTypeBiMap _scriptTypeBiMap = new();
        private static PathScriptTypeBiMap _pathTypeBiMap = new();

        private static ConcurrentDictionary<IntPtr, (string? assemblyName, string classFullName)>
            _scriptDataForReload = new();

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
                using var stringName = StringName.CreateTakingOwnershipOfDisposableValue(
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
        internal static unsafe godot_bool CreateManagedForGodotObjectScriptInstance(IntPtr scriptPtr,
            IntPtr godotObject,
            godot_variant** args, int argCount)
        {
            // TODO: Optimize with source generators and delegate pointers.

            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                Type scriptType = _scriptTypeBiMap.GetScriptType(scriptPtr);

                Debug.Assert(!scriptType.IsAbstract,
                    $"Cannot create script instance. The class '{scriptType.FullName}' is abstract.");

                var ctor = scriptType
                    .GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                    .Where(c => c.GetParameters().Length == argCount)
                    .FirstOrDefault();

                if (ctor == null)
                {
                    if (argCount == 0)
                    {
                        throw new MissingMemberException(
                            $"Cannot create script instance. The class '{scriptType.FullName}' does not define a parameterless constructor.");
                    }
                    else
                    {
                        throw new MissingMemberException(
                            $"The class '{scriptType.FullName}' does not define a constructor that takes {argCount} parameters.");
                    }
                }

                var obj = (GodotObject)RuntimeHelpers.GetUninitializedObject(scriptType);

                var parameters = ctor.GetParameters();
                int paramCount = parameters.Length;

                var invokeParams = new object?[paramCount];

                for (int i = 0; i < paramCount; i++)
                {
                    invokeParams[i] = DelegateUtils.RuntimeTypeConversionHelper.ConvertToObjectOfType(
                        *args[i], parameters[i].ParameterType);
                }

                obj.NativePtr = godotObject;

                _ = ctor.Invoke(obj, invokeParams);


                return godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetScriptNativeName(IntPtr scriptPtr, godot_string_name* outRes)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                if (!_scriptTypeBiMap.TryGetScriptType(scriptPtr, out Type? scriptType))
                {
                    *outRes = default;
                    return;
                }

                var nativeName = GodotObject.InternalGetClassNativeBaseName(scriptType);

                if (nativeName == null)
                {
                    *outRes = default;
                    return;
                }

                *outRes = NativeFuncs.godotsharp_string_name_new_copy((godot_string_name)nativeName.NativeValue);
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outRes = default;
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

            if (!_pathTypeBiMap.TryGetScriptType(scriptPathStr, out Type? scriptType))
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

                Type native = GodotObject.InternalGetClassNativeBase(scriptType);
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

        // Called from GodotPlugins
        // ReSharper disable once UnusedMember.Local
        public static void LookupScriptsInAssembly(Assembly assembly)
        {
            static void LookupScriptForClass(Type type)
            {
                var scriptPathAttr = type.GetCustomAttributes(inherit: false)
                    .OfType<ScriptPathAttribute>()
                    .FirstOrDefault();

                if (scriptPathAttr == null)
                    return;

                _pathTypeBiMap.Add(scriptPathAttr.Path, type);

                if (AlcReloadCfg.IsAlcReloadingEnabled)
                {
                    AddTypeForAlcReloading(type);
                }
            }

            var assemblyHasScriptsAttr = assembly.GetCustomAttributes(inherit: false)
                .OfType<AssemblyHasScriptsAttribute>()
                .FirstOrDefault();

            if (assemblyHasScriptsAttr == null)
                return;

            if (assemblyHasScriptsAttr.RequiresLookup)
            {
                // This is supported for scenarios where specifying all types would be cumbersome,
                // such as when disabling C# source generators (for whatever reason) or when using a
                // language other than C# that has nothing similar to source generators to automate it.

                var typeOfGodotObject = typeof(GodotObject);

                foreach (var type in assembly.GetTypes())
                {
                    if (type.IsNested)
                        continue;

                    if (!typeOfGodotObject.IsAssignableFrom(type))
                        continue;

                    LookupScriptForClass(type);
                }
            }
            else
            {
                // This is the most likely scenario as we use C# source generators

                var scriptTypes = assemblyHasScriptsAttr.ScriptTypes;

                if (scriptTypes != null)
                {
                    foreach (var type in scriptTypes)
                    {
                        LookupScriptForClass(type);
                    }
                }
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void RaiseEventSignal(IntPtr ownerGCHandlePtr,
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

        [UnmanagedCallersOnly]
        internal static godot_bool TryReloadRegisteredScriptWithClass(IntPtr scriptPtr)
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

        private static unsafe void GetScriptTypeInfo(Type scriptType, godot_csharp_type_info* outTypeInfo)
        {
            godot_string className = Marshaling.ConvertStringToNative(ReflectionUtils.ConstructTypeName(scriptType));

            StringName? nativeBase = GodotObject.InternalGetClassNativeBaseName(scriptType);

            godot_string_name nativeBaseName = nativeBase != null
                ? NativeFuncs.godotsharp_string_name_new_copy((godot_string_name)nativeBase.NativeValue)
                : default;

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
            outTypeInfo->NativeBaseName = nativeBaseName;
            outTypeInfo->IconPath = iconPath;
            outTypeInfo->IsTool = isTool.ToGodotBool();
            outTypeInfo->IsGlobalClass = isGlobalClass.ToGodotBool();
            outTypeInfo->IsAbstract = scriptType.IsAbstract.ToGodotBool();
            outTypeInfo->IsGenericTypeDefinition = scriptType.IsGenericTypeDefinition.ToGodotBool();
            outTypeInfo->IsConstructedGenericType = scriptType.IsConstructedGenericType.ToGodotBool();
        }

        [ThreadStatic] private static TrampolineCollectorPool? _cachedTrampolineCollectorPool;

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptTrampolines(
            IntPtr scriptPtr, godot_bool* outShouldFallbackToLegacyTrampolines,
            TryAddNameToProxyNameMapDelegate tryAddNameToProxyNameMap,
            TryAddMethodTrampolineDelegate tryAddMethodTrampoline,
            TryAddPropertyTrampolineDelegate tryAddPropertyTrampoline,
            TryAddRaiseSignalTrampolineDelegate tryAddRaiseSignalTrampoline)
        {
            try
            {
                var scriptType = _scriptTypeBiMap.GetScriptType(scriptPtr);
                Debug.Assert(!scriptType.IsGenericTypeDefinition,
                    $"Script type must be a constructed generic type or not generic at all. Type: {scriptType}.");

                TrampolineCollectorPool collectorPool;

                if (_cachedTrampolineCollectorPool == null)
                {
                    _cachedTrampolineCollectorPool = new(
                        TwoArgumentArray: new object[2],
                        Collectors: new(
                            new(scriptPtr, tryAddNameToProxyNameMap),
                            new(scriptPtr, tryAddMethodTrampoline),
                            new(scriptPtr, tryAddPropertyTrampoline),
                            new(scriptPtr, tryAddRaiseSignalTrampoline)),
                        CollectionOptions: new(IncludeAncestors: true));

                    collectorPool = _cachedTrampolineCollectorPool.Value;
                }
                else
                {
                    collectorPool = _cachedTrampolineCollectorPool.Value;
                    collectorPool.Collectors.UpdateCollectors(scriptPtr, tryAddNameToProxyNameMap,
                        tryAddMethodTrampoline, tryAddPropertyTrampoline, tryAddRaiseSignalTrampoline);
                }

                GetGodotClassTrampolinesForType(scriptType, collectorPool);

                Type native = GodotObject.InternalGetClassNativeBase(scriptType);

                bool DoesUserScriptContainMethod(string methodName)
                {
                    var methodInfo = scriptType.GetMethod(methodName,
                        BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);

                    if (methodInfo == null)
                        return false;

                    for (Type? top = scriptType; top != null && top != native; top = top.BaseType)
                    {
                        if (methodInfo.DeclaringType == top)
                            return true;
                    }

                    return false;
                }

                // No need to check for "HasGodotClassMethod" nor "HasGodotClassSignal",
                // as these always accompany "InvokeGodotClassMethod" and "RaiseGodotClassSignalCallbacks".
                *outShouldFallbackToLegacyTrampolines =
                    (DoesUserScriptContainMethod("InvokeGodotClassMethod")
                     || DoesUserScriptContainMethod("SetGodotClassPropertyValue")
                     || DoesUserScriptContainMethod("GetGodotClassPropertyValue")
                     || DoesUserScriptContainMethod("RaiseGodotClassSignalCallbacks")).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptClassInfo(IntPtr scriptPtr, godot_csharp_type_info* outTypeInfo,
            godot_array* outMethodsDest, godot_dictionary* outRpcFunctionsDest, godot_dictionary* outEventSignalsDest,
            godot_ref* outBaseScript)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                var scriptType = _scriptTypeBiMap.GetScriptType(scriptPtr);
                Debug.Assert(!scriptType.IsGenericTypeDefinition,
                    $"Script type must be a constructed generic type or not generic at all. Type: {scriptType}.");

                GetScriptTypeInfo(scriptType, outTypeInfo);

                Type native = GodotObject.InternalGetClassNativeBase(scriptType);

                // Methods

                // Performance is not critical here as this will be replaced with source generators.
                using var methods = new Collections.Array();

                if (scriptType != native)
                {
                    var methodList = GetMethodListForType(scriptType);

                    if (methodList != null)
                    {
                        foreach (var method in methodList)
                        {
                            var methodInfo = new Collections.Dictionary();

                            methodInfo.Add("name", method.Name);

                            var returnVal = new Collections.Dictionary()
                            {
                                { "name", method.ReturnVal.Name },
                                { "type", (int)method.ReturnVal.Type },
                                { "usage", (int)method.ReturnVal.Usage }
                            };
                            if (method.ReturnVal.ClassName != null)
                            {
                                returnVal["class_name"] = method.ReturnVal.ClassName;
                            }

                            methodInfo.Add("return_val", returnVal);

                            var methodParams = new Collections.Array();

                            if (method.Arguments != null)
                            {
                                foreach (var param in method.Arguments)
                                {
                                    var pinfo = new Collections.Dictionary()
                                    {
                                        { "name", param.Name },
                                        { "type", (int)param.Type },
                                        { "usage", (int)param.Usage }
                                    };
                                    if (param.ClassName != null)
                                    {
                                        pinfo["class_name"] = param.ClassName;
                                    }

                                    methodParams.Add(pinfo);
                                }
                            }

                            methodInfo.Add("params", methodParams);

                            methodInfo.Add("flags", (int)method.Flags);

                            methods.Add(methodInfo);
                        }
                    }
                }

                *outMethodsDest = NativeFuncs.godotsharp_array_new_copy(
                    (godot_array)methods.NativeValue);

                // RPC functions

                Collections.Dictionary rpcFunctions = new();

                Type? top = scriptType;

                while (top != null && top != native)
                {
                    foreach (var method in top.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Instance |
                                                          BindingFlags.NonPublic | BindingFlags.Public))
                    {
                        if (method.IsStatic)
                            continue;

                        string methodName = method.Name;

                        if (rpcFunctions.ContainsKey(methodName))
                            continue;

                        var rpcAttr = method.GetCustomAttributes(inherit: false)
                            .OfType<RpcAttribute>().FirstOrDefault();

                        if (rpcAttr == null)
                            continue;

                        var rpcConfig = new Collections.Dictionary();

                        rpcConfig["rpc_mode"] = (long)rpcAttr.Mode;
                        rpcConfig["call_local"] = rpcAttr.CallLocal;
                        rpcConfig["transfer_mode"] = (long)rpcAttr.TransferMode;
                        rpcConfig["channel"] = rpcAttr.TransferChannel;

                        rpcFunctions.Add(methodName, rpcConfig);
                    }

                    top = top.BaseType;
                }

                *outRpcFunctionsDest = NativeFuncs.godotsharp_dictionary_new_copy(
                    (godot_dictionary)rpcFunctions.NativeValue);

                // Event signals

                // Performance is not critical here as this will be replaced with source generators.
                using var signals = new Collections.Dictionary();

                if (scriptType != native)
                {
                    var signalList = GetSignalListForType(scriptType);

                    if (signalList != null)
                    {
                        foreach (var signal in signalList)
                        {
                            string signalName = signal.Name;

                            if (signals.ContainsKey(signalName))
                                continue;

                            var signalParams = new Collections.Array();

                            if (signal.Arguments != null)
                            {
                                foreach (var param in signal.Arguments)
                                {
                                    var pinfo = new Collections.Dictionary()
                                    {
                                        { "name", param.Name },
                                        { "type", (int)param.Type },
                                        { "usage", (int)param.Usage }
                                    };
                                    if (param.ClassName != null)
                                    {
                                        pinfo["class_name"] = param.ClassName;
                                    }

                                    signalParams.Add(pinfo);
                                }
                            }

                            signals.Add(signalName, signalParams);
                        }
                    }
                }

                *outEventSignalsDest = NativeFuncs.godotsharp_dictionary_new_copy(
                    (godot_dictionary)signals.NativeValue);

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
                *outTypeInfo = default;
                *outMethodsDest = NativeFuncs.godotsharp_array_new();
                *outRpcFunctionsDest = NativeFuncs.godotsharp_dictionary_new();
                *outEventSignalsDest = NativeFuncs.godotsharp_dictionary_new();
                *outBaseScript = default;
            }
        }

        private static List<MethodInfo>? GetSignalListForType(Type type)
        {
            var getGodotSignalListMethod = type.GetMethod(
                "GetGodotSignalList",
                BindingFlags.DeclaredOnly | BindingFlags.Static |
                BindingFlags.NonPublic | BindingFlags.Public);

            if (getGodotSignalListMethod == null)
                return null;

            return (List<MethodInfo>?)getGodotSignalListMethod.Invoke(null, null);
        }

        private static List<MethodInfo>? GetMethodListForType(Type type)
        {
            var getGodotMethodListMethod = type.GetMethod(
                "GetGodotMethodList",
                BindingFlags.DeclaredOnly | BindingFlags.Static |
                BindingFlags.NonPublic | BindingFlags.Public);

            if (getGodotMethodListMethod == null)
                return null;

            return (List<MethodInfo>?)getGodotMethodListMethod.Invoke(null, null);
        }

        /// <summary>
        /// This is used to collect the method name to proxy name map for a script.
        /// </summary>
        /// <remarks>
        /// The user script can implement a static method "GetGodotClassTrampolines"
        /// that receives an instance of <see cref="TrampolineCollectors"/> which
        /// contains an instance of this class. The user script can then call the
        /// <see cref="TryAdd"/> method of this class to add proxy name mappings to the script.<br/>
        /// <br/>
        /// "GetGodotClassTrampolines" must be called on the most derived class first,
        /// and after collecting the members for that class, its implementation must call the base
        /// implementation to also collect members from the base classes in inheritance order.<br/>
        /// As a result, <see cref="TryAdd"/> will not add a method from a base class if the derived class
        /// has already added a method with the same MethodKey.
        /// </remarks>
        public class NameToProxyNameMapCollector
        {
            private IntPtr _scriptPtr;
            private unsafe TryAddNameToProxyNameMapDelegate _tryAddDelegate;

            internal unsafe NameToProxyNameMapCollector(IntPtr scriptPtr,
                TryAddNameToProxyNameMapDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            internal unsafe void Update(IntPtr scriptPtr, TryAddNameToProxyNameMapDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            /// <summary>
            /// Adds a method name to proxy name mapping to the script if a mapping for the given method key doesn't already exist.
            /// </summary>
            public unsafe void TryAdd(MethodKey methodKey, StringName proxyName)
            {
                var nameSelf = (godot_string_name)methodKey.Name.NativeValue;
                var proxyNameSelf = (godot_string_name)proxyName.NativeValue;
                _tryAddDelegate(_scriptPtr, &nameSelf, methodKey.ArgumentCount, &proxyNameSelf);
            }
        }

        /// <summary>
        /// This is used to collect the method trampolines for a script.
        /// </summary>
        /// <remarks>
        /// The user script can implement a static method "GetGodotClassTrampolines"
        /// that receives an instance of <see cref="TrampolineCollectors"/> which
        /// contains an instance of this class. The user script can then call the
        /// <see cref="TryAdd"/> method of this class to add methodsto the script.<br/>
        /// <br/>
        /// "GetGodotClassTrampolines" must be called on the most derived class first,
        /// and after collecting the members for that class, its implementation must call the base
        /// implementation to also collect members from the base classes in inheritance order.<br/>
        /// As a result, <see cref="TryAdd"/> will not add a method from a base class if the derived class
        /// has already added a method with the same MethodKey.
        /// </remarks>
        public class MethodTrampolineCollector
        {
            private IntPtr _scriptPtr;
            private unsafe TryAddMethodTrampolineDelegate _tryAddDelegate;

            internal unsafe MethodTrampolineCollector(IntPtr scriptPtr, TryAddMethodTrampolineDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            internal unsafe void Update(IntPtr scriptPtr, TryAddMethodTrampolineDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            /// <summary>
            /// Adds a method trampoline to the script if a trampoline for the given method key doesn't already exist.
            /// </summary>
            public unsafe void TryAdd(MethodKey methodKey, MethodTrampoline trampoline)
            {
                var nameSelf = (godot_string_name)methodKey.Name.NativeValue;
                _tryAddDelegate(_scriptPtr, &nameSelf, methodKey.ArgumentCount,
                    trampoline.TrampolineDelegate, trampoline.IsStatic.ToGodotBool());
            }
        }

        /// <summary>
        /// This is used to collect the property trampolines for a script.
        /// </summary>
        /// <remarks>
        /// The user script can implement a static method "GetGodotClassTrampolines"
        /// that receives an instance of <see cref="TrampolineCollectors"/> which
        /// contains an instance of this class. The user script can then call the
        /// <see cref="TryAdd"/> method of this class to add properties to the script.<br/>
        /// <br/>
        /// "GetGodotClassTrampolines" must be called on the most derived class first,
        /// and after collecting the members for that class, its implementation must call the base
        /// implementation to also collect members from the base classes in inheritance order.<br/>
        /// As a result, <see cref="TryAdd"/> will not add a property from a base class if the derived class
        /// has already added a property with the same name.
        /// </remarks>
        public class PropertyTrampolineCollector
        {
            private IntPtr _scriptPtr;
            private unsafe TryAddPropertyTrampolineDelegate _tryAddDelegate;

            internal unsafe PropertyTrampolineCollector(IntPtr scriptPtr,
                TryAddPropertyTrampolineDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            internal unsafe void Update(IntPtr scriptPtr, TryAddPropertyTrampolineDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            /// <summary>
            /// Adds property trampolines to the script if trampolines for the given property name don't already exist.
            /// If trampolines for the property already exist, but one of the getter or setter trampolines is missing,
            /// the missing trampoline will be added if provided. This allows a derived class to introduce a readonly
            /// or writeonly property with the same name as a property in the base class, overriding that specific
            /// accessor while inheriting the other one. This is done only to match the behavior of the old
            /// trampoline system (SetGodotClassPropertyTrampoline and GetGodotClassPropertyTrampoline).
            /// </summary>
            public unsafe void TryAdd(StringName propertyName, PropertyTrampolines trampolines)
            {
                var propertyNameSelf = (godot_string_name)propertyName.NativeValue;
                _tryAddDelegate(_scriptPtr, &propertyNameSelf,
                    trampolines.getterTramp.TrampolineDelegate,
                    trampolines.setterTramp.TrampolineDelegate);
            }
        }

        /// <summary>
        /// This is used to collect the raise signal trampolines for a script.
        /// </summary>
        /// <remarks>
        /// The user script can implement a static method "GetGodotClassTrampolines"
        /// that receives an instance of <see cref="TrampolineCollectors"/> which
        /// contains an instance of this class. The user script can then call the
        /// <see cref="TryAdd"/> method of this class to add signals to the script.<br/>
        /// <br/>
        /// "GetGodotClassTrampolines" must be called on the most derived class first,
        /// and after collecting the members for that class, its implementation must call the base
        /// implementation to also collect members from the base classes in inheritance order.<br/>
        /// As a result, <see cref="TryAdd"/> will not add a signal from a base class if the derived class
        /// has already added a signal with the same SignalKey.
        /// </remarks>
        public class RaiseSignalTrampolineCollector
        {
            private IntPtr _scriptPtr;
            private unsafe TryAddRaiseSignalTrampolineDelegate _tryAddDelegate;

            internal unsafe RaiseSignalTrampolineCollector(IntPtr scriptPtr,
                TryAddRaiseSignalTrampolineDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            internal unsafe void Update(IntPtr scriptPtr, TryAddRaiseSignalTrampolineDelegate tryAddDelegate)
            {
                _scriptPtr = scriptPtr;
                _tryAddDelegate = tryAddDelegate;
            }

            /// <summary>
            /// Adds a raise signal trampoline to the script if a trampoline for the given signal key doesn't already exist.
            /// </summary>
            public unsafe void TryAdd(SignalKey signalKey, RaiseSignalTrampoline trampoline)
            {
                var nameSelf = (godot_string_name)signalKey.Name.NativeValue;
                _tryAddDelegate(_scriptPtr, &nameSelf, signalKey.ArgumentCount, trampoline.TrampolineDelegate);
            }
        }

        /// <summary>
        /// Group of collectors passed to the user script to collect trampolines
        /// and method name to proxy name mappings for a script.
        /// </summary>
        public record TrampolineCollectors(
            NameToProxyNameMapCollector NameToProxyNameMapCollector,
            MethodTrampolineCollector MethodTrampolineCollector,
            PropertyTrampolineCollector PropertyTrampolineCollector,
            RaiseSignalTrampolineCollector RaiseSignalTrampolineCollector)
        {
            internal unsafe void UpdateCollectors(IntPtr scriptPtr,
                TryAddNameToProxyNameMapDelegate tryAddNameToProxyNameMap,
                TryAddMethodTrampolineDelegate tryAddMethodTrampoline,
                TryAddPropertyTrampolineDelegate tryAddPropertyTrampoline,
                TryAddRaiseSignalTrampolineDelegate tryAddRaiseSignalTrampoline)
            {
                NameToProxyNameMapCollector.Update(scriptPtr, tryAddNameToProxyNameMap);
                MethodTrampolineCollector.Update(scriptPtr, tryAddMethodTrampoline);
                PropertyTrampolineCollector.Update(scriptPtr, tryAddPropertyTrampoline);
                RaiseSignalTrampolineCollector.Update(scriptPtr, tryAddRaiseSignalTrampoline);
            }
        }

        /// <summary>
        /// Options for collecting trampolines for a script.
        /// </summary>
        /// <param name="IncludeAncestors">
        /// Whether trampolines from ancestor classes should also be collected.
        /// If true, the trampoline collection method of each ancestor class must be called
        /// after the trampoline collection method of the current class.
        /// </param>
        public record TrampolineCollectionOptions(bool IncludeAncestors);

        /// <summary>
        /// This is used as a pool to avoid having to allocate multiple instances of the
        /// collectors and argument arrays when updating trampolines for a script.
        /// </summary>
        private record struct TrampolineCollectorPool(
            object[] TwoArgumentArray,
            TrampolineCollectors Collectors,
            TrampolineCollectionOptions CollectionOptions);

        private static void GetGodotClassTrampolinesForType(Type type, TrampolineCollectorPool collectorPool)
        {
            var getGodotMethodNameToProxyNameMap = type.GetMethod(
                "GetGodotClassTrampolines",
                BindingFlags.DeclaredOnly | BindingFlags.Static |
                BindingFlags.NonPublic | BindingFlags.Public);

            if (getGodotMethodNameToProxyNameMap == null)
                return;

            collectorPool.TwoArgumentArray[0] = collectorPool.Collectors;
            collectorPool.TwoArgumentArray[1] = collectorPool.CollectionOptions;
            getGodotMethodNameToProxyNameMap.Invoke(null, collectorPool.TwoArgumentArray);
        }

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
                Type scriptType = _scriptTypeBiMap.GetScriptType(scriptPtr);
                GetPropertyInfoListForType(scriptType, scriptPtr, addPropInfoFunc);
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        private static unsafe void GetPropertyInfoListForType(Type type, IntPtr scriptPtr,
            delegate* unmanaged<IntPtr, godot_string*, void*, int, void> addPropInfoFunc)
        {
            try
            {
                var getGodotPropertyListMethod = type.GetMethod(
                    "GetGodotPropertyList",
                    BindingFlags.DeclaredOnly | BindingFlags.Static |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (getGodotPropertyListMethod == null)
                    return;

                var properties = (List<PropertyInfo>?)
                    getGodotPropertyListMethod.Invoke(null, null);

                if (properties == null || properties.Count <= 0)
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

                    using godot_string currentClassName =
                        Marshaling.ConvertStringToNative(ReflectionUtils.ConstructTypeName(type));

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

        private delegate bool InvokeGodotClassStaticMethodDelegate(in godot_string_name method,
            NativeVariantPtrArgs args, out godot_variant ret);

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CallStatic(IntPtr scriptPtr, godot_string_name* method,
            godot_variant** args, int argCount, godot_variant_call_error* refCallError, godot_variant* outRet)
        {
            try
            {
                Type scriptType = _scriptTypeBiMap.GetScriptType(scriptPtr);

                Type? top = scriptType;
                Type native = GodotObject.InternalGetClassNativeBase(top);

                while (top != null && top != native)
                {
                    var invokeGodotClassStaticMethod = top.GetMethod(
                        "InvokeGodotClassStaticMethod",
                        BindingFlags.DeclaredOnly | BindingFlags.Static |
                        BindingFlags.NonPublic | BindingFlags.Public);

                    if (invokeGodotClassStaticMethod != null)
                    {
                        var invoked =
                            invokeGodotClassStaticMethod.CreateDelegate<InvokeGodotClassStaticMethodDelegate>()(
                                CustomUnsafe.AsRef(method), new NativeVariantPtrArgs(args, argCount),
                                out godot_variant retValue);
                        if (invoked)
                        {
                            *outRet = retValue;
                            return godot_bool.True;
                        }
                    }

                    top = top.BaseType;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outRet = default;
                return godot_bool.False;
            }

            *outRet = default;
            (*refCallError).Error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD;
            return godot_bool.False;
        }

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
                Type? top = _scriptTypeBiMap.GetScriptType(scriptPtr);
                Type native = GodotObject.InternalGetClassNativeBase(top);

                while (top != null && top != native)
                {
                    GetPropertyDefaultValuesForType(top, scriptPtr, addDefValFunc);

                    top = top.BaseType;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [SkipLocalsInit]
        private static unsafe void GetPropertyDefaultValuesForType(Type type, IntPtr scriptPtr,
            delegate* unmanaged<IntPtr, void*, int, void> addDefValFunc)
        {
            try
            {
                var getGodotPropertyDefaultValuesMethod = type.GetMethod(
                    "GetGodotPropertyDefaultValues",
                    BindingFlags.DeclaredOnly | BindingFlags.Static |
                    BindingFlags.NonPublic | BindingFlags.Public);

                if (getGodotPropertyDefaultValuesMethod == null)
                    return;

                var defaultValuesObj = getGodotPropertyDefaultValuesMethod.Invoke(null, null);

                if (defaultValuesObj == null)
                    return;

                Dictionary<StringName, Variant> defaultValues;

                if (defaultValuesObj is Dictionary<StringName, object> defaultValuesLegacy)
                {
                    // We have to support this for some time, otherwise this could cause data loss for projects
                    // built with previous releases. Ideally, we should remove this before Godot 4.0 stable.

                    if (defaultValuesLegacy.Count <= 0)
                        return;

                    defaultValues = new();

                    foreach (var pair in defaultValuesLegacy)
                    {
                        defaultValues[pair.Key] = Variant.CreateTakingOwnershipOfDisposableValue(
                            DelegateUtils.RuntimeTypeConversionHelper.ConvertToVariant(pair.Value));
                    }
                }
                else
                {
                    defaultValues = (Dictionary<StringName, Variant>)defaultValuesObj;
                }

                if (defaultValues.Count <= 0)
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
    }
}
