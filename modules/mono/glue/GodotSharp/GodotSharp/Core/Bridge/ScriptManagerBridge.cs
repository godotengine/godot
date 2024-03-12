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
using System.Runtime.Serialization;
using Godot.NativeInterop;

namespace Godot.Bridge
{
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
                        !_pathTypeBiMap.TryGetScriptPath(type, out _))
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
            // TODO: Optimize with source generators and delegate pointers.

            try
            {
                using var stringName = StringName.CreateTakingOwnershipOfDisposableValue(
                    NativeFuncs.godotsharp_string_name_new_copy(CustomUnsafe.AsRef(nativeTypeName)));
                string nativeTypeNameStr = stringName.ToString();

                Type nativeType = TypeGetProxyClass(nativeTypeNameStr) ?? throw new InvalidOperationException(
                    "Wrapper class not found for type: " + nativeTypeNameStr);
                var obj = (GodotObject)FormatterServices.GetUninitializedObject(nativeType);

                var ctor = nativeType.GetConstructor(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);

                obj.NativePtr = godotObject;

                _ = ctor!.Invoke(obj, null);

                return GCHandle.ToIntPtr(CustomGCHandle.AllocStrong(obj));
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

                Debug.Assert(!scriptType.IsAbstract, $"Cannot create script instance. The class '{scriptType.FullName}' is abstract.");

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

                var obj = (GodotObject)FormatterServices.GetUninitializedObject(scriptType);

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

                var native = GodotObject.InternalGetClassNativeBase(scriptType);

                var field = native?.GetField("NativeName", BindingFlags.DeclaredOnly | BindingFlags.Static |
                                                           BindingFlags.Public | BindingFlags.NonPublic);

                if (field == null)
                {
                    *outRes = default;
                    return;
                }

                var nativeName = (StringName?)field.GetValue(null);

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

        private static Type? TypeGetProxyClass(string nativeTypeNameStr)
        {
            // Performance is not critical here as this will be replaced with a generated dictionary.

            if (nativeTypeNameStr[0] == '_')
                nativeTypeNameStr = nativeTypeNameStr.Substring(1);

            Type? wrapperType = typeof(GodotObject).Assembly.GetType("Godot." + nativeTypeNameStr);

            if (wrapperType == null)
            {
                wrapperType = GetTypeByGodotClassAttr(typeof(GodotObject).Assembly, nativeTypeNameStr);
            }

            if (wrapperType == null)
            {
                var editorAssembly = AppDomain.CurrentDomain.GetAssemblies()
                    .FirstOrDefault(a => a.GetName().Name == "GodotSharpEditor");
                wrapperType = editorAssembly?.GetType("Godot." + nativeTypeNameStr);

                if (wrapperType == null)
                {
                    wrapperType = GetTypeByGodotClassAttr(editorAssembly, nativeTypeNameStr);
                }
            }

            static Type? GetTypeByGodotClassAttr(Assembly assembly, string nativeTypeNameStr)
            {
                var types = assembly.GetTypes();
                foreach (var type in types)
                {
                    var attr = type.GetCustomAttribute<GodotClassNameAttribute>();
                    if (attr?.Name == nativeTypeNameStr)
                    {
                        return type;
                    }
                }
                return null;
            }

            static bool IsStatic(Type type) => type.IsAbstract && type.IsSealed;

            if (wrapperType != null && IsStatic(wrapperType))
            {
                // A static class means this is a Godot singleton class. Try to get the Instance proxy type.
                wrapperType = TypeGetProxyClass($"{wrapperType.Name}Instance");
                if (wrapperType == null)
                {
                    // Otherwise, fallback to GodotObject.
                    return typeof(GodotObject);
                }
            }

            return wrapperType;
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
                    if (type.IsNested || type.IsGenericType)
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
                        if (type.IsGenericType)
                            continue;

                        LookupScriptForClass(type);
                    }
                }
            }

            // This method may be called before initialization.
            if (NativeFuncs.godotsharp_dotnet_module_is_initialized().ToBool() && Engine.IsEditorHint())
            {
                foreach (var scriptPath in _pathTypeBiMap.Paths)
                {
                    using godot_string nativeScriptPath = Marshaling.ConvertStringToNative(scriptPath);
                    NativeFuncs.godotsharp_internal_editor_file_system_update_file(nativeScriptPath);
                }
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void RaiseEventSignal(IntPtr ownerGCHandlePtr,
            godot_string_name* eventSignalName, godot_variant** args, int argCount, godot_bool* outOwnerIsNull)
        {
            try
            {
                var owner = (GodotObject?)GCHandle.FromIntPtr(ownerGCHandlePtr).Target;

                if (owner == null)
                {
                    *outOwnerIsNull = godot_bool.True;
                    return;
                }

                *outOwnerIsNull = godot_bool.False;

                owner.RaiseGodotClassSignalCallbacks(CustomUnsafe.AsRef(eventSignalName),
                    new NativeVariantPtrArgs(args, argCount));
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outOwnerIsNull = godot_bool.False;
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
                lock (_scriptTypeBiMap.ReadWriteLock)
                {
                    if (!_scriptTypeBiMap.IsScriptRegistered(scriptPtr))
                    {
                        string scriptPathStr = Marshaling.ConvertStringToManaged(*scriptPath);

                        if (!_pathTypeBiMap.TryGetScriptType(scriptPathStr, out Type? scriptType))
                            return godot_bool.False;

                        _scriptTypeBiMap.Add(scriptPtr, scriptType);
                    }
                }

                return godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
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

            GetOrCreateScriptBridgeForType(scriptType, outScript);
        }

        private static unsafe void GetOrCreateScriptBridgeForType(Type scriptType, godot_ref* outScript)
        {
            lock (_scriptTypeBiMap.ReadWriteLock)
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
        }

        internal static unsafe void GetOrLoadOrCreateScriptForType(Type scriptType, godot_ref* outScript)
        {
            static bool GetPathOtherwiseGetOrCreateScript(Type scriptType, godot_ref* outScript,
                [MaybeNullWhen(false)] out string scriptPath)
            {
                lock (_scriptTypeBiMap.ReadWriteLock)
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

                    CreateScriptBridgeForType(scriptType, outScript);
                    scriptPath = null;
                    return false;
                }
            }

            if (GetPathOtherwiseGetOrCreateScript(scriptType, outScript, out string? scriptPath))
            {
                // This path is slower, but it's only executed for the first instantiation of the type

                // This must be done outside the read-write lock, as the script resource loading can lock it
                using godot_string scriptPathIn = Marshaling.ConvertStringToNative(scriptPath);
                if (!NativeFuncs.godotsharp_internal_script_load(scriptPathIn, outScript).ToBool())
                {
                    GD.PushError($"Cannot load script for type '{scriptType.FullName}'. Path: '{scriptPath}'.");

                    // If loading of the script fails, best we can do create a new script
                    // with no path, as we do for types without an associated script file.
                    GetOrCreateScriptBridgeForType(scriptType, outScript);
                }
            }
        }

        private static unsafe void CreateScriptBridgeForType(Type scriptType, godot_ref* outScript)
        {
            NativeFuncs.godotsharp_internal_new_csharp_script(outScript);
            IntPtr scriptPtr = outScript->Reference;

            // Caller takes care of locking
            _scriptTypeBiMap.Add(scriptPtr, scriptType);

            NativeFuncs.godotsharp_internal_reload_registered_script(scriptPtr);
        }

        [UnmanagedCallersOnly]
        internal static void RemoveScriptBridge(IntPtr scriptPtr)
        {
            try
            {
                lock (_scriptTypeBiMap.ReadWriteLock)
                {
                    _scriptTypeBiMap.Remove(scriptPtr);
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static godot_bool TryReloadRegisteredScriptWithClass(IntPtr scriptPtr)
        {
            try
            {
                lock (_scriptTypeBiMap.ReadWriteLock)
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

                    // ReSharper disable once RedundantNameQualifier
                    if (!typeof(GodotObject).IsAssignableFrom(scriptType))
                    {
                        // The class no longer inherits GodotObject, can't reload
                        return godot_bool.False;
                    }

                    _scriptTypeBiMap.Add(scriptPtr, scriptType);

                    NativeFuncs.godotsharp_internal_reload_registered_script(scriptPtr);

                    return godot_bool.True;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptClassInfo(IntPtr scriptPtr, godot_string* outClassName,
            godot_bool* outTool, godot_bool* outGlobal, godot_bool* outAbstract, godot_string* outIconPath,
            godot_array* outMethodsDest, godot_dictionary* outRpcFunctionsDest,
            godot_dictionary* outEventSignalsDest, godot_ref* outBaseScript)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                var scriptType = _scriptTypeBiMap.GetScriptType(scriptPtr);

                *outClassName = Marshaling.ConvertStringToNative(scriptType.Name);

                *outTool = scriptType.GetCustomAttributes(inherit: false)
                    .OfType<ToolAttribute>()
                    .Any().ToGodotBool();

                if (!(*outTool).ToBool() && scriptType.IsNested)
                {
                    *outTool = (scriptType.DeclaringType?.GetCustomAttributes(inherit: false)
                        .OfType<ToolAttribute>()
                        .Any() ?? false).ToGodotBool();
                }

                if (!(*outTool).ToBool() && scriptType.Assembly.GetName().Name == "GodotTools")
                    *outTool = godot_bool.True;

                var globalAttr = scriptType.GetCustomAttributes(inherit: false)
                    .OfType<GlobalClassAttribute>()
                    .FirstOrDefault();

                *outGlobal = (globalAttr != null).ToGodotBool();

                var iconAttr = scriptType.GetCustomAttributes(inherit: false)
                    .OfType<IconAttribute>()
                    .FirstOrDefault();
                *outIconPath = Marshaling.ConvertStringToNative(iconAttr?.Path);

                *outAbstract = scriptType.IsAbstract.ToGodotBool();

                // Methods

                // Performance is not critical here as this will be replaced with source generators.
                using var methods = new Collections.Array();

                Type? top = scriptType;
                Type native = GodotObject.InternalGetClassNativeBase(top);

                while (top != null && top != native)
                {
                    var methodList = GetMethodListForType(top);

                    if (methodList != null)
                    {
                        foreach (var method in methodList)
                        {
                            var methodInfo = new Collections.Dictionary();

                            methodInfo.Add("name", method.Name);

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

                    top = top.BaseType;
                }

                *outMethodsDest = NativeFuncs.godotsharp_array_new_copy(
                    (godot_array)methods.NativeValue);

                // RPC functions

                Collections.Dictionary rpcFunctions = new();

                top = scriptType;

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

                top = scriptType;

                while (top != null && top != native)
                {
                    var signalList = GetSignalListForType(top);

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

                    top = top.BaseType;
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
                *outClassName = default;
                *outTool = godot_bool.False;
                *outGlobal = godot_bool.False;
                *outAbstract = godot_bool.False;
                *outIconPath = default;
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

        // ReSharper disable once InconsistentNaming
        [SuppressMessage("ReSharper", "NotAccessedField.Local")]
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
                // stackMaxSize = stackMaxLength * sizeof(godotsharp_property_info)
                const int stackMaxLength = 32;
                bool useStack = length < stackMaxLength;

                godotsharp_property_info* interopProperties;

                if (useStack)
                {
                    // Weird limitation, hence the need for aux:
                    // "In the case of pointer types, you can use a stackalloc expression only in a local variable declaration to initialize the variable."
                    var aux = stackalloc godotsharp_property_info[stackMaxLength];
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

                    using godot_string currentClassName = Marshaling.ConvertStringToNative(type.Name);

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

        // ReSharper disable once InconsistentNaming
        [SuppressMessage("ReSharper", "NotAccessedField.Local")]
        [StructLayout(LayoutKind.Sequential)]
        private ref struct godotsharp_property_def_val_pair
        {
            // Careful with padding...
            public godot_string_name Name; // Not owned
            public godot_variant Value; // Not owned
        }

        private delegate bool InvokeGodotClassStaticMethodDelegate(in godot_string_name method, NativeVariantPtrArgs args, out godot_variant ret);

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CallStatic(IntPtr scriptPtr, godot_string_name* method,
            godot_variant** args, int argCount, godot_variant_call_error* refCallError, godot_variant* ret)
        {
            // TODO: Optimize with source generators and delegate pointers.

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
                        var invoked = invokeGodotClassStaticMethod.CreateDelegate<InvokeGodotClassStaticMethodDelegate>()(
                            CustomUnsafe.AsRef(method), new NativeVariantPtrArgs(args, argCount), out godot_variant retValue);
                        if (invoked)
                        {
                            *ret = retValue;
                            return godot_bool.True;
                        }
                    }

                    top = top.BaseType;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *ret = default;
                return godot_bool.False;
            }

            *ret = default;
            (*refCallError).Error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD;
            return godot_bool.False;
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
                // stackMaxSize = stackMaxLength * sizeof(godotsharp_property_def_val_pair)
                const int stackMaxLength = 32;
                bool useStack = length < stackMaxLength;

                godotsharp_property_def_val_pair* interopDefaultValues;

                if (useStack)
                {
                    // Weird limitation, hence the need for aux:
                    // "In the case of pointer types, you can use a stackalloc expression only in a local variable declaration to initialize the variable."
                    var aux = stackalloc godotsharp_property_def_val_pair[stackMaxLength];
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
                var newGCHandle = createWeak.ToBool() ?
                    CustomGCHandle.AllocWeak(target) :
                    CustomGCHandle.AllocStrong(target);

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
