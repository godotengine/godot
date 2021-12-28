using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Runtime.Serialization;
using Godot.Collections;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    internal static class ScriptManagerBridge
    {
        private static System.Collections.Generic.Dictionary<string, ScriptLookupInfo> _scriptLookupMap = new();
        private static System.Collections.Generic.Dictionary<IntPtr, Type> _scriptBridgeMap = new();

        private struct ScriptLookupInfo
        {
            public string ClassNamespace { get; private set; }
            public string ClassName { get; private set; }
            public Type ScriptType { get; private set; }

            public ScriptLookupInfo(string classNamespace, string className, Type scriptType)
            {
                ClassNamespace = classNamespace;
                ClassName = className;
                ScriptType = scriptType;
            }
        };

        [UnmanagedCallersOnly]
        internal static void FrameCallback()
        {
            try
            {
                Dispatcher.DefaultGodotTaskScheduler?.Activate();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe IntPtr CreateManagedForGodotObjectBinding(godot_string_name* nativeTypeName,
            IntPtr godotObject)
        {
            // TODO: Optimize with source generators and delegate pointers

            try
            {
                Type nativeType = TypeGetProxyClass(nativeTypeName);
                var obj = (Object)FormatterServices.GetUninitializedObject(nativeType);

                Object.HandlePendingForNextInstance = godotObject;

                var ctor = nativeType.GetConstructor(
                    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                    null, Type.EmptyTypes, null);
                _ = ctor!.Invoke(obj, null);

                return GCHandle.ToIntPtr(GCHandle.Alloc(obj));
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugPrintUnhandledException(e);
                return IntPtr.Zero;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CreateManagedForGodotObjectScriptInstance(IntPtr scriptPtr,
            IntPtr godotObject,
            godot_variant** args, int argCount)
        {
            // TODO: Optimize with source generators and delegate pointers

            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                Type scriptType = _scriptBridgeMap[scriptPtr];
                var obj = (Object)FormatterServices.GetUninitializedObject(scriptType);

                Object.HandlePendingForNextInstance = godotObject;

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
                            $"The class '{scriptType.FullName}' does not define a constructor that takes x parameters.");
                    }
                }

                var parameters = ctor.GetParameters();
                int paramCount = parameters.Length;

                object[] invokeParams = new object[paramCount];

                for (int i = 0; i < paramCount; i++)
                {
                    invokeParams[i] = Marshaling.ConvertVariantToManagedObjectOfType(
                        *args[i], parameters[i].ParameterType);
                }

                ctor.Invoke(obj, invokeParams);
                return true.ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugPrintUnhandledException(e);
                return false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetScriptNativeName(IntPtr scriptPtr, godot_string_name* outRes)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                if (!_scriptBridgeMap.TryGetValue(scriptPtr, out var scriptType))
                {
                    *outRes = default;
                    return;
                }

                var native = Object.InternalGetClassNativeBase(scriptType);

                var field = native?.GetField("NativeName", BindingFlags.DeclaredOnly | BindingFlags.Static |
                                                           BindingFlags.Public | BindingFlags.NonPublic);

                if (field == null)
                {
                    *outRes = default;
                    return;
                }

                var nativeName = (StringName)field.GetValue(null);

                if (nativeName == null)
                {
                    *outRes = default;
                    return;
                }

                *outRes = NativeFuncs.godotsharp_string_name_new_copy((godot_string_name)nativeName.NativeValue);
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                *outRes = default;
            }
        }

        [UnmanagedCallersOnly]
        internal static void SetGodotObjectPtr(IntPtr gcHandlePtr, IntPtr newPtr)
        {
            try
            {
                var target = (Object)GCHandle.FromIntPtr(gcHandlePtr).Target;
                if (target != null)
                    target.NativePtr = newPtr;
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
            }
        }

        private static unsafe Type TypeGetProxyClass(godot_string_name* nativeTypeName)
        {
            // Performance is not critical here as this will be replaced with a generated dictionary.
            using var stringName = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(CustomUnsafe.AsRef(nativeTypeName)));
            string nativeTypeNameStr = stringName.ToString();

            if (nativeTypeNameStr[0] == '_')
                nativeTypeNameStr = nativeTypeNameStr.Substring(1);

            Type wrapperType = typeof(Object).Assembly.GetType("Godot." + nativeTypeNameStr);

            if (wrapperType == null)
            {
                wrapperType = AppDomain.CurrentDomain.GetAssemblies()
                    .First(a => a.GetName().Name == "GodotSharpEditor")
                    .GetType("Godot." + nativeTypeNameStr);
            }

            static bool IsStatic(Type type) => type.IsAbstract && type.IsSealed;

            if (wrapperType != null && IsStatic(wrapperType))
            {
                // A static class means this is a Godot singleton class. If an instance is needed we use Godot.Object.
                return typeof(Object);
            }

            return wrapperType;
        }

        // Called from GodotPlugins
        // ReSharper disable once UnusedMember.Local
        private static void LookupScriptsInAssembly(Assembly assembly)
        {
            static void LookupScriptForClass(Type type)
            {
                var scriptPathAttr = type.GetCustomAttributes(inherit: false)
                    .OfType<ScriptPathAttribute>()
                    .FirstOrDefault();

                if (scriptPathAttr == null)
                    return;

                _scriptLookupMap[scriptPathAttr.Path] = new ScriptLookupInfo(type.Namespace, type.Name, type);
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

                var typeOfGodotObject = typeof(Object);

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
                    for (int i = 0; i < scriptTypes.Length; i++)
                    {
                        LookupScriptForClass(scriptTypes[i]);
                    }
                }
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void RaiseEventSignal(IntPtr ownerGCHandlePtr,
            godot_string_name* eventSignalName, godot_variant** args, int argCount, godot_bool* outOwnerIsNull)
        {
            try
            {
                var owner = (Object)GCHandle.FromIntPtr(ownerGCHandlePtr).Target;

                if (owner == null)
                {
                    *outOwnerIsNull = true.ToGodotBool();
                    return;
                }

                *outOwnerIsNull = false.ToGodotBool();

                owner.InternalRaiseEventSignal(CustomUnsafe.AsRef(eventSignalName),
                    new NativeVariantPtrArgs(args), argCount);
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugPrintUnhandledException(e);
                *outOwnerIsNull = false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetScriptSignalList(IntPtr scriptPtr, godot_dictionary* outRetSignals)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                using var signals = new Dictionary();

                Type top = _scriptBridgeMap[scriptPtr];
                Type native = Object.InternalGetClassNativeBase(top);

                while (top != null && top != native)
                {
                    // Legacy signals

                    foreach (var signalDelegate in top
                                 .GetNestedTypes(BindingFlags.DeclaredOnly | BindingFlags.NonPublic |
                                                 BindingFlags.Public)
                                 .Where(nestedType => typeof(Delegate).IsAssignableFrom(nestedType))
                                 .Where(@delegate => @delegate.GetCustomAttributes().OfType<SignalAttribute>().Any()))
                    {
                        var invokeMethod = signalDelegate.GetMethod("Invoke");

                        if (invokeMethod == null)
                            throw new MissingMethodException(signalDelegate.FullName, "Invoke");

                        var signalParams = new Collections.Array();

                        foreach (var parameters in invokeMethod.GetParameters())
                        {
                            var paramType = Marshaling.ConvertManagedTypeToVariantType(
                                parameters.ParameterType, out bool nilIsVariant);
                            signalParams.Add(new Dictionary()
                            {
                                { "name", parameters.Name },
                                { "type", paramType },
                                { "nil_is_variant", nilIsVariant }
                            });
                        }

                        signals.Add(signalDelegate.Name, signalParams);
                    }

                    // Event signals

                    var foundEventSignals = top.GetEvents(
                            BindingFlags.DeclaredOnly | BindingFlags.Instance |
                            BindingFlags.NonPublic | BindingFlags.Public)
                        .Where(ev => ev.GetCustomAttributes().OfType<SignalAttribute>().Any())
                        .Select(ev => ev.Name);

                    var fields = top.GetFields(
                        BindingFlags.DeclaredOnly | BindingFlags.Instance |
                        BindingFlags.NonPublic | BindingFlags.Public);

                    foreach (var eventSignalField in fields
                                 .Where(f => typeof(Delegate).IsAssignableFrom(f.FieldType))
                                 .Where(f => foundEventSignals.Contains(f.Name)))
                    {
                        var delegateType = eventSignalField.FieldType;
                        var invokeMethod = delegateType.GetMethod("Invoke");

                        if (invokeMethod == null)
                            throw new MissingMethodException(delegateType.FullName, "Invoke");

                        var signalParams = new Collections.Array();

                        foreach (var parameters in invokeMethod.GetParameters())
                        {
                            var paramType = Marshaling.ConvertManagedTypeToVariantType(
                                parameters.ParameterType, out bool nilIsVariant);
                            signalParams.Add(new Dictionary()
                            {
                                { "name", parameters.Name },
                                { "type", paramType },
                                { "nil_is_variant", nilIsVariant }
                            });
                        }

                        signals.Add(eventSignalField.Name, signalParams);
                    }

                    top = top.BaseType;
                }

                *outRetSignals = NativeFuncs.godotsharp_dictionary_new_copy((godot_dictionary)signals.NativeValue);
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                *outRetSignals = NativeFuncs.godotsharp_dictionary_new();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool HasScriptSignal(IntPtr scriptPtr, godot_string* signalName)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                using var signals = new Dictionary();

                string signalNameStr = Marshaling.ConvertStringToManaged(*signalName);

                Type top = _scriptBridgeMap[scriptPtr];
                Type native = Object.InternalGetClassNativeBase(top);

                while (top != null && top != native)
                {
                    // Legacy signals

                    if (top
                        .GetNestedTypes(BindingFlags.DeclaredOnly | BindingFlags.NonPublic | BindingFlags.Public)
                        .Where(nestedType => typeof(Delegate).IsAssignableFrom(nestedType))
                        .Where(@delegate => @delegate.GetCustomAttributes().OfType<SignalAttribute>().Any())
                        .Any(signalDelegate => signalDelegate.Name == signalNameStr)
                       )
                    {
                        return true.ToGodotBool();
                    }

                    // Event signals

                    if (top.GetEvents(
                            BindingFlags.DeclaredOnly | BindingFlags.Instance |
                            BindingFlags.NonPublic | BindingFlags.Public)
                        .Where(ev => ev.GetCustomAttributes().OfType<SignalAttribute>().Any())
                        .Any(eventSignal => eventSignal.Name == signalNameStr)
                       )
                    {
                        return true.ToGodotBool();
                    }

                    top = top.BaseType;
                }

                return false.ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                return false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool HasMethodUnknownParams(IntPtr scriptPtr, godot_string* method,
            godot_bool deep)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                if (!_scriptBridgeMap.TryGetValue(scriptPtr, out var scriptType))
                    return false.ToGodotBool();

                string methodStr = Marshaling.ConvertStringToManaged(*method);

                if (deep.ToBool())
                {
                    Type top = scriptType;
                    Type native = Object.InternalGetClassNativeBase(scriptType);

                    while (top != null && top != native)
                    {
                        var methodInfo = top.GetMethod(methodStr,
                            BindingFlags.DeclaredOnly | BindingFlags.Instance |
                            BindingFlags.NonPublic | BindingFlags.Public);

                        if (methodInfo != null)
                            return true.ToGodotBool();

                        top = top.BaseType;
                    }

                    top = native;
                    Type typeOfSystemObject = typeof(System.Object);
                    while (top != null && top != typeOfSystemObject)
                    {
                        bool found = top.GetMethods(BindingFlags.DeclaredOnly | BindingFlags.Instance |
                                                    BindingFlags.NonPublic | BindingFlags.Public)
                            .Where(m => m.GetCustomAttributes(false).OfType<GodotMethodAttribute>()
                                .Where(a => a.MethodName == methodStr)
                                .Any())
                            .Any();

                        if (found)
                            return true.ToGodotBool();

                        top = top.BaseType;
                    }

                    return false.ToGodotBool();
                }
                else
                {
                    var methodInfo = scriptType.GetMethod(methodStr, BindingFlags.DeclaredOnly | BindingFlags.Instance |
                                                                     BindingFlags.NonPublic | BindingFlags.Public);
                    return (methodInfo != null).ToGodotBool();
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                return false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        internal static godot_bool ScriptIsOrInherits(IntPtr scriptPtr, IntPtr scriptPtrMaybeBase)
        {
            try
            {
                if (!_scriptBridgeMap.TryGetValue(scriptPtr, out var scriptType))
                    return false.ToGodotBool();

                if (!_scriptBridgeMap.TryGetValue(scriptPtrMaybeBase, out var maybeBaseType))
                    return false.ToGodotBool();

                return (scriptType == maybeBaseType || maybeBaseType.IsAssignableFrom(scriptType)).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                return false.ToGodotBool();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool AddScriptBridge(IntPtr scriptPtr, godot_string* scriptPath)
        {
            try
            {
                string scriptPathStr = Marshaling.ConvertStringToManaged(*scriptPath);

                if (!_scriptLookupMap.TryGetValue(scriptPathStr, out var lookupInfo))
                    return false.ToGodotBool();

                _scriptBridgeMap.Add(scriptPtr, lookupInfo.ScriptType);

                return true.ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                return false.ToGodotBool();
            }
        }

        internal static void AddScriptBridgeWithType(IntPtr scriptPtr, Type scriptType)
            => _scriptBridgeMap.Add(scriptPtr, scriptType);

        [UnmanagedCallersOnly]
        internal static void RemoveScriptBridge(IntPtr scriptPtr)
        {
            try
            {
                _scriptBridgeMap.Remove(scriptPtr);
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptClassInfo(IntPtr scriptPtr, godot_bool* outTool,
            godot_dictionary* outRpcFunctionsDest)
        {
            try
            {
                // Performance is not critical here as this will be replaced with source generators.
                var scriptType = _scriptBridgeMap[scriptPtr];

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
                    *outTool = true.ToGodotBool();

                // RPC functions

                static RPCMode MemberGetRpcMode(MemberInfo memberInfo)
                {
                    var customAttrs = memberInfo.GetCustomAttributes(inherit: false);

                    if (customAttrs.OfType<AnyPeerAttribute>().Any())
                        return RPCMode.AnyPeer;

                    if (customAttrs.OfType<AuthorityAttribute>().Any())
                        return RPCMode.Auth;

                    return RPCMode.Disabled;
                }

                Dictionary<string, Dictionary> rpcFunctions = new();

                Type top = _scriptBridgeMap[scriptPtr];
                Type native = Object.InternalGetClassNativeBase(top);

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

                        var rpcMode = MemberGetRpcMode(method);

                        if (rpcMode == RPCMode.Disabled)
                            continue;

                        var rpcConfig = new Dictionary();
                        rpcConfig["rpc_mode"] = (int)rpcMode;
                        // TODO Transfer mode, channel
                        rpcConfig["transfer_mode"] = (int)TransferMode.Reliable;
                        rpcConfig["channel"] = 0;

                        rpcFunctions.Add(methodName, rpcConfig);
                    }

                    top = top.BaseType;
                }

                *outRpcFunctionsDest =
                    NativeFuncs.godotsharp_dictionary_new_copy(
                        (godot_dictionary)((Dictionary)rpcFunctions).NativeValue);
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                *outTool = false.ToGodotBool();
                *outRpcFunctionsDest = NativeFuncs.godotsharp_dictionary_new();
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool SwapGCHandleForType(IntPtr oldGCHandlePtr, IntPtr* outNewGCHandlePtr,
            godot_bool createWeak)
        {
            try
            {
                var oldGCHandle = GCHandle.FromIntPtr(oldGCHandlePtr);

                object target = oldGCHandle.Target;

                if (target == null)
                {
                    *outNewGCHandlePtr = IntPtr.Zero;
                    return false.ToGodotBool(); // Called after the managed side was collected, so nothing to do here
                }

                // Release the current weak handle and replace it with a strong handle.
                var newGCHandle = GCHandle.Alloc(target,
                    createWeak.ToBool() ? GCHandleType.Weak : GCHandleType.Normal);
                *outNewGCHandlePtr = GCHandle.ToIntPtr(newGCHandle);

                return true.ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
                *outNewGCHandlePtr = IntPtr.Zero;
                return false.ToGodotBool();
            }
        }
    }
}
