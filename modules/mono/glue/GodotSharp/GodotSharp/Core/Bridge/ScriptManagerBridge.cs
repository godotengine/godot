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

        internal static void FrameCallback()
        {
            Dispatcher.DefaultGodotTaskScheduler?.Activate();
        }

        internal static unsafe IntPtr CreateManagedForGodotObjectBinding(godot_string_name* nativeTypeName,
            IntPtr godotObject)
        {
            Type nativeType = TypeGetProxyClass(nativeTypeName);
            var obj = (Object)FormatterServices.GetUninitializedObject(nativeType);

            obj.NativePtr = godotObject;

            var ctor = nativeType.GetConstructor(
                BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
                null, Type.EmptyTypes, null);
            _ = ctor!.Invoke(obj, null);

            return GCHandle.ToIntPtr(GCHandle.Alloc(obj));
        }

        internal static unsafe void CreateManagedForGodotObjectScriptInstance(IntPtr scriptPtr, IntPtr godotObject,
            godot_variant** args, int argCount)
        {
            // Performance is not critical here as this will be replaced with source generators.
            Type scriptType = _scriptBridgeMap[scriptPtr];
            var obj = (Object)FormatterServices.GetUninitializedObject(scriptType);

            obj.NativePtr = godotObject;

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
                invokeParams[i] = Marshaling.variant_to_mono_object_of_type(
                    args[i], parameters[i].ParameterType);
            }

            ctor.Invoke(obj, invokeParams);
        }

        private static unsafe void GetScriptNativeName(IntPtr scriptPtr, godot_string_name* r_res)
        {
            // Performance is not critical here as this will be replaced with source generators.
            if (!_scriptBridgeMap.TryGetValue(scriptPtr, out var scriptType))
            {
                *r_res = default;
                return;
            }

            var native = Object.InternalGetClassNativeBase(scriptType);

            var field = native?.GetField("NativeName", BindingFlags.DeclaredOnly | BindingFlags.Static |
                                                       BindingFlags.Public | BindingFlags.NonPublic);

            if (field == null)
            {
                *r_res = default;
                return;
            }

            var nativeName = (StringName)field.GetValue(null);

            *r_res = NativeFuncs.godotsharp_string_name_new_copy(nativeName.NativeValue);
        }

        private static void SetGodotObjectPtr(IntPtr gcHandlePtr, IntPtr newPtr)
        {
            var target = (Object)GCHandle.FromIntPtr(gcHandlePtr).Target;
            if (target != null)
                target.NativePtr = newPtr;
        }

        private static unsafe Type TypeGetProxyClass(godot_string_name* nativeTypeName)
        {
            // Performance is not critical here as this will be replaced with a generated dictionary.
            using var stringName = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(nativeTypeName));
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

        internal static void LookupScriptsInAssembly(Assembly assembly)
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

        internal static unsafe void RaiseEventSignal(IntPtr ownerGCHandlePtr,
            godot_string_name* eventSignalName, godot_variant** args, int argCount, bool* r_ownerIsNull)
        {
            var owner = (Object)GCHandle.FromIntPtr(ownerGCHandlePtr).Target;

            if (owner == null)
            {
                *r_ownerIsNull = true;
                return;
            }

            *r_ownerIsNull = false;

            owner.InternalRaiseEventSignal(eventSignalName, args, argCount);
        }

        internal static unsafe void GetScriptSignalList(IntPtr scriptPtr, godot_dictionary* r_retSignals)
        {
            // Performance is not critical here as this will be replaced with source generators.
            using var signals = new Dictionary();

            Type top = _scriptBridgeMap[scriptPtr];
            Type native = Object.InternalGetClassNativeBase(top);

            while (top != null && top != native)
            {
                // Legacy signals

                foreach (var signalDelegate in top
                    .GetNestedTypes(BindingFlags.DeclaredOnly | BindingFlags.NonPublic | BindingFlags.Public)
                    .Where(nestedType => typeof(Delegate).IsAssignableFrom(nestedType))
                    .Where(@delegate => @delegate.GetCustomAttributes().OfType<SignalAttribute>().Any()))
                {
                    var invokeMethod = signalDelegate.GetMethod("Invoke");

                    if (invokeMethod == null)
                        throw new MissingMethodException(signalDelegate.FullName, "Invoke");

                    var signalParams = new Collections.Array();

                    foreach (var parameters in invokeMethod.GetParameters())
                    {
                        var paramType = Marshaling.managed_to_variant_type(
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
                        var paramType = Marshaling.managed_to_variant_type(
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

            *r_retSignals = NativeFuncs.godotsharp_dictionary_new_copy(signals.NativeValue);
        }

        internal static unsafe bool HasScriptSignal(IntPtr scriptPtr, godot_string* signalName)
        {
            // Performance is not critical here as this will be replaced with source generators.
            using var signals = new Dictionary();

            string signalNameStr = Marshaling.mono_string_from_godot(*signalName);

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
                    return true;
                }

                // Event signals

                if (top.GetEvents(
                        BindingFlags.DeclaredOnly | BindingFlags.Instance |
                        BindingFlags.NonPublic | BindingFlags.Public)
                    .Where(ev => ev.GetCustomAttributes().OfType<SignalAttribute>().Any())
                    .Any(eventSignal => eventSignal.Name == signalNameStr)
                )
                {
                    return true;
                }

                top = top.BaseType;
            }

            return false;
        }

        internal static unsafe bool HasMethodUnknownParams(IntPtr scriptPtr, godot_string* method, bool deep)
        {
            // Performance is not critical here as this will be replaced with source generators.
            if (!_scriptBridgeMap.TryGetValue(scriptPtr, out var scriptType))
                return false;

            string methodStr = Marshaling.mono_string_from_godot(*method);

            if (deep)
            {
                Type top = scriptType;
                Type native = Object.InternalGetClassNativeBase(scriptType);

                while (top != null && top != native)
                {
                    var methodInfo = top.GetMethod(methodStr,
                        BindingFlags.DeclaredOnly | BindingFlags.Instance |
                        BindingFlags.NonPublic | BindingFlags.Public);

                    if (methodInfo != null)
                        return true;

                    top = top.BaseType;
                }

                return false;
            }
            else
            {
                var methodInfo = scriptType.GetMethod(methodStr, BindingFlags.DeclaredOnly | BindingFlags.Instance |
                                                                 BindingFlags.NonPublic | BindingFlags.Public);
                return methodInfo != null;
            }
        }

        internal static bool ScriptIsOrInherits(IntPtr scriptPtr, IntPtr scriptPtrMaybeBase)
        {
            if (!_scriptBridgeMap.TryGetValue(scriptPtr, out var scriptType))
                return false;

            if (!_scriptBridgeMap.TryGetValue(scriptPtrMaybeBase, out var maybeBaseType))
                return false;

            return scriptType == maybeBaseType || maybeBaseType.IsAssignableFrom(scriptType);
        }

        internal static unsafe bool AddScriptBridge(IntPtr scriptPtr, godot_string* scriptPath)
        {
            string scriptPathStr = Marshaling.mono_string_from_godot(*scriptPath);

            if (!_scriptLookupMap.TryGetValue(scriptPathStr, out var lookupInfo))
                return false;

            _scriptBridgeMap.Add(scriptPtr, lookupInfo.ScriptType);

            return true;
        }

        internal static void AddScriptBridgeWithType(IntPtr scriptPtr, Type scriptType)
            => _scriptBridgeMap.Add(scriptPtr, scriptType);

        internal static void RemoveScriptBridge(IntPtr scriptPtr)
            => _scriptBridgeMap.Remove(scriptPtr);

        internal static unsafe void UpdateScriptClassInfo(IntPtr scriptPtr, bool* r_tool,
            godot_dictionary* r_rpcFunctionsDest)
        {
            // Performance is not critical here as this will be replaced with source generators.
            var scriptType = _scriptBridgeMap[scriptPtr];

            *r_tool = scriptType.GetCustomAttributes(inherit: false)
                .OfType<ToolAttribute>()
                .Any();

            if (!*r_tool && scriptType.IsNested)
            {
                *r_tool = scriptType.DeclaringType?.GetCustomAttributes(inherit: false)
                    .OfType<ToolAttribute>()
                    .Any() ?? false;
            }

            if (!*r_tool && scriptType.Assembly.GetName().Name == "GodotTools")
                *r_tool = true;

            // RPC functions

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

                    var rpcAttr = method.GetCustomAttributes(inherit: false)
                        .OfType<RPCAttribute>().FirstOrDefault();

                    if (rpcAttr == null)
                        continue;

                    var rpcConfig = new Dictionary();

                    rpcConfig["rpc_mode"] = (long)rpcAttr.Mode;
                    rpcConfig["call_local"] = rpcAttr.CallLocal;
                    rpcConfig["transfer_mode"] = (long)rpcAttr.TransferMode;
                    rpcConfig["channel"] = rpcAttr.TransferChannel;

                    rpcFunctions.Add(methodName, rpcConfig);
                }

                top = top.BaseType;
            }

            *r_rpcFunctionsDest = NativeFuncs.godotsharp_dictionary_new_copy(((Dictionary)rpcFunctions).NativeValue);
        }

        internal static unsafe bool SwapGCHandleForType(IntPtr oldGCHandlePtr, IntPtr* r_newGCHandlePtr,
            bool createWeak)
        {
            var oldGCHandle = GCHandle.FromIntPtr(oldGCHandlePtr);

            object target = oldGCHandle.Target;

            if (target == null)
            {
                oldGCHandle.Free();
                *r_newGCHandlePtr = IntPtr.Zero;
                return false; // Called after the managed side was collected, so nothing to do here
            }

            // Release the current weak handle and replace it with a strong handle.
            var newGCHandle = GCHandle.Alloc(target, createWeak ? GCHandleType.Weak : GCHandleType.Normal);

            oldGCHandle.Free();
            *r_newGCHandlePtr = GCHandle.ToIntPtr(newGCHandle);
            return true;
        }
    }
}
