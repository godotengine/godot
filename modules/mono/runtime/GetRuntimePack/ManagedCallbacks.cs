using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot.NativeInterop
{
    public enum godot_bool : byte { }

    public ref struct godot_ref { }

    public ref struct godot_variant_call_error { }

    public ref struct godot_csharp_type_info { }

    public ref struct godot_variant { }

    public ref struct godot_string { }

    public ref struct godot_string_name { }

    public ref struct godot_array { }

    public ref struct godot_dictionary { }
}

namespace Godot
{
    public class SignalAwaiter
    {
        [UnmanagedCallersOnly]
        internal static unsafe void SignalCallback(IntPtr awaiterGCHandlePtr, godot_variant** args, int argCount, godot_bool* outAwaiterIsNull) { }
    }

    internal static class DelegateUtils
    {
        [UnmanagedCallersOnly]
        internal static godot_bool DelegateEquals(IntPtr delegateGCHandleA, IntPtr delegateGCHandleB)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static int DelegateHash(IntPtr delegateGCHandle)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe int GetArgumentCount(IntPtr delegateGCHandle, godot_bool* outIsValid)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void InvokeWithVariantArgs(IntPtr delegateGCHandle, void* trampoline, godot_variant** args, int argc, godot_variant* outRet) { }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool TrySerializeDelegateWithGCHandle(IntPtr delegateGCHandle,
            godot_array* nSerializedData)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool TryDeserializeDelegateWithGCHandle(godot_array* nSerializedData,
            IntPtr* delegateGCHandle)
        {
            throw null!;
        }
    }

    internal static class DebuggingUtils
    {
        [UnmanagedCallersOnly]
        internal static unsafe void GetCurrentStackInfo(void* destVector) { }
    }

    internal static class DisposablesTracker
    {
        [UnmanagedCallersOnly]
        internal static void OnGodotShuttingDown() { }
    }

    public static class GD
    {
        [UnmanagedCallersOnly]
        internal static void OnCoreApiAssemblyLoaded(godot_bool isDebug) { }
    }
}

namespace Godot.Bridge
{
    public static class ScriptManagerBridge
    {
        [UnmanagedCallersOnly]
        internal static void FrameCallback() { }

        [UnmanagedCallersOnly]
        internal static unsafe IntPtr CreateManagedForGodotObjectBinding(godot_string_name* nativeTypeName, IntPtr godotObject)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CreateManagedForGodotObjectScriptInstance(IntPtr scriptPtr, IntPtr godotObject, godot_variant** args, int argCount)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetScriptNativeName(IntPtr scriptPtr, godot_string_name* outRes) { }

        [UnmanagedCallersOnly]
        internal static unsafe void GetGlobalClassName(godot_string* scriptPath, godot_string* outBaseType, godot_string* outIconPath, godot_string* outClassName) { }

        [UnmanagedCallersOnly]
        internal static void SetGodotObjectPtr(IntPtr gcHandlePtr, IntPtr newPtr) { }

        [UnmanagedCallersOnly]
        internal static unsafe void RaiseEventSignal(IntPtr ownerGCHandlePtr, godot_string_name* eventSignalName, godot_variant** args, int argCount, godot_bool* outOwnerIsNull) { }

        [UnmanagedCallersOnly]
        internal static godot_bool ScriptIsOrInherits(IntPtr scriptPtr, IntPtr scriptPtrMaybeBase)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool AddScriptBridge(IntPtr scriptPtr, godot_string* scriptPath)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetOrCreateScriptBridgeForPath(godot_string* scriptPath, godot_ref* outScript) { }

        [UnmanagedCallersOnly]
        internal static void RemoveScriptBridge(IntPtr scriptPtr) { }

        [UnmanagedCallersOnly]
        internal static godot_bool TryReloadRegisteredScriptWithClass(IntPtr scriptPtr)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void UpdateScriptClassInfo(IntPtr scriptPtr, godot_csharp_type_info* outTypeInfo, godot_array* outMethodsDest, godot_dictionary* outRpcFunctionsDest, godot_dictionary* outEventSignalsDest, godot_ref* outBaseScript) { }

        [UnmanagedCallersOnly]
        internal static unsafe void GetPropertyInfoList(IntPtr scriptPtr, delegate* unmanaged<IntPtr, godot_string*, void*, int, void> addPropInfoFunc) { }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool CallStatic(IntPtr scriptPtr, godot_string_name* method, godot_variant** args, int argCount, godot_variant_call_error* refCallError, godot_variant* ret)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void GetPropertyDefaultValues(IntPtr scriptPtr, delegate* unmanaged<IntPtr, void*, int, void> addDefValFunc) { }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool SwapGCHandleForType(IntPtr oldGCHandlePtr, IntPtr* outNewGCHandlePtr, godot_bool createWeak)
        {
            throw null!;
        }
    }

    internal static class CSharpInstanceBridge
    {
        [UnmanagedCallersOnly]
        internal static unsafe godot_bool Call(IntPtr godotObjectGCHandle, godot_string_name* method, godot_variant** args, int argCount, godot_variant_call_error* refCallError, godot_variant* ret)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool Set(IntPtr godotObjectGCHandle, godot_string_name* name, godot_variant* value)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool Get(IntPtr godotObjectGCHandle, godot_string_name* name, godot_variant* outRet)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static void CallDispose(IntPtr godotObjectGCHandle, godot_bool okIfNull) { }

        [UnmanagedCallersOnly]
        internal static unsafe void CallToString(IntPtr godotObjectGCHandle, godot_string* outRes, godot_bool* outValid) { }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool HasMethodUnknownParams(IntPtr godotObjectGCHandle, godot_string_name* method)
        {
            throw null!;
        }

        [UnmanagedCallersOnly]
        internal static unsafe void SerializeState(IntPtr godotObjectGCHandle, godot_dictionary* propertiesState, godot_dictionary* signalEventsState) { }

        [UnmanagedCallersOnly]
        internal static unsafe void DeserializeState(IntPtr godotObjectGCHandle, godot_dictionary* propertiesState, godot_dictionary* signalEventsState) { }
    }

    internal static class GCHandleBridge
    {
        [UnmanagedCallersOnly]
        internal static void FreeGCHandle(IntPtr gcHandlePtr) { }

        [UnmanagedCallersOnly]
        internal static godot_bool GCHandleIsTargetCollectible(IntPtr gcHandlePtr)
        {
            throw null!;
        }
    }
}
