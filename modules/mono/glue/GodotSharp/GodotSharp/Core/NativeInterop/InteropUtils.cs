using System;
using System.Runtime.InteropServices;
using Godot.Bridge;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    internal static class InteropUtils
    {
        public static GodotObject UnmanagedGetManaged(IntPtr unmanaged)
        {
            // The native pointer may be null
            if (unmanaged == IntPtr.Zero)
                return null;

            IntPtr gcHandlePtr;
            godot_bool hasCsScriptInstance;

            // First try to get the tied managed instance from a CSharpInstance script instance

            gcHandlePtr = NativeFuncs.godotsharp_internal_unmanaged_get_script_instance_managed(
                unmanaged, out hasCsScriptInstance);

            if (gcHandlePtr != IntPtr.Zero)
                return (GodotObject)GCHandle.FromIntPtr(gcHandlePtr).Target;

            // Otherwise, if the object has a CSharpInstance script instance, return null

            if (hasCsScriptInstance.ToBool())
                return null;

            // If it doesn't have a CSharpInstance script instance, try with native instance bindings

            gcHandlePtr = NativeFuncs.godotsharp_internal_unmanaged_get_instance_binding_managed(unmanaged);

            object target = gcHandlePtr != IntPtr.Zero ? GCHandle.FromIntPtr(gcHandlePtr).Target : null;

            if (target != null)
                return (GodotObject)target;

            // If the native instance binding GC handle target was collected, create a new one

            gcHandlePtr = NativeFuncs.godotsharp_internal_unmanaged_instance_binding_create_managed(
                unmanaged, gcHandlePtr);

            return gcHandlePtr != IntPtr.Zero ? (GodotObject)GCHandle.FromIntPtr(gcHandlePtr).Target : null;
        }

        public static void TieManagedToUnmanaged(GodotObject managed, IntPtr unmanaged,
            StringName nativeName, bool refCounted, Type type, Type nativeType)
        {
            var gcHandle = refCounted ?
                CustomGCHandle.AllocWeak(managed) :
                CustomGCHandle.AllocStrong(managed, type);

            if (type == nativeType)
            {
                var nativeNameSelf = (godot_string_name)nativeName.NativeValue;
                NativeFuncs.godotsharp_internal_tie_native_managed_to_unmanaged(
                    GCHandle.ToIntPtr(gcHandle), unmanaged, nativeNameSelf, refCounted.ToGodotBool());
            }
            else
            {
                unsafe
                {
                    // We don't dispose `script` ourselves here.
                    // `tie_user_managed_to_unmanaged` does it for us to avoid another P/Invoke call.
                    godot_ref script;
                    ScriptManagerBridge.GetOrLoadOrCreateScriptForType(type, &script);

                    // IMPORTANT: This must be called after GetOrCreateScriptBridgeForType
                    NativeFuncs.godotsharp_internal_tie_user_managed_to_unmanaged(
                        GCHandle.ToIntPtr(gcHandle), unmanaged, &script, refCounted.ToGodotBool());
                }
            }
        }

        public static void TieManagedToUnmanagedWithPreSetup(GodotObject managed, IntPtr unmanaged,
            Type type, Type nativeType)
        {
            if (type == nativeType)
                return;

            var strongGCHandle = CustomGCHandle.AllocStrong(managed);
            NativeFuncs.godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup(
                GCHandle.ToIntPtr(strongGCHandle), unmanaged);
        }

        public static GodotObject EngineGetSingleton(string name)
        {
            using godot_string src = Marshaling.ConvertStringToNative(name);
            return UnmanagedGetManaged(NativeFuncs.godotsharp_engine_get_singleton(src));
        }
    }
}
