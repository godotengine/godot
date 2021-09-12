using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.Bridge;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    internal static class InteropUtils
    {
        public static Object UnmanagedGetManaged(IntPtr unmanaged)
        {
            // The native pointer may be null
            if (unmanaged == IntPtr.Zero)
                return null;

            IntPtr gcHandlePtr;
            godot_bool has_cs_script_instance = false.ToGodotBool();

            // First try to get the tied managed instance from a CSharpInstance script instance

            unsafe
            {
                gcHandlePtr = NativeFuncs.godotsharp_internal_unmanaged_get_script_instance_managed(
                    unmanaged, &has_cs_script_instance);
            }

            if (gcHandlePtr != IntPtr.Zero)
                return (Object)GCHandle.FromIntPtr(gcHandlePtr).Target;

            // Otherwise, if the object has a CSharpInstance script instance, return null

            if (has_cs_script_instance.ToBool())
                return null;

            // If it doesn't have a CSharpInstance script instance, try with native instance bindings

            gcHandlePtr = NativeFuncs.godotsharp_internal_unmanaged_get_instance_binding_managed(unmanaged);

            object target = gcHandlePtr != IntPtr.Zero ? GCHandle.FromIntPtr(gcHandlePtr).Target : null;

            if (target != null)
                return (Object)target;

            // If the native instance binding GC handle target was collected, create a new one

            gcHandlePtr = NativeFuncs.godotsharp_internal_unmanaged_instance_binding_create_managed(
                unmanaged, gcHandlePtr);

            return gcHandlePtr != IntPtr.Zero ? (Object)GCHandle.FromIntPtr(gcHandlePtr).Target : null;
        }

        public static void TieManagedToUnmanaged(Object managed, IntPtr unmanaged,
            StringName nativeName, bool refCounted, Type type, Type nativeType)
        {
            var gcHandle = GCHandle.Alloc(managed, refCounted ? GCHandleType.Weak : GCHandleType.Normal);

            if (type == nativeType)
            {
                unsafe
                {
                    godot_string_name nativeNameAux = nativeName.NativeValue;
                    NativeFuncs.godotsharp_internal_tie_native_managed_to_unmanaged(
                        GCHandle.ToIntPtr(gcHandle), unmanaged, &nativeNameAux, refCounted.ToGodotBool());
                }
            }
            else
            {
                IntPtr scriptPtr = NativeFuncs.godotsharp_internal_new_csharp_script();

                ScriptManagerBridge.AddScriptBridgeWithType(scriptPtr, type);

                // IMPORTANT: This must be called after AddScriptWithTypeBridge
                NativeFuncs.godotsharp_internal_tie_user_managed_to_unmanaged(
                    GCHandle.ToIntPtr(gcHandle), unmanaged, scriptPtr, refCounted.ToGodotBool());
            }
        }

        public static void TieManagedToUnmanagedWithPreSetup(Object managed, IntPtr unmanaged,
            Type type, Type nativeType)
        {
            if (type == nativeType)
                return;

            var strongGCHandle = GCHandle.Alloc(managed, GCHandleType.Normal);
            NativeFuncs.godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup(
                GCHandle.ToIntPtr(strongGCHandle), unmanaged);
        }

        public static unsafe Object EngineGetSingleton(string name)
        {
            using godot_string src = Marshaling.mono_string_to_godot(name);
            return UnmanagedGetManaged(NativeFuncs.godotsharp_engine_get_singleton(&src));
        }
    }
}
