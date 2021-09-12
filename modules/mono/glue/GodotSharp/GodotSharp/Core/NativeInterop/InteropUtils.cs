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
            bool has_cs_script_instance = false;

            // First try to get the tied managed instance from a CSharpInstance script instance

            unsafe
            {
                gcHandlePtr = unmanaged_get_script_instance_managed(unmanaged, &has_cs_script_instance);
            }

            if (gcHandlePtr != IntPtr.Zero)
                return (Object)GCHandle.FromIntPtr(gcHandlePtr).Target;

            // Otherwise, if the object has a CSharpInstance script instance, return null

            if (has_cs_script_instance)
                return null;

            // If it doesn't have a CSharpInstance script instance, try with native instance bindings

            gcHandlePtr = unmanaged_get_instance_binding_managed(unmanaged);

            object target = gcHandlePtr != IntPtr.Zero ? GCHandle.FromIntPtr(gcHandlePtr).Target : null;

            if (target != null)
                return (Object)target;

            // If the native instance binding GC handle target was collected, create a new one

            gcHandlePtr = unmanaged_instance_binding_create_managed(unmanaged, gcHandlePtr);

            return gcHandlePtr != IntPtr.Zero ? (Object)GCHandle.FromIntPtr(gcHandlePtr).Target : null;
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern unsafe IntPtr unmanaged_get_script_instance_managed(IntPtr p_unmanaged,
            bool* r_has_cs_script_instance);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr unmanaged_get_instance_binding_managed(IntPtr p_unmanaged);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr unmanaged_instance_binding_create_managed(IntPtr p_unmanaged,
            IntPtr oldGCHandlePtr);

        public static void TieManagedToUnmanaged(Object managed, IntPtr unmanaged,
            StringName nativeName, bool refCounted, Type type, Type nativeType)
        {
            var gcHandle = GCHandle.Alloc(managed, refCounted ? GCHandleType.Weak : GCHandleType.Normal);

            if (type == nativeType)
            {
                unsafe
                {
                    godot_string_name nativeNameAux = nativeName.NativeValue;
                    internal_tie_native_managed_to_unmanaged(GCHandle.ToIntPtr(gcHandle), unmanaged,
                        &nativeNameAux, refCounted);
                }
            }
            else
            {
                IntPtr scriptPtr = internal_new_csharp_script();

                ScriptManagerBridge.AddScriptBridgeWithType(scriptPtr, type);

                // IMPORTANT: This must be called after AddScriptWithTypeBridge
                internal_tie_user_managed_to_unmanaged(GCHandle.ToIntPtr(gcHandle), unmanaged,
                    scriptPtr, refCounted);
            }
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern unsafe void internal_tie_native_managed_to_unmanaged(IntPtr gcHandleIntPtr,
            IntPtr unmanaged, godot_string_name* nativeName, bool refCounted);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_tie_user_managed_to_unmanaged(IntPtr gcHandleIntPtr,
            IntPtr unmanaged, IntPtr scriptPtr, bool refCounted);

        public static void TieManagedToUnmanagedWithPreSetup(Object managed, IntPtr unmanaged,
            Type type, Type nativeType)
        {
            if (type == nativeType)
                return;

            var strongGCHandle = GCHandle.Alloc(managed, GCHandleType.Normal);
            internal_tie_managed_to_unmanaged_with_pre_setup(GCHandle.ToIntPtr(strongGCHandle), unmanaged);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_tie_managed_to_unmanaged_with_pre_setup(
            IntPtr gcHandleIntPtr, IntPtr unmanaged);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr internal_new_csharp_script();

        public static unsafe Object EngineGetSingleton(string name)
        {
            using godot_string src = Marshaling.mono_string_to_godot(name);
            return UnmanagedGetManaged(NativeFuncs.godotsharp_engine_get_singleton(&src));
        }
    }
}
