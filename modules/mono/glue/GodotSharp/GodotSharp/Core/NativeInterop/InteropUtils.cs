using System;
using System.Runtime.CompilerServices;

namespace Godot.NativeInterop
{
    internal static class InteropUtils
    {
        public static Object UnmanagedGetManaged(IntPtr unmanaged)
        {
            // TODO: Move to C#
            return internal_unmanaged_get_managed(unmanaged);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Object internal_unmanaged_get_managed(IntPtr unmanaged);

        public static void TieManagedToUnmanaged(Object managed, IntPtr unmanaged)
        {
            // TODO: Move to C#
            internal_tie_managed_to_unmanaged(managed, unmanaged);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_tie_managed_to_unmanaged(Object managed, IntPtr unmanaged);

        public static unsafe Object EngineGetSingleton(string name)
        {
            using godot_string src = Marshaling.mono_string_to_godot(name);
            return UnmanagedGetManaged(NativeFuncs.godotsharp_engine_get_singleton(&src));
        }
    }
}
