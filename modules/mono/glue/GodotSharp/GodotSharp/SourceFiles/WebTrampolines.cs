using System;
using System.Runtime.InteropServices;

namespace GodotPlugins.Game
{
    internal static partial class Initializer
    {
        // Generate web trampolines.
        // C# doesn't automatically generate them if delegate is of type 'delegate* unmanaged<[contains long or ulong]>'.
        // Should be updated when new function pointers with long or ulong argument type are added.

        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        private delegate IntPtr classdb_get_method_bind_sig(IntPtr _1, IntPtr _2, long _3);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr godotsharp_method_bind_get_method_with_compatibility_sig(IntPtr _0, IntPtr _1, ulong _2);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate Godot.Color godotsharp_color_from_ok_hsl_sig(float _0, float _1, float _2, float _3);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate float godotsharp_color_get_ok_hsl_get_sig(IntPtr _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate long godotsharp_variant_as_int_sig(IntPtr _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate double godotsharp_variant_as_float_sig(IntPtr _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void godotsharp_packed_byte_array_decompress_sig(IntPtr _0, long _1, int _2, IntPtr _3);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void godotsharp_packed_byte_array_decompress_dynamic_sig(IntPtr _0, long _1, int _2, IntPtr _3);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        internal delegate IntPtr godotsharp_instance_from_id_sig(ulong _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        internal delegate uint godotsharp_rand_from_seed_sig(ulong _0, IntPtr _1);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void godotsharp_seed_sig(ulong _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate float godotsharp_randf_sig();
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate double godotsharp_randf_range_sig(double _0, double _1);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate long godotsharp_array_size_sig(IntPtr _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate Godot.Error godotsharp_stack_info_vector_resize_sig(IntPtr _0, int _1);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate Godot.Error godotsharp_internal_signal_awaiter_connect_sig(IntPtr _0, IntPtr _1, IntPtr _3, IntPtr _4);
    }
}
