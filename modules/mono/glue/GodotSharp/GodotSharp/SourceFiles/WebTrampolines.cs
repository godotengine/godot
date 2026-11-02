using System;
using System.Runtime.InteropServices;

namespace GodotPlugins.Game
{
    internal static partial class Initializer
    {
        // 2dog: declare each NativeFuncs signature so wasm generates unmanaged calli trampolines.
        // 2dog: WebTrampolineCoverageTests verifies coverage.

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
        public delegate int godotsharp_stack_info_vector_resize_sig(IntPtr _0, int _1);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate int godotsharp_internal_signal_awaiter_connect_sig(IntPtr _0, IntPtr _1, IntPtr _3, IntPtr _4);

        // 2dog: pointer/int32-only shapes, previously covered only by unrelated DllImport scans.
        // 2dog: Named by wasm signature cookie: return type first ('i' ptr/int32, 'v' void), then args.
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr sig_i();
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_v();
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr sig_ii(IntPtr _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_vi(IntPtr _0);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_vii(IntPtr _0, IntPtr _1);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr sig_iii(IntPtr _0, IntPtr _1);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr sig_iiii(IntPtr _0, IntPtr _1, IntPtr _2);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr sig_iiiii(IntPtr _0, IntPtr _1, IntPtr _2, IntPtr _3);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_viii(IntPtr _0, IntPtr _1, IntPtr _2);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_viiii(IntPtr _0, IntPtr _1, IntPtr _2, IntPtr _3);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate IntPtr sig_iiiiii(IntPtr _0, IntPtr _1, IntPtr _2, IntPtr _3, IntPtr _4);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_viiiii(IntPtr _0, IntPtr _1, IntPtr _2, IntPtr _3, IntPtr _4);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_viiiiii(IntPtr _0, IntPtr _1, IntPtr _2, IntPtr _3, IntPtr _4, IntPtr _5);
        [UnmanagedFunctionPointer(CallingConvention.Winapi)]
        public delegate void sig_viiiiiii(IntPtr _0, IntPtr _1, IntPtr _2, IntPtr _3, IntPtr _4, IntPtr _5, IntPtr _6);
    }
}
