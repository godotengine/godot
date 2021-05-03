using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;

#endif

namespace Godot.NativeInterop
{
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_bool
    {
        public byte _value;

        public unsafe godot_bool(bool value) => _value = *(byte*)&value;

        public static unsafe implicit operator bool(godot_bool godotBool) => *(bool*)&godotBool._value;
        public static implicit operator godot_bool(bool @bool) => new godot_bool(@bool);
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_ref : IDisposable
    {
        internal IntPtr _reference;

        public void Dispose()
        {
            if (_reference == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_ref_destroy(ref this);
            _reference = IntPtr.Zero;
        }
    }

    [SuppressMessage("ReSharper", "InconsistentNaming")]
    internal enum godot_variant_call_error_error
    {
        GODOT_CALL_ERROR_CALL_OK = 0,
        GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD,
        GODOT_CALL_ERROR_CALL_ERROR_INVALID_ARGUMENT,
        GODOT_CALL_ERROR_CALL_ERROR_TOO_MANY_ARGUMENTS,
        GODOT_CALL_ERROR_CALL_ERROR_TOO_FEW_ARGUMENTS,
        GODOT_CALL_ERROR_CALL_ERROR_INSTANCE_IS_NULL,
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_variant_call_error
    {
        godot_variant_call_error_error error;
        int argument;
        Godot.Variant.Type expected;
    }

    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_variant : IDisposable
    {
        [FieldOffset(0)] public Godot.Variant.Type _type;

        // There's padding here

        [FieldOffset(8)] internal godot_variant_data _data;

        [StructLayout(LayoutKind.Explicit)]
        // ReSharper disable once InconsistentNaming
        internal unsafe struct godot_variant_data
        {
            [FieldOffset(0)] public godot_bool _bool;
            [FieldOffset(0)] public long _int;
            [FieldOffset(0)] public double _float;
            [FieldOffset(0)] public Transform2D* _transform2d;
            [FieldOffset(0)] public AABB* _aabb;
            [FieldOffset(0)] public Basis* _basis;
            [FieldOffset(0)] public Transform3D* _transform3d;
            [FieldOffset(0)] private godot_variant_data_mem _mem;

            // The following fields are not in the C++ union, but this is how they're stored in _mem.
            [FieldOffset(0)] public godot_string_name _m_string_name;
            [FieldOffset(0)] public godot_string _m_string;
            [FieldOffset(0)] public Vector3 _m_vector3;
            [FieldOffset(0)] public Vector3i _m_vector3i;
            [FieldOffset(0)] public Vector2 _m_vector2;
            [FieldOffset(0)] public Vector2i _m_vector2i;
            [FieldOffset(0)] public Rect2 _m_rect2;
            [FieldOffset(0)] public Rect2i _m_rect2i;
            [FieldOffset(0)] public Plane _m_plane;
            [FieldOffset(0)] public Quaternion _m_quaternion;
            [FieldOffset(0)] public Color _m_color;
            [FieldOffset(0)] public godot_node_path _m_node_path;
            [FieldOffset(0)] public RID _m_rid;
            [FieldOffset(0)] public godot_variant_obj_data _m_obj_data;
            [FieldOffset(0)] public godot_callable _m_callable;
            [FieldOffset(0)] public godot_signal _m_signal;
            [FieldOffset(0)] public godot_dictionary _m_dictionary;
            [FieldOffset(0)] public godot_array _m_array;

            [StructLayout(LayoutKind.Sequential)]
            // ReSharper disable once InconsistentNaming
            internal struct godot_variant_obj_data
            {
                public UInt64 id;
                public IntPtr obj;
            }

            [StructLayout(LayoutKind.Sequential)]
            // ReSharper disable once InconsistentNaming
            private struct godot_variant_data_mem
            {
#pragma warning disable 169
                private real_t _mem0;
                private real_t _mem1;
                private real_t _mem2;
                private real_t _mem3;
#pragma warning restore 169
            }
        }

        public void Dispose()
        {
            switch (_type)
            {
                case Variant.Type.Nil:
                case Variant.Type.Bool:
                case Variant.Type.Int:
                case Variant.Type.Float:
                case Variant.Type.Vector2:
                case Variant.Type.Vector2i:
                case Variant.Type.Rect2:
                case Variant.Type.Rect2i:
                case Variant.Type.Vector3:
                case Variant.Type.Vector3i:
                case Variant.Type.Plane:
                case Variant.Type.Quaternion:
                case Variant.Type.Color:
                case Variant.Type.Rid:
                    return;
            }

            NativeFuncs.godotsharp_variant_destroy(ref this);
            _type = Variant.Type.Nil;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_string : IDisposable
    {
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_string_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_string_name : IDisposable
    {
        internal IntPtr _data;

        public void Dispose()
        {
            if (_data == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_string_name_destroy(ref this);
            _data = IntPtr.Zero;
        }

        // An static method because an instance method could result in a hidden copy if called on an `in` parameter.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEmpty(in godot_string_name name) =>
            // This is all that's needed to check if it's empty. Equivalent to `== StringName()` in C++.
            name._data == IntPtr.Zero;
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_node_path : IDisposable
    {
        internal IntPtr _data;

        public void Dispose()
        {
            if (_data == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_node_path_destroy(ref this);
            _data = IntPtr.Zero;
        }

        // An static method because an instance method could result in a hidden copy if called on an `in` parameter.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool IsEmpty(in godot_node_path nodePath) =>
            // This is all that's needed to check if it's empty. It's what the `is_empty()` C++ method does.
            nodePath._data == IntPtr.Zero;
    }

    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_signal : IDisposable
    {
        [FieldOffset(0)] public godot_string_name _name;

        // There's padding here on 32-bit

        [FieldOffset(8)] public UInt64 _objectId;

        public void Dispose()
        {
            if (_name._data == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_signal_destroy(ref this);
            _name._data = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_callable : IDisposable
    {
        [FieldOffset(0)] public godot_string_name _method;

        // There's padding here on 32-bit

        [FieldOffset(8)] public UInt64 _objectId;
        [FieldOffset(8)] public IntPtr _custom;

        public void Dispose()
        {
            if (_method._data == IntPtr.Zero && _custom == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_callable_destroy(ref this);
            _method._data = IntPtr.Zero;
            _custom = IntPtr.Zero;
        }
    }

    // A correctly constructed value needs to call the native default constructor to allocate `_p`.
    // Don't pass a C# default constructed `godot_array` to native code, unless it's going to
    // be re-assigned a new value (the copy constructor checks if `_p` is null so that's fine).
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_array : IDisposable
    {
        internal IntPtr _p;

        public void Dispose()
        {
            if (_p == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_array_destroy(ref this);
            _p = IntPtr.Zero;
        }
    }

    // IMPORTANT:
    // A correctly constructed value needs to call the native default constructor to allocate `_p`.
    // Don't pass a C# default constructed `godot_dictionary` to native code, unless it's going to
    // be re-assigned a new value (the copy constructor checks if `_p` is null so that's fine).
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_dictionary : IDisposable
    {
        internal IntPtr _p;

        public void Dispose()
        {
            if (_p == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_dictionary_destroy(ref this);
            _p = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_byte_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_byte_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_int32_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_int32_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_int64_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_int64_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_float32_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_float32_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_float64_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_float64_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_string_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_string_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_vector2_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_vector2_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_vector3_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_vector3_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    internal struct godot_packed_color_array : IDisposable
    {
        internal IntPtr _writeProxy;
        internal IntPtr _ptr;

        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_packed_color_array_destroy(ref this);
            _ptr = IntPtr.Zero;
        }
    }
}
