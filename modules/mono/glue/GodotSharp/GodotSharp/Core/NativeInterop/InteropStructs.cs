using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.NativeInterop
{
    // NOTES:
    // ref structs cannot implement interfaces, but they still work in `using` directives if they declare Dispose()

    /// <summary>
    /// Collection of extensions for the Godot representation of a <see langword="bool"/>.
    /// </summary>
    public static class GodotBoolExtensions
    {
        /// <summary>
        /// Handles conversion of a <see langword="bool"/> to a <see cref="godot_bool"/>.
        /// </summary>
        /// <param name="bool">The <see langword="bool"/> to convert.</param>
        /// <returns>The newly converted <see cref="godot_bool"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe godot_bool ToGodotBool(this bool @bool)
        {
            return *(godot_bool*)&@bool;
        }

        /// <summary>
        /// Handles conversion of a <see cref="godot_bool"/> to a <see langword="bool"/>.
        /// </summary>
        /// <param name="godotBool">The <see cref="godot_bool"/> to convert.</param>
        /// <returns>The newly converted <see langword="bool"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe bool ToBool(this godot_bool godotBool)
        {
            return *(bool*)&godotBool;
        }
    }

    // Apparently a struct with a byte is not blittable? It crashes when calling a UnmanagedCallersOnly function ptr.
    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="bool"/>.
    /// </summary>
    // ReSharper disable once InconsistentNaming
    public enum godot_bool : byte
    {
        /// <summary>
        /// <see langword="true"/>.
        /// </summary>
        True = 1,
        /// <summary>
        /// <see langword="false"/>.
        /// </summary>
        False = 0
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="ref"/>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_ref
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_ref* GetUnsafeAddress()
            => (godot_ref*)Unsafe.AsPointer(ref Unsafe.AsRef(in _reference));

        private IntPtr _reference;

        /// <summary>
        /// Disposes of this <see cref="godot_ref"/>.
        /// </summary>
        public void Dispose()
        {
            if (_reference == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_ref_destroy(ref this);
            _reference = IntPtr.Zero;
        }

        /// <summary>
        /// The pointer for this <see cref="godot_ref"/>.
        /// </summary>
        public readonly IntPtr Reference
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _reference;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_ref"/> is null.
        /// </summary>
        public readonly bool IsNull
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _reference == IntPtr.Zero;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent for the type of error that will be called.
    /// </summary>
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public enum godot_variant_call_error_error
    {
        /// <summary>
        /// No error.
        /// </summary>
        GODOT_CALL_ERROR_CALL_OK = 0,
        /// <summary>
        /// Method passed was invalid.
        /// </summary>
        GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD,
        /// <summary>
        /// Argument passed was invalid.
        /// </summary>
        GODOT_CALL_ERROR_CALL_ERROR_INVALID_ARGUMENT,
        /// <summary>
        /// Too many arguments were passed.
        /// </summary>
        GODOT_CALL_ERROR_CALL_ERROR_TOO_MANY_ARGUMENTS,
        /// <summary>
        /// Too few arguments were passed.
        /// </summary>
        GODOT_CALL_ERROR_CALL_ERROR_TOO_FEW_ARGUMENTS,
        /// <summary>
        /// The instance was null.
        /// </summary>
        GODOT_CALL_ERROR_CALL_ERROR_INSTANCE_IS_NULL,
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent for calling an error.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_variant_call_error
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_variant_call_error* GetUnsafeAddress()
            => (godot_variant_call_error*)Unsafe.AsPointer(ref Unsafe.AsRef(in error));

        private godot_variant_call_error_error error;
        private int argument;
        private int expected;

        /// <summary>
        /// The error state of this call.
        /// </summary>
        public godot_variant_call_error_error Error
        {
            readonly get => error;
            set => error = value;
        }

        /// <summary>
        /// The argument for this call.
        /// </summary>
        public int Argument
        {
            readonly get => argument;
            set => argument = value;
        }

        /// <summary>
        /// The expected type for this call.
        /// </summary>
        public Godot.Variant.Type Expected
        {
            readonly get => (Godot.Variant.Type)expected;
            set => expected = (int)value;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Variant"/>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_variant
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_variant* GetUnsafeAddress()
            => (godot_variant*)Unsafe.AsPointer(ref Unsafe.AsRef(in _typeField));

        // Variant.Type is generated as an enum of type long, so we can't use for the field as it must only take 32-bits.
        private int _typeField;

        // There's padding here

        private godot_variant_data _data;

        [StructLayout(LayoutKind.Explicit)]
        // ReSharper disable once InconsistentNaming
        private unsafe ref struct godot_variant_data
        {
            [FieldOffset(0)] public godot_bool _bool;
            [FieldOffset(0)] public long _int;
            [FieldOffset(0)] public double _float;
            [FieldOffset(0)] public Transform2D* _transform2d;
            [FieldOffset(0)] public Aabb* _aabb;
            [FieldOffset(0)] public Basis* _basis;
            [FieldOffset(0)] public Transform3D* _transform3d;
            [FieldOffset(0)] public Projection* _projection;
            [FieldOffset(0)] private godot_variant_data_mem _mem;

            // The following fields are not in the C++ union, but this is how they're stored in _mem.
            [FieldOffset(0)] public godot_string_name _m_string_name;
            [FieldOffset(0)] public godot_string _m_string;
            [FieldOffset(0)] public Vector4 _m_vector4;
            [FieldOffset(0)] public Vector4I _m_vector4i;
            [FieldOffset(0)] public Vector3 _m_vector3;
            [FieldOffset(0)] public Vector3I _m_vector3i;
            [FieldOffset(0)] public Vector2 _m_vector2;
            [FieldOffset(0)] public Vector2I _m_vector2i;
            [FieldOffset(0)] public Rect2 _m_rect2;
            [FieldOffset(0)] public Rect2I _m_rect2i;
            [FieldOffset(0)] public Plane _m_plane;
            [FieldOffset(0)] public Quaternion _m_quaternion;
            [FieldOffset(0)] public Color _m_color;
            [FieldOffset(0)] public godot_node_path _m_node_path;
            [FieldOffset(0)] public Rid _m_rid;
            [FieldOffset(0)] public godot_variant_obj_data _m_obj_data;
            [FieldOffset(0)] public godot_callable _m_callable;
            [FieldOffset(0)] public godot_signal _m_signal;
            [FieldOffset(0)] public godot_dictionary _m_dictionary;
            [FieldOffset(0)] public godot_array _m_array;

            [StructLayout(LayoutKind.Sequential)]
            // ReSharper disable once InconsistentNaming
            public struct godot_variant_obj_data
            {
                public ulong id;
                public IntPtr obj;
            }

            [StructLayout(LayoutKind.Sequential)]
            // ReSharper disable once InconsistentNaming
            public struct godot_variant_data_mem
            {
#pragma warning disable 169
                private real_t _mem0;
                private real_t _mem1;
                private real_t _mem2;
                private real_t _mem3;
#pragma warning restore 169
            }
        }

        /// <summary>
        /// Represents the internal <see cref="Variant.Type"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Variant.Type Type
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => (Variant.Type)_typeField;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _typeField = (int)value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_bool"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_bool Bool
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._bool;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._bool = value;
        }

        /// <summary>
        /// Represents the internal <see langword="long"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public long Int
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._int;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._int = value;
        }

        /// <summary>
        /// Represents the internal <see langword="double"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public double Float
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._float;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._float = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Transform2D"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public readonly unsafe Transform2D* Transform2D
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data._transform2d;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Aabb"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public readonly unsafe Aabb* Aabb
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data._aabb;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Basis"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public readonly unsafe Basis* Basis
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data._basis;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Transform3D"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public readonly unsafe Transform3D* Transform3D
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data._transform3d;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Projection"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public readonly unsafe Projection* Projection
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data._projection;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_string_name"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_string_name StringName
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_string_name;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_string_name = value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_string"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_string String
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_string;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_string = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Vector4"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Vector4 Vector4
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_vector4;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_vector4 = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Vector4I"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Vector4I Vector4I
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_vector4i;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_vector4i = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Vector3"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Vector3 Vector3
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_vector3;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_vector3 = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Vector3I"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Vector3I Vector3I
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_vector3i;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_vector3i = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Vector2"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Vector2 Vector2
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_vector2;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_vector2 = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Vector2I"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Vector2I Vector2I
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_vector2i;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_vector2i = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Rect2"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Rect2 Rect2
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_rect2;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_rect2 = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Rect2I"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Rect2I Rect2I
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_rect2i;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_rect2i = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Plane"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Plane Plane
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_plane;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_plane = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Quaternion"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Quaternion Quaternion
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_quaternion;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_quaternion = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Color"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Color Color
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_color;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_color = value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_node_path"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_node_path NodePath
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_node_path;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_node_path = value;
        }

        /// <summary>
        /// Represents the internal <see cref="Godot.Rid"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public Rid Rid
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_rid;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_rid = value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_callable"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_callable Callable
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_callable;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_callable = value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_signal"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_signal Signal
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_signal;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_signal = value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_dictionary"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_dictionary Dictionary
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_dictionary;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_dictionary = value;
        }

        /// <summary>
        /// Represents the internal <see cref="godot_array"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public godot_array Array
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            readonly get => _data._m_array;
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set => _data._m_array = value;
        }

        /// <summary>
        /// Represents the internal <see cref="IntPtr"/> of this <see cref="godot_variant"/>.
        /// </summary>
        public readonly IntPtr Object
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data._m_obj_data.obj;
        }

        /// <summary>
        /// Disposes of this <see cref="godot_variant"/>.
        /// </summary>
        public void Dispose()
        {
            switch (Type)
            {
                case Variant.Type.Nil:
                case Variant.Type.Bool:
                case Variant.Type.Int:
                case Variant.Type.Float:
                case Variant.Type.Vector2:
                case Variant.Type.Vector2I:
                case Variant.Type.Rect2:
                case Variant.Type.Rect2I:
                case Variant.Type.Vector3:
                case Variant.Type.Vector3I:
                case Variant.Type.Vector4:
                case Variant.Type.Vector4I:
                case Variant.Type.Plane:
                case Variant.Type.Quaternion:
                case Variant.Type.Color:
                case Variant.Type.Rid:
                    return;
            }

            NativeFuncs.godotsharp_variant_destroy(ref this);
            Type = Variant.Type.Nil;
        }

        [StructLayout(LayoutKind.Explicit)]
        // ReSharper disable once InconsistentNaming
        internal struct movable
        {
            // Variant.Type is generated as an enum of type long, so we can't use for the field as it must only take 32-bits.
            [FieldOffset(0)] private int _typeField;

            // There's padding here

            [FieldOffset(8)] private godot_variant_data.godot_variant_data_mem _data;

            public static unsafe explicit operator movable(in godot_variant value)
                => *(movable*)CustomUnsafe.AsPointer(ref CustomUnsafe.AsRef(value));

            public static unsafe explicit operator godot_variant(movable value)
                => *(godot_variant*)Unsafe.AsPointer(ref value);

            public unsafe ref godot_variant DangerousSelfRef =>
                ref CustomUnsafe.AsRef((godot_variant*)Unsafe.AsPointer(ref this));
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="string"/>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_string
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_string* GetUnsafeAddress()
            => (godot_string*)Unsafe.AsPointer(ref Unsafe.AsRef(in _ptr));

        private IntPtr _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_string"/>.
        /// </summary>
        public void Dispose()
        {
            if (_ptr == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_string_destroy(ref this);
            _ptr = IntPtr.Zero;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_string"/>.
        /// </summary>
        public readonly IntPtr Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        // Size including the null termination character
        /// <summary>
        /// Evaluates the size of this <see cref="godot_string"/>, including the null ternimation character.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != IntPtr.Zero ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="StringName"/>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_string_name
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_string_name* GetUnsafeAddress()
            => (godot_string_name*)Unsafe.AsPointer(ref Unsafe.AsRef(in _data));

        private IntPtr _data;

        /// <summary>
        /// Disposes of this <see cref="godot_string_name"/>.
        /// </summary>
        public void Dispose()
        {
            if (_data == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_string_name_destroy(ref this);
            _data = IntPtr.Zero;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_string_name"/> has been allocated.
        /// </summary>
        public readonly bool IsAllocated
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data != IntPtr.Zero;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_string_name"/> is empty.
        /// </summary>
        public readonly bool IsEmpty
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            // This is all that's needed to check if it's empty. Equivalent to `== StringName()` in C++.
            get => _data == IntPtr.Zero;
        }

        /// <summary>
        /// Evaluates if the <see cref="godot_string_name"/> instances are exactly equal.
        /// </summary>
        /// <param name="left">The left <see cref="godot_string_name"/>.</param>
        /// <param name="right">The right <see cref="godot_string_name"/>.</param>
        /// <returns><see langword="true"/> if these <see cref="godot_string_name"/> are
        /// exactly equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator ==(godot_string_name left, godot_string_name right)
        {
            return left._data == right._data;
        }

        /// <summary>
        /// Evaluates if the <see cref="godot_string_name"/> instances are not equal.
        /// </summary>
        /// <param name="left">The left <see cref="godot_string_name"/>.</param>
        /// <param name="right">The right <see cref="godot_string_name"/>.</param>
        /// <returns><see langword="true"/> if these <see cref="godot_string_name"/> are
        /// not equal; otherwise, <see langword="false"/>.</returns>
        public static bool operator !=(godot_string_name left, godot_string_name right)
        {
            return !(left == right);
        }

        /// <summary>
        /// Evaluates if the <see cref="godot_string_name"/> instances are exactly equal.
        /// </summary>
        /// <param name="other">The <see cref="godot_string_name"/> to compare with.</param>
        /// <returns><see langword="true"/> if these <see cref="godot_string_name"/> are
        /// exactly equal; otherwise, <see langword="false"/>.</returns>
        public bool Equals(godot_string_name other)
        {
            return _data == other._data;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_string_name"/> is exactly equal to the
        /// given object (<paramref name="obj"/>).
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns><see langword="true"/> if this <see cref="godot_string_name"/>and
        /// the object are equal; otherwise, <see langword="false"/>.</returns>
        public override bool Equals(object obj)
        {
            return obj is StringName s && s.Equals(this);
        }

        /// <summary>
        /// Serves as the hash function for <see cref="godot_string_name"/>.
        /// </summary>
        /// <returns>A hash code for this <see cref="godot_string_name"/>.</returns>
        public override int GetHashCode()
        {
            return _data.GetHashCode();
        }

        [StructLayout(LayoutKind.Sequential)]
        // ReSharper disable once InconsistentNaming
        internal struct movable
        {
            private IntPtr _data;

            public static unsafe explicit operator movable(in godot_string_name value)
                => *(movable*)CustomUnsafe.AsPointer(ref CustomUnsafe.AsRef(value));

            public static unsafe explicit operator godot_string_name(movable value)
                => *(godot_string_name*)Unsafe.AsPointer(ref value);

            public unsafe ref godot_string_name DangerousSelfRef =>
                ref CustomUnsafe.AsRef((godot_string_name*)Unsafe.AsPointer(ref this));
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="NodePath"/>.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_node_path
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_node_path* GetUnsafeAddress()
            => (godot_node_path*)Unsafe.AsPointer(ref Unsafe.AsRef(in _data));

        private IntPtr _data;

        /// <summary>
        /// Disposes of this <see cref="godot_node_path"/>.
        /// </summary>
        public void Dispose()
        {
            if (_data == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_node_path_destroy(ref this);
            _data = IntPtr.Zero;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_node_path"/> has been allocated.
        /// </summary>
        public readonly bool IsAllocated
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _data != IntPtr.Zero;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_node_path"/> is empty.
        /// </summary>
        public readonly bool IsEmpty
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            // This is all that's needed to check if it's empty. It's what the `is_empty()` C++ method does.
            get => _data == IntPtr.Zero;
        }

        [StructLayout(LayoutKind.Sequential)]
        // ReSharper disable once InconsistentNaming
        internal struct movable
        {
            private IntPtr _data;

            public static unsafe explicit operator movable(in godot_node_path value)
                => *(movable*)CustomUnsafe.AsPointer(ref CustomUnsafe.AsRef(value));

            public static unsafe explicit operator godot_node_path(movable value)
                => *(godot_node_path*)Unsafe.AsPointer(ref value);

            public unsafe ref godot_node_path DangerousSelfRef =>
                ref CustomUnsafe.AsRef((godot_node_path*)Unsafe.AsPointer(ref this));
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Signal"/>.
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_signal
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_signal* GetUnsafeAddress()
            => (godot_signal*)Unsafe.AsPointer(ref Unsafe.AsRef(in _getUnsafeAddressHelper));

        [FieldOffset(0)] private byte _getUnsafeAddressHelper;

        [FieldOffset(0)] private godot_string_name _name;

        // There's padding here on 32-bit

        [FieldOffset(8)] private ulong _objectId;

        /// <summary>
        /// Constructs a new <see cref="godot_signal"/> with the given <paramref name="name"/>
        /// and owner <paramref name="objectId"/>.
        /// </summary>
        /// <param name="name">The name of this signal.</param>
        /// <param name="objectId">The objectId of this signal's owner.</param>
        public godot_signal(godot_string_name name, ulong objectId) : this()
        {
            _name = name;
            _objectId = objectId;
        }

        /// <summary>
        /// Retrieves the name of this <see cref="godot_signal"/>.
        /// </summary>
        public godot_string_name Name
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _name;
        }

        /// <summary>
        /// Retrieves the ObjectId belonging to the owner of this <see cref="godot_signal"/>.
        /// </summary>
        public ulong ObjectId
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _objectId;
        }

        /// <summary>
        /// Disposes of this <see cref="godot_signal"/>.
        /// </summary>
        public void Dispose()
        {
            if (!_name.IsAllocated)
                return;
            NativeFuncs.godotsharp_signal_destroy(ref this);
            _name = default;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Callable"/>.
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_callable
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_callable* GetUnsafeAddress()
            => (godot_callable*)Unsafe.AsPointer(ref Unsafe.AsRef(in _getUnsafeAddressHelper));

        [FieldOffset(0)] private byte _getUnsafeAddressHelper;

        [FieldOffset(0)] private godot_string_name _method;

        // There's padding here on 32-bit

        // ReSharper disable once PrivateFieldCanBeConvertedToLocalVariable
        [FieldOffset(8)] private ulong _objectId;
        [FieldOffset(8)] private IntPtr _custom;

        /// <summary>
        /// Constructs a new <see cref="godot_signal"/> with the given <paramref name="method"/>
        /// and owner <paramref name="objectId"/>.
        /// </summary>
        /// <param name="method"></param>
        /// <param name="objectId"></param>
        public godot_callable(godot_string_name method, ulong objectId) : this()
        {
            _method = method;
            _objectId = objectId;
        }

        /// <summary>
        /// Disposes of this <see cref="godot_callable"/>.
        /// </summary>
        public void Dispose()
        {
            // _custom needs freeing as well
            if (!_method.IsAllocated && _custom == IntPtr.Zero)
                return;
            NativeFuncs.godotsharp_callable_destroy(ref this);
            _method = default;
            _custom = IntPtr.Zero;
        }
    }

    // A correctly constructed value needs to call the native default constructor to allocate `_p`.
    // Don't pass a C# default constructed `godot_array` to native code, unless it's going to
    // be re-assigned a new value (the copy constructor checks if `_p` is null so that's fine).
    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of an <see cref="Collections.Array"/>.
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_array* GetUnsafeAddress()
            => (godot_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _getUnsafeAddressHelper));

        [FieldOffset(0)] private byte _getUnsafeAddressHelper;

        [FieldOffset(0)] private unsafe ArrayPrivate* _p;

        [StructLayout(LayoutKind.Sequential)]
        private struct ArrayPrivate
        {
            private uint _safeRefCount;

            public VariantVector _arrayVector;

            private unsafe godot_variant* _readOnly;

            // There are more fields here, but we don't care as we never store this in C#

            public readonly int Size
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => _arrayVector.Size;
            }

            public readonly unsafe bool IsReadOnly
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => _readOnly != null;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct VariantVector
        {
            private IntPtr _writeProxy;
            public unsafe godot_variant* _ptr;

            public readonly unsafe int Size
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => _ptr != null ? *((int*)_ptr - 1) : 0;
            }
        }

        /// <summary>
        /// Evaluates the <see cref="godot_variant"/> elements within
        /// this <see cref="godot_array"/>.
        /// </summary>
        public readonly unsafe godot_variant* Elements
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _p->_arrayVector._ptr;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_array"/> has been allocated.
        /// </summary>
        public readonly unsafe bool IsAllocated
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _p != null;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_array"/> is empty.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _p != null ? _p->Size : 0;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_array"/> is in a read-only state.
        /// </summary>
        public readonly unsafe bool IsReadOnly
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _p != null && _p->IsReadOnly;
        }

        /// <summary>
        /// Disposes of this <see cref="godot_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_p == null)
                return;
            NativeFuncs.godotsharp_array_destroy(ref this);
            _p = null;
        }

        [StructLayout(LayoutKind.Sequential)]
        // ReSharper disable once InconsistentNaming
        internal struct movable
        {
            private unsafe ArrayPrivate* _p;

            public static unsafe explicit operator movable(in godot_array value)
                => *(movable*)CustomUnsafe.AsPointer(ref CustomUnsafe.AsRef(value));

            public static unsafe explicit operator godot_array(movable value)
                => *(godot_array*)Unsafe.AsPointer(ref value);

            public unsafe ref godot_array DangerousSelfRef =>
                ref CustomUnsafe.AsRef((godot_array*)Unsafe.AsPointer(ref this));
        }
    }

    // IMPORTANT:
    // A correctly constructed value needs to call the native default constructor to allocate `_p`.
    // Don't pass a C# default constructed `godot_dictionary` to native code, unless it's going to
    // be re-assigned a new value (the copy constructor checks if `_p` is null so that's fine).
    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Collections.Dictionary"/>.
    /// </summary>
    [StructLayout(LayoutKind.Explicit)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_dictionary
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_dictionary* GetUnsafeAddress()
            => (godot_dictionary*)Unsafe.AsPointer(ref Unsafe.AsRef(in _getUnsafeAddressHelper));

        [FieldOffset(0)] private byte _getUnsafeAddressHelper;

        [FieldOffset(0)] private unsafe DictionaryPrivate* _p;

        [StructLayout(LayoutKind.Sequential)]
        private struct DictionaryPrivate
        {
            private uint _safeRefCount;

            private unsafe godot_variant* _readOnly;

            // There are more fields here, but we don't care as we never store this in C#

            public readonly unsafe bool IsReadOnly
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => _readOnly != null;
            }
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_dictionary"/> has been allocated.
        /// </summary>
        public readonly unsafe bool IsAllocated
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _p != null;
        }

        /// <summary>
        /// Evaluates if this <see cref="godot_dictionary"/> is in a read-only state.
        /// </summary>
        public readonly unsafe bool IsReadOnly
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _p != null && _p->IsReadOnly;
        }

        /// <summary>
        /// Disposes of this <see cref="godot_dictionary"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_p == null)
                return;
            NativeFuncs.godotsharp_dictionary_destroy(ref this);
            _p = null;
        }

        [StructLayout(LayoutKind.Sequential)]
        // ReSharper disable once InconsistentNaming
        internal struct movable
        {
            private unsafe DictionaryPrivate* _p;

            public static unsafe explicit operator movable(in godot_dictionary value)
                => *(movable*)CustomUnsafe.AsPointer(ref CustomUnsafe.AsRef(value));

            public static unsafe explicit operator godot_dictionary(movable value)
                => *(godot_dictionary*)Unsafe.AsPointer(ref value);

            public unsafe ref godot_dictionary DangerousSelfRef =>
                ref CustomUnsafe.AsRef((godot_dictionary*)Unsafe.AsPointer(ref this));
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="byte"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_byte_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_byte_array* GetUnsafeAddress()
            => (godot_packed_byte_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe byte* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_byte_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_byte_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_byte_array"/>.
        /// </summary>
        public readonly unsafe byte* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_byte_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of an <see langword="int"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_int32_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_int32_array* GetUnsafeAddress()
            => (godot_packed_int32_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe int* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_int32_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_int32_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_int32_array"/>.
        /// </summary>
        public readonly unsafe int* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_int32_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *(_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="long"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_int64_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_int64_array* GetUnsafeAddress()
            => (godot_packed_int64_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe long* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_int64_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_int64_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_int64_array"/>.
        /// </summary>
        public readonly unsafe long* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_int64_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="float"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_float32_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_float32_array* GetUnsafeAddress()
            => (godot_packed_float32_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe float* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_float32_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_float32_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_float32_array"/>.
        /// </summary>
        public readonly unsafe float* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_float32_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="double"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_float64_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_float64_array* GetUnsafeAddress()
            => (godot_packed_float64_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe double* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_float64_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_float64_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_float64_array"/>.
        /// </summary>
        public readonly unsafe double* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_float64_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see langword="string"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_string_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_string_array* GetUnsafeAddress()
            => (godot_packed_string_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe godot_string* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_string_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_string_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_string_array"/>.
        /// </summary>
        public readonly unsafe godot_string* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_string_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Vector2"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_vector2_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_vector2_array* GetUnsafeAddress()
            => (godot_packed_vector2_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe Vector2* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_vector2_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_vector2_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_vector2_array"/>.
        /// </summary>
        public readonly unsafe Vector2* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_vector2_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Vector3"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_vector3_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_vector3_array* GetUnsafeAddress()
            => (godot_packed_vector3_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe Vector3* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_vector3_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_vector3_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_vector3_array"/>.
        /// </summary>
        public readonly unsafe Vector3* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_vector3_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }

    /// <summary>
    /// Represents the <see cref="NativeInterop"/> equivalent of a <see cref="Color"/>[].
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    // ReSharper disable once InconsistentNaming
    public ref struct godot_packed_color_array
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal readonly unsafe godot_packed_color_array* GetUnsafeAddress()
            => (godot_packed_color_array*)Unsafe.AsPointer(ref Unsafe.AsRef(in _writeProxy));

        private IntPtr _writeProxy;
        private unsafe Color* _ptr;

        /// <summary>
        /// Disposes of this <see cref="godot_packed_color_array"/>.
        /// </summary>
        public unsafe void Dispose()
        {
            if (_ptr == null)
                return;
            NativeFuncs.godotsharp_packed_color_array_destroy(ref this);
            _ptr = null;
        }

        /// <summary>
        /// Evaluates the buffer of this <see cref="godot_packed_color_array"/>.
        /// </summary>
        public readonly unsafe Color* Buffer
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr;
        }

        /// <summary>
        /// Evaluates the size of this <see cref="godot_packed_color_array"/>.
        /// </summary>
        public readonly unsafe int Size
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => _ptr != null ? *((int*)_ptr - 1) : 0;
        }
    }
}
