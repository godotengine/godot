using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// StringNames are immutable strings designed for general-purpose representation of unique names.
    /// StringName ensures that only one instance of a given name exists (so two StringNames with the
    /// same value are the same object).
    /// Comparing them is much faster than with regular strings, because only the pointers are compared,
    /// not the whole strings.
    /// </summary>
    public sealed partial class StringName : IDisposable
    {
        private IntPtr ptr;

        internal static IntPtr GetPtr(StringName instance)
        {
            if (instance == null)
                throw new NullReferenceException($"The instance of type {nameof(StringName)} is null.");

            if (instance.ptr == IntPtr.Zero)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~StringName()
        {
            Dispose(false);
        }

        /// <summary>
        /// Disposes of this <see cref="StringName"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (ptr != IntPtr.Zero)
            {
                godot_icall_StringName_Dtor(ptr);
                ptr = IntPtr.Zero;
            }
        }

        internal StringName(IntPtr ptr)
        {
            this.ptr = ptr;
        }

        /// <summary>
        /// Constructs an empty <see cref="StringName"/>.
        /// </summary>
        public StringName()
        {
            ptr = IntPtr.Zero;
        }

        /// <summary>
        /// Constructs a <see cref="StringName"/> from the given <paramref name="path"/> string.
        /// </summary>
        /// <param name="path">String to construct the <see cref="StringName"/> from.</param>
        public StringName(string path)
        {
            ptr = path == null ? IntPtr.Zero : godot_icall_StringName_Ctor(path);
        }

        /// <summary>
        /// Converts a string to a <see cref="StringName"/>.
        /// </summary>
        /// <param name="from">The string to convert.</param>
        public static implicit operator StringName(string from) => new StringName(from);

        /// <summary>
        /// Converts a <see cref="StringName"/> to a string.
        /// </summary>
        /// <param name="from">The <see cref="StringName"/> to convert.</param>
        public static implicit operator string(StringName from) => from.ToString();

        /// <summary>
        /// Converts this <see cref="StringName"/> to a string.
        /// </summary>
        /// <returns>A string representation of this <see cref="StringName"/>.</returns>
        public override string ToString()
        {
            return ptr == IntPtr.Zero ? string.Empty : godot_icall_StringName_operator_String(GetPtr(this));
        }

        /// <summary>
        /// Check whether this <see cref="StringName"/> is empty.
        /// </summary>
        /// <returns>If the <see cref="StringName"/> is empty.</returns>
        public bool IsEmpty()
        {
            return ptr == IntPtr.Zero || godot_icall_StringName_is_empty(GetPtr(this));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern IntPtr godot_icall_StringName_Ctor(string path);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void godot_icall_StringName_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_StringName_operator_String(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool godot_icall_StringName_is_empty(IntPtr ptr);
    }
}
