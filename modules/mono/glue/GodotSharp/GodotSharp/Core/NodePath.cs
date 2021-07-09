using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public sealed partial class NodePath : IDisposable
    {
        private bool _disposed = false;

        internal IntPtr ptr;

        internal static IntPtr GetPtr(NodePath instance)
        {
            if (instance == null)
                throw new NullReferenceException($"The instance of type {nameof(NodePath)} is null.");

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~NodePath() => Dispose(false);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (ptr != IntPtr.Zero)
            {
                godot_icall_NodePath_Dtor(ptr);
                ptr = IntPtr.Zero;
            }

            _disposed = true;
        }

        internal NodePath(IntPtr ptr) => this.ptr = ptr;

        public IntPtr NativeInstance => ptr;

        public NodePath() : this(string.Empty) { }

        public NodePath(string path) => ptr = godot_icall_NodePath_Ctor(path);

        public static implicit operator NodePath(string from) => new NodePath(from);

        public static implicit operator string(NodePath from) => godot_icall_NodePath_operator_String(GetPtr(from));

        public override string ToString() => (string)this;

        public NodePath GetAsPropertyPath() => new NodePath(godot_icall_NodePath_get_as_property_path(GetPtr(this)));

        public string GetConcatenatedSubnames() => godot_icall_NodePath_get_concatenated_subnames(GetPtr(this));

        public string GetName(int idx) => godot_icall_NodePath_get_name(GetPtr(this), idx);

        public int GetNameCount() => godot_icall_NodePath_get_name_count(GetPtr(this));

        public string GetSubname(int idx) => godot_icall_NodePath_get_subname(GetPtr(this), idx);

        public int GetSubnameCount() => godot_icall_NodePath_get_subname_count(GetPtr(this));

        public bool IsAbsolute() => godot_icall_NodePath_is_absolute(GetPtr(this));

        public bool IsEmpty() => godot_icall_NodePath_is_empty(GetPtr(this));

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_NodePath_Ctor(string path);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_NodePath_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_NodePath_operator_String(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_NodePath_get_as_property_path(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_NodePath_get_concatenated_subnames(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_NodePath_get_name(IntPtr ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_NodePath_get_name_count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_NodePath_get_subname(IntPtr ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_NodePath_get_subname_count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_NodePath_is_absolute(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_NodePath_is_empty(IntPtr ptr);
    }
}
