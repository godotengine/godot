using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class NodePath : IDisposable
    {
        private bool disposed = false;

        internal IntPtr ptr;

        internal static IntPtr GetPtr(NodePath instance)
        {
            return instance == null ? IntPtr.Zero : instance.ptr;
        }

        ~NodePath()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (ptr != IntPtr.Zero)
            {
                godot_icall_NodePath_Dtor(ptr);
                ptr = IntPtr.Zero;
            }

            disposed = true;
        }

        internal NodePath(IntPtr ptr)
        {
            this.ptr = ptr;
        }

        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        public NodePath() : this(string.Empty) {}

        public NodePath(string path)
        {
            this.ptr = godot_icall_NodePath_Ctor(path);
        }

        public static implicit operator NodePath(string from)
        {
            return new NodePath(from);
        }

        public static implicit operator string(NodePath from)
        {
            return godot_icall_NodePath_operator_String(NodePath.GetPtr(from));
        }

        public override string ToString()
        {
            return (string)this;
        }

        public NodePath GetAsPropertyPath()
        {
            return new NodePath(godot_icall_NodePath_get_as_property_path(NodePath.GetPtr(this)));
        }

        public string GetConcatenatedSubnames()
        {
            return godot_icall_NodePath_get_concatenated_subnames(NodePath.GetPtr(this));
        }

        public string GetName(int idx)
        {
            return godot_icall_NodePath_get_name(NodePath.GetPtr(this), idx);
        }

        public int GetNameCount()
        {
            return godot_icall_NodePath_get_name_count(NodePath.GetPtr(this));
        }

        public string GetSubname(int idx)
        {
            return godot_icall_NodePath_get_subname(NodePath.GetPtr(this), idx);
        }

        public int GetSubnameCount()
        {
            return godot_icall_NodePath_get_subname_count(NodePath.GetPtr(this));
        }

        public bool IsAbsolute()
        {
            return godot_icall_NodePath_is_absolute(NodePath.GetPtr(this));
        }

        public bool IsEmpty()
        {
            return godot_icall_NodePath_is_empty(NodePath.GetPtr(this));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_NodePath_Ctor(string path);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_NodePath_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_NodePath_operator_String(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_NodePath_get_as_property_path(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_NodePath_get_concatenated_subnames(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_NodePath_get_name(IntPtr ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_NodePath_get_name_count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_NodePath_get_subname(IntPtr ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_NodePath_get_subname_count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_NodePath_is_absolute(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_NodePath_is_empty(IntPtr ptr);
    }
}
