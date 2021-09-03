using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public sealed partial class RID : IDisposable
    {
        private bool _disposed = false;

        internal IntPtr ptr;

        internal static IntPtr GetPtr(RID instance)
        {
            if (instance == null)
                throw new NullReferenceException($"The instance of type {nameof(RID)} is null.");

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~RID()
        {
            Dispose(false);
        }

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
                godot_icall_RID_Dtor(ptr);
                ptr = IntPtr.Zero;
            }

            _disposed = true;
        }

        internal RID(IntPtr ptr)
        {
            this.ptr = ptr;
        }

        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        internal RID()
        {
            this.ptr = IntPtr.Zero;
        }

        public RID(Object from)
        {
            this.ptr = godot_icall_RID_Ctor(Object.GetPtr(from));
        }

        public int GetId()
        {
            return godot_icall_RID_get_id(RID.GetPtr(this));
        }

        public override string ToString() => "[RID]";

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_RID_Ctor(IntPtr from);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_RID_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_RID_get_id(IntPtr ptr);
    }
}
