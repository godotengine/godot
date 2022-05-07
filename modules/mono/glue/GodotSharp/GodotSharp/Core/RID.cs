using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    /// <summary>
    /// The RID type is used to access the unique integer ID of a resource.
    /// They are opaque, which means they do not grant access to the associated
    /// resource by themselves. They are used by and with the low-level Server
    /// classes such as <see cref="VisualServer"/>.
    /// </summary>
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

        /// <summary>
        /// Disposes of this <see cref="RID"/>.
        /// </summary>
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

        /// <summary>
        /// The pointer to the native instance of this <see cref="RID"/>.
        /// </summary>
        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        internal RID()
        {
            this.ptr = IntPtr.Zero;
        }

        /// <summary>
        /// Constructs a new <see cref="RID"/> for the given <see cref="Object"/> <paramref name="from"/>.
        /// </summary>
        public RID(Object from)
        {
            this.ptr = godot_icall_RID_Ctor(Object.GetPtr(from));
        }

        /// <summary>
        /// Returns the ID of the referenced resource.
        /// </summary>
        /// <returns>The ID of the referenced resource.</returns>
        public int GetId()
        {
            return godot_icall_RID_get_id(GetPtr(this));
        }

        /// <summary>
        /// Converts this <see cref="RID"/> to a string.
        /// </summary>
        /// <returns>A string representation of this RID.</returns>
        public override string ToString() => "[RID]";

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_RID_Ctor(IntPtr from);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_RID_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_RID_get_id(IntPtr ptr);
    }
}
