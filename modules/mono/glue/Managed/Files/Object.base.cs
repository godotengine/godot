using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object : IDisposable
    {
        private bool disposed = false;

        private const string nativeName = "Object";

        internal IntPtr ptr;
        internal bool memoryOwn;

        public Object() : this(false)
        {
            if (ptr == IntPtr.Zero)
                ptr = godot_icall_Object_Ctor(this);
        }

        internal Object(bool memoryOwn)
        {
            this.memoryOwn = memoryOwn;
        }

        public IntPtr NativeInstance
        {
            get { return ptr; }
        }

        internal static IntPtr GetPtr(Object instance)
        {
            return instance == null ? IntPtr.Zero : instance.ptr;
        }

        ~Object()
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
                if (memoryOwn)
                {
                    memoryOwn = false;
                    godot_icall_Reference_Disposed(this, ptr, !disposing);
                }
                else
                {
                    godot_icall_Object_Disposed(this, ptr);
                }

                this.ptr = IntPtr.Zero;
            }

            disposed = true;
        }

        public SignalAwaiter ToSignal(Object source, string signal)
        {
            return new SignalAwaiter(source, signal, this);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_Object_Ctor(Object obj);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Object_Disposed(Object obj, IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Reference_Disposed(Object obj, IntPtr ptr, bool isFinalizer);

        // Used by the generated API
        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_Object_ClassDB_get_method(string type, string method);
    }
}
