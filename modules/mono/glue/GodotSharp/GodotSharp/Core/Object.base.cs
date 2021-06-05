using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object : IDisposable
    {
        private bool _disposed = false;

        private const string nativeName = "Object";

        internal IntPtr ptr;
        internal bool memoryOwn;

        public Object() : this(false)
        {
            if (ptr == IntPtr.Zero)
                ptr = godot_icall_Object_Ctor(this);
        }

        internal Object(bool memoryOwn) => this.memoryOwn = memoryOwn;

        public IntPtr NativeInstance => ptr;

        internal static IntPtr GetPtr(Object instance)
        {
            if (instance == null)
                return IntPtr.Zero;

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.ptr;
        }

        ~Object() => Dispose(false);

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
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

                ptr = IntPtr.Zero;
            }

            _disposed = true;
        }

        public override string ToString() => godot_icall_Object_ToString(GetPtr(this));

        /// <summary>
        /// Returns a new <see cref="Godot.SignalAwaiter"/> awaiter configured to complete when the instance
        /// <paramref name="source"/> emits the signal specified by the <paramref name="signal"/> parameter.
        /// </summary>
        /// <param name="source">
        /// The instance the awaiter will be listening to.
        /// </param>
        /// <param name="signal">
        /// The signal the awaiter will be waiting for.
        /// </param>
        /// <example>
        /// This sample prints a message once every frame up to 100 times.
        /// <code>
        /// public override void _Ready()
        /// {
        ///     for (int i = 0; i &lt; 100; i++)
        ///     {
        ///         await ToSignal(GetTree(), "idle_frame");
        ///         GD.Print($"Frame {i}");
        ///     }
        /// }
        /// </code>
        /// </example>
        public SignalAwaiter ToSignal(Object source, string signal) => new SignalAwaiter(source, signal, this);

        /// <summary>
        /// Gets a new <see cref="Godot.DynamicGodotObject"/> associated with this instance.
        /// </summary>
        public dynamic DynamicObject => new DynamicGodotObject(this);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Object_Ctor(Object obj);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Object_Disposed(Object obj, IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Reference_Disposed(Object obj, IntPtr ptr, bool isFinalizer);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Object_ToString(IntPtr ptr);

        // Used by the generated API
        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Object_ClassDB_get_method(string type, string method);
    }
}
