using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object : IDisposable
    {
        private bool _disposed = false;

        internal IntPtr NativePtr;
        internal bool MemoryOwn;

        public Object() : this(false)
        {
            if (NativePtr == IntPtr.Zero)
            {
#if NET
                unsafe
                {
                    ptr = NativeCtor();
                }
#else
                NativePtr = _gd__invoke_class_constructor(NativeCtor);
#endif
                NativeInterop.InteropUtils.TieManagedToUnmanaged(this, NativePtr);
            }

            _InitializeGodotScriptInstanceInternals();
        }

        internal void _InitializeGodotScriptInstanceInternals()
        {
            godot_icall_Object_ConnectEventSignals(NativePtr);
        }

        internal Object(bool memoryOwn)
        {
            this.MemoryOwn = memoryOwn;
        }

        public IntPtr NativeInstance => NativePtr;

        internal static IntPtr GetPtr(Object instance)
        {
            if (instance == null)
                return IntPtr.Zero;

            if (instance._disposed)
                throw new ObjectDisposedException(instance.GetType().FullName);

            return instance.NativePtr;
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
            if (_disposed)
                return;

            if (NativePtr != IntPtr.Zero)
            {
                if (MemoryOwn)
                {
                    MemoryOwn = false;
                    godot_icall_RefCounted_Disposed(this, NativePtr, !disposing);
                }
                else
                {
                    godot_icall_Object_Disposed(this, NativePtr);
                }

                this.NativePtr = IntPtr.Zero;
            }

            _disposed = true;
        }

        public override string ToString()
        {
            return godot_icall_Object_ToString(GetPtr(this));
        }

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
        public SignalAwaiter ToSignal(Object source, StringName signal)
        {
            return new SignalAwaiter(source, signal, this);
        }

        /// <summary>
        /// Gets a new <see cref="Godot.DynamicGodotObject"/> associated with this instance.
        /// </summary>
        public dynamic DynamicObject => new DynamicGodotObject(this);

        internal static unsafe IntPtr ClassDB_get_method(StringName type, string method)
        {
            IntPtr methodBind;
            fixed (char* methodChars = method)
            {
                methodBind = NativeInterop.NativeFuncs
                    .godotsharp_method_bind_get_method(ref type.NativeValue, methodChars);
            }

            if (methodBind == IntPtr.Zero)
                throw new NativeMethodBindNotFoundException(type + "." + method);

            return methodBind;
        }

#if NET
        internal static unsafe delegate* unmanaged<IntPtr> _gd__ClassDB_get_constructor(StringName type)
        {
            // for some reason the '??' operator doesn't support 'delegate*'
            var nativeConstructor = NativeInterop.NativeFuncs
                .godotsharp_get_class_constructor(ref type.NativeValue);

            if (nativeConstructor == null)
                throw new NativeConstructorNotFoundException(type);

            return nativeConstructor;
        }
#else
        internal static IntPtr ClassDB_get_constructor(StringName type)
        {
            // for some reason the '??' operator doesn't support 'delegate*'
            var nativeConstructor = NativeInterop.NativeFuncs
                .godotsharp_get_class_constructor(ref type.NativeValue);

            if (nativeConstructor == IntPtr.Zero)
                throw new NativeConstructorNotFoundException(type);

            return nativeConstructor;
        }

        internal static IntPtr _gd__invoke_class_constructor(IntPtr ctorFuncPtr)
            => NativeInterop.NativeFuncs.godotsharp_invoke_class_constructor(ctorFuncPtr);
#endif

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Object_Disposed(Object obj, IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_RefCounted_Disposed(Object obj, IntPtr ptr, bool isFinalizer);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Object_ConnectEventSignals(IntPtr obj);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Object_ToString(IntPtr ptr);
    }
}
