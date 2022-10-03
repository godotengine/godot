using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    /// <summary>
    /// Callable is a first class object which can be held in variables and passed to functions.
    /// It represents a given method in an Object, and is typically used for signal callbacks.
    /// </summary>
    /// <example>
    /// <code>
    /// public void PrintArgs(object ar1, object arg2, object arg3 = null)
    /// {
    ///     GD.PrintS(arg1, arg2, arg3);
    /// }
    ///
    /// public void Test()
    /// {
    ///     // This Callable object will call the PrintArgs method defined above.
    ///     Callable callable = new Callable(this, nameof(PrintArgs));
    ///     callable.Call("hello", "world"); // Prints "hello world null".
    ///     callable.Call(Vector2.Up, 42, callable); // Prints "(0, -1) 42 Node(Node.cs)::PrintArgs".
    ///     callable.Call("invalid"); // Invalid call, should have at least 2 arguments.
    /// }
    /// </code>
    /// </example>
    public readonly struct Callable
    {
        private readonly Object _target;
        private readonly StringName _method;
        private readonly Delegate _delegate;

        /// <summary>
        /// Object that contains the method.
        /// </summary>
        public Object Target => _target;

        /// <summary>
        /// Name of the method that will be called.
        /// </summary>
        public StringName Method => _method;

        /// <summary>
        /// Delegate of the method that will be called.
        /// </summary>
        public Delegate Delegate => _delegate;

        /// <summary>
        /// Converts a <see cref="Delegate"/> to a <see cref="Callable"/>.
        /// </summary>
        /// <param name="delegate">The delegate to convert.</param>
        public static implicit operator Callable(Delegate @delegate) => new Callable(@delegate);

        /// <summary>
        /// Constructs a new <see cref="Callable"/> for the method called <paramref name="method"/>
        /// in the specified <paramref name="target"/>.
        /// </summary>
        /// <param name="target">Object that contains the method.</param>
        /// <param name="method">Name of the method that will be called.</param>
        public Callable(Object target, StringName method)
        {
            _target = target;
            _method = method;
            _delegate = null;
        }

        /// <summary>
        /// Constructs a new <see cref="Callable"/> for the given <paramref name="delegate"/>.
        /// </summary>
        /// <param name="delegate">Delegate method that will be called.</param>
        public Callable(Delegate @delegate)
        {
            _target = @delegate?.Target as Object;
            _method = null;
            _delegate = @delegate;
        }

        private const int VarArgsSpanThreshold = 5;

        /// <summary>
        /// Calls the method represented by this <see cref="Callable"/>.
        /// Arguments can be passed and should match the method's signature.
        /// </summary>
        /// <param name="args">Arguments that will be passed to the method call.</param>
        /// <returns>The value returned by the method.</returns>
        public unsafe Variant Call(params Variant[] args)
        {
            using godot_callable callable = Marshaling.ConvertCallableToNative(this);

            int argc = args.Length;

            Span<godot_variant.movable> argsStoreSpan = argc <= VarArgsSpanThreshold ?
                stackalloc godot_variant.movable[VarArgsSpanThreshold].Cleared() :
                new godot_variant.movable[argc];

            Span<IntPtr> argsSpan = argc <= 10 ?
                stackalloc IntPtr[argc] :
                new IntPtr[argc];

            using var variantSpanDisposer = new VariantSpanDisposer(argsStoreSpan);

            fixed (godot_variant* varargs = &MemoryMarshal.GetReference(argsStoreSpan).DangerousSelfRef)
            fixed (IntPtr* argsPtr = &MemoryMarshal.GetReference(argsSpan))
            {
                for (int i = 0; i < argc; i++)
                {
                    varargs[i] = (godot_variant)args[i].NativeVar;
                    argsPtr[i] = new IntPtr(&varargs[i]);
                }

                godot_variant ret = NativeFuncs.godotsharp_callable_call(callable,
                    (godot_variant**)argsPtr, argc, out _);
                return Variant.CreateTakingOwnershipOfDisposableValue(ret);
            }
        }

        /// <summary>
        /// Calls the method represented by this <see cref="Callable"/> in deferred mode, i.e. during the idle frame.
        /// Arguments can be passed and should match the method's signature.
        /// </summary>
        /// <param name="args">Arguments that will be passed to the method call.</param>
        public unsafe void CallDeferred(params Variant[] args)
        {
            using godot_callable callable = Marshaling.ConvertCallableToNative(this);

            int argc = args.Length;

            Span<godot_variant.movable> argsStoreSpan = argc <= VarArgsSpanThreshold ?
                stackalloc godot_variant.movable[VarArgsSpanThreshold].Cleared() :
                new godot_variant.movable[argc];

            Span<IntPtr> argsSpan = argc <= 10 ?
                stackalloc IntPtr[argc] :
                new IntPtr[argc];

            using var variantSpanDisposer = new VariantSpanDisposer(argsStoreSpan);

            fixed (godot_variant* varargs = &MemoryMarshal.GetReference(argsStoreSpan).DangerousSelfRef)
            fixed (IntPtr* argsPtr = &MemoryMarshal.GetReference(argsSpan))
            {
                for (int i = 0; i < argc; i++)
                {
                    varargs[i] = (godot_variant)args[i].NativeVar;
                    argsPtr[i] = new IntPtr(&varargs[i]);
                }

                NativeFuncs.godotsharp_callable_call_deferred(callable, (godot_variant**)argsPtr, argc);
            }
        }
    }
}
