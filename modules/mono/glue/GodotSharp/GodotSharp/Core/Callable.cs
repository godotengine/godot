using System;
using System.Runtime.CompilerServices;

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
    public struct Callable
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
            _target = null;
            _method = null;
            _delegate = @delegate;
        }

        /// <summary>
        /// Calls the method represented by this <see cref="Callable"/>.
        /// Arguments can be passed and should match the method's signature.
        /// </summary>
        /// <param name="args">Arguments that will be passed to the method call.</param>
        /// <returns>The value returned by the method.</returns>
        public object Call(params object[] args)
        {
            return godot_icall_Callable_Call(ref this, args);
        }

        /// <summary>
        /// Calls the method represented by this <see cref="Callable"/> in deferred mode, i.e. during the idle frame.
        /// Arguments can be passed and should match the method's signature.
        /// </summary>
        /// <param name="args">Arguments that will be passed to the method call.</param>
        public void CallDeferred(params object[] args)
        {
            godot_icall_Callable_CallDeferred(ref this, args);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Callable_Call(ref Callable callable, object[] args);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Callable_CallDeferred(ref Callable callable, object[] args);
    }
}
