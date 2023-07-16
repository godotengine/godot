using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    /// <summary>
    /// Represents an awaiter for when an instance emits a specified <see cref="Signal"/>
    /// </summary>
    public class SignalAwaiter : IAwaiter<Variant[]>, IAwaitable<Variant[]>
    {
        private bool _completed;
        private Variant[] _result;
        private Action _continuation;

        /// <summary>
        /// Constructs a new <see cref="SignalAwaiter"/> that will complete when the instance
        /// <paramref name="source"/> emits the signal specified by the <paramref name="signal"/>
        /// parameter, thereby notifying <paramref name="target"/>.
        /// </summary>
        /// <param name="source">The instance this will listen to.</param>
        /// <param name="signal">The signal this will be waiting for.</param>
        /// <param name="target">The instance this will notify on completion.</param>
        public SignalAwaiter(GodotObject source, StringName signal, GodotObject target)
        {
            var awaiterGcHandle = CustomGCHandle.AllocStrong(this);
            using godot_string_name signalSrc = NativeFuncs.godotsharp_string_name_new_copy(
                (godot_string_name)(signal?.NativeValue ?? default));
            NativeFuncs.godotsharp_internal_signal_awaiter_connect(GodotObject.GetPtr(source), in signalSrc,
                GodotObject.GetPtr(target), GCHandle.ToIntPtr(awaiterGcHandle));
        }

        /// <summary>
        /// Returns the signal's completion status.
        /// </summary>
        public bool IsCompleted => _completed;

        /// <summary>
        /// Called upon completion.
        /// </summary>
        /// <param name="continuation">The action to perform upon completion.</param>
        public void OnCompleted(Action continuation)
        {
            _continuation = continuation;
        }

        /// <summary>
        /// Gets the result of the signal's completion.
        /// </summary>
        /// <returns>The result of completion.</returns>
        public Variant[] GetResult() => _result;

        /// <summary>
        /// Retrieve this signal awaiter.
        /// </summary>
        /// <returns>This.</returns>
        public IAwaiter<Variant[]> GetAwaiter() => this;

        [UnmanagedCallersOnly]
        internal static unsafe void SignalCallback(IntPtr awaiterGCHandlePtr, godot_variant** args, int argCount,
            godot_bool* outAwaiterIsNull)
        {
            try
            {
                var awaiter = (SignalAwaiter)GCHandle.FromIntPtr(awaiterGCHandlePtr).Target;

                if (awaiter == null)
                {
                    *outAwaiterIsNull = godot_bool.True;
                    return;
                }

                *outAwaiterIsNull = godot_bool.False;

                awaiter._completed = true;

                Variant[] signalArgs = new Variant[argCount];

                for (int i = 0; i < argCount; i++)
                    signalArgs[i] = Variant.CreateCopyingBorrowed(*args[i]);

                awaiter._result = signalArgs;

                awaiter._continuation?.Invoke();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outAwaiterIsNull = godot_bool.False;
            }
        }
    }
}
