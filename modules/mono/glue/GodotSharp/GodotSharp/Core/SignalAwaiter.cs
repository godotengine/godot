using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<Variant[]>, IAwaitable<Variant[]>
    {
        private bool _completed;
        private Variant[] _result;
        private Action _continuation;

        public SignalAwaiter(GodotObject source, StringName signal, GodotObject target)
        {
            var awaiterGcHandle = CustomGCHandle.AllocStrong(this);
            using godot_string_name signalSrc = NativeFuncs.godotsharp_string_name_new_copy(
                (godot_string_name)(signal?.NativeValue ?? default));
            NativeFuncs.godotsharp_internal_signal_awaiter_connect(GodotObject.GetPtr(source), in signalSrc,
                GodotObject.GetPtr(target), GCHandle.ToIntPtr(awaiterGcHandle));
        }

        public bool IsCompleted => _completed;

        public void OnCompleted(Action continuation)
        {
            _continuation = continuation;
        }

        public Variant[] GetResult() => _result;

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

                if (argCount > 0)
                {
                    Variant[] signalArgs = new Variant[argCount];

                    for (int i = 0; i < argCount; i++)
                        signalArgs[i] = Variant.CreateCopyingBorrowed(*args[i]);

                    awaiter._result = signalArgs;
                }
                else
                {
                    awaiter._result = [];
                }

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
