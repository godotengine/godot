using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<object[]>, IAwaitable<object[]>
    {
        private bool _completed;
        private object[] _result;
        private Action _continuation;

        public SignalAwaiter(Object source, StringName signal, Object target)
        {
            var awaiterGcHandle = CustomGCHandle.AllocStrong(this);
            using godot_string_name signalSrc = NativeFuncs.godotsharp_string_name_new_copy(
                (godot_string_name)(signal?.NativeValue ?? default));
            NativeFuncs.godotsharp_internal_signal_awaiter_connect(Object.GetPtr(source), in signalSrc,
                Object.GetPtr(target), GCHandle.ToIntPtr(awaiterGcHandle));
        }

        public bool IsCompleted => _completed;

        public void OnCompleted(Action continuation)
        {
            _continuation = continuation;
        }

        public object[] GetResult() => _result;

        public IAwaiter<object[]> GetAwaiter() => this;

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

                object[] signalArgs = new object[argCount];

                for (int i = 0; i < argCount; i++)
                    signalArgs[i] = Marshaling.ConvertVariantToManagedObject(*args[i]);

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
