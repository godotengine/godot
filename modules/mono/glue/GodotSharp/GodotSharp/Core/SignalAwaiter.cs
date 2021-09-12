using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<object[]>, IAwaitable<object[]>
    {
        private bool _completed;
        private object[] _result;
        private Action _action;

        public SignalAwaiter(Object source, StringName signal, Object target)
        {
            godot_icall_SignalAwaiter_connect(Object.GetPtr(source), ref signal.NativeValue,
                Object.GetPtr(target), GCHandle.ToIntPtr(GCHandle.Alloc(this)));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_SignalAwaiter_connect(IntPtr source, ref godot_string_name signal,
            IntPtr target, IntPtr awaiterHandlePtr);

        public bool IsCompleted => _completed;

        public void OnCompleted(Action action)
        {
            this._action = action;
        }

        public object[] GetResult() => _result;

        public IAwaiter<object[]> GetAwaiter() => this;

        internal static unsafe void SignalCallback(IntPtr awaiterGCHandlePtr,
            godot_variant** args, int argCount,
            bool* r_awaiterIsNull)
        {
            var awaiter = (SignalAwaiter)GCHandle.FromIntPtr(awaiterGCHandlePtr).Target;

            if (awaiter == null)
            {
                *r_awaiterIsNull = true;
                return;
            }

            *r_awaiterIsNull = false;

            awaiter._completed = true;

            object[] signalArgs = new object[argCount];

            for (int i = 0; i < argCount; i++)
                signalArgs[i] = Marshaling.variant_to_mono_object(args[i]);

            awaiter._result = signalArgs;

            awaiter._action?.Invoke();
        }
    }
}
