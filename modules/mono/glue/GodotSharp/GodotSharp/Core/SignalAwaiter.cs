using System;
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
            NativeFuncs.godotsharp_internal_signal_awaiter_connect(Object.GetPtr(source), ref signal.NativeValue,
                Object.GetPtr(target), GCHandle.ToIntPtr(GCHandle.Alloc(this)));
        }

        public bool IsCompleted => _completed;

        public void OnCompleted(Action action)
        {
            this._action = action;
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
                    *outAwaiterIsNull = true.ToGodotBool();
                    return;
                }

                *outAwaiterIsNull = false.ToGodotBool();

                awaiter._completed = true;

                object[] signalArgs = new object[argCount];

                for (int i = 0; i < argCount; i++)
                    signalArgs[i] = Marshaling.variant_to_mono_object(args[i]);

                awaiter._result = signalArgs;

                awaiter._action?.Invoke();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugPrintUnhandledException(e);
                *outAwaiterIsNull = false.ToGodotBool();
            }
        }
    }
}
