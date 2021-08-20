using System;
using System.Runtime.CompilerServices;
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
            godot_icall_SignalAwaiter_connect(Object.GetPtr(source), ref signal.NativeValue, Object.GetPtr(target), this);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_SignalAwaiter_connect(IntPtr source, ref godot_string_name signal, IntPtr target, SignalAwaiter awaiter);

        public bool IsCompleted => _completed;

        public void OnCompleted(Action action)
        {
            this._action = action;
        }

        public object[] GetResult() => _result;

        public IAwaiter<object[]> GetAwaiter() => this;

        internal void SignalCallback(object[] args)
        {
            _completed = true;
            _result = args;

            if (_action != null)
            {
                _action();
            }
        }
    }
}
