using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<object[]>, IAwaitable<object[]>
    {
        private bool _completed;
        private object[] _result;
        private Action _action;

        public SignalAwaiter(Object source, StringName signal, Object target)
        {
            godot_icall_SignalAwaiter_connect(Object.GetPtr(source), StringName.GetPtr(signal), Object.GetPtr(target), this);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_SignalAwaiter_connect(IntPtr source, IntPtr signal, IntPtr target, SignalAwaiter awaiter);

        public bool IsCompleted
        {
            get
            {
                return _completed;
            }
        }

        public void OnCompleted(Action action)
        {
            this._action = action;
        }

        public object[] GetResult()
        {
            return _result;
        }

        public IAwaiter<object[]> GetAwaiter()
        {
            return this;
        }

        internal void SignalCallback(object[] args)
        {
            _completed = true;
            _result = args;
            _action?.Invoke();
        }
    }
}
