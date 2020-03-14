using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<object[]>, IAwaitable<object[]>
    {
        private bool completed;
        private object[] result;
        private Action action;

        public SignalAwaiter(Object source, StringName signal, Object target)
        {
            godot_icall_SignalAwaiter_connect(Object.GetPtr(source), StringName.GetPtr(signal), Object.GetPtr(target), this);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static Error godot_icall_SignalAwaiter_connect(IntPtr source, IntPtr signal, IntPtr target, SignalAwaiter awaiter);

        public bool IsCompleted
        {
            get
            {
                return completed;
            }
        }

        public void OnCompleted(Action action)
        {
            this.action = action;
        }

        public object[] GetResult()
        {
            return result;
        }

        public IAwaiter<object[]> GetAwaiter()
        {
            return this;
        }

        internal void SignalCallback(object[] args)
        {
            completed = true;
            result = args;

            if (action != null)
            {
                action();
            }
        }
    }
}
