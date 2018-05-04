using System;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<object[]>, IAwaitable<object[]>
    {
        private bool completed;
        private object[] result;
        private Action action;

        public SignalAwaiter(Object source, string signal, Object target)
        {
            NativeCalls.godot_icall_Object_connect_signal_awaiter(
                Object.GetPtr(source),
                signal, Object.GetPtr(target), this
                );
        }

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

        internal void FailureCallback()
        {
            action = null;
            completed = true;
        }
    }
}
