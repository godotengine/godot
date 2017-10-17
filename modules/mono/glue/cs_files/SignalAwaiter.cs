using System;

namespace Godot
{
    public class SignalAwaiter : IAwaiter<object[]>, IAwaitable<object[]>
    {
        private bool completed = false;
        private object[] result = null;
        private Action action = null;

        public SignalAwaiter(Godot.Object source, string signal, Godot.Object target)
        {
            NativeCalls.godot_icall_Object_connect_signal_awaiter(
                Godot.Object.GetPtr(source),
                signal, Godot.Object.GetPtr(target), this
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
