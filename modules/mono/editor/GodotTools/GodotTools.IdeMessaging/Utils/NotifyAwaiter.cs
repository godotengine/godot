// ReSharper disable ParameterHidesMember
// ReSharper disable UnusedMember.Global

using System;
using System.Runtime.CompilerServices;

namespace GodotTools.IdeMessaging.Utils
{
    public class NotifyAwaiter<T> : INotifyCompletion
    {
        private Action? continuation;
        private Exception? exception;
        private T? result;

        public bool IsCompleted { get; private set; }

        public T GetResult()
        {
            if (exception != null)
                throw exception;
            return result!;
        }

        public void OnCompleted(Action continuation)
        {
            if (this.continuation != null)
                throw new InvalidOperationException("This awaiter already has a continuation.");
            this.continuation = continuation;
        }

        public void SetResult(T result)
        {
            if (IsCompleted)
                throw new InvalidOperationException("This awaiter is already completed.");

            IsCompleted = true;
            this.result = result;

            continuation?.Invoke();
        }

        public void SetException(Exception exception)
        {
            if (IsCompleted)
                throw new InvalidOperationException("This awaiter is already completed.");

            IsCompleted = true;
            this.exception = exception;

            continuation?.Invoke();
        }

        public NotifyAwaiter<T> Reset()
        {
            continuation = null;
            exception = null;
            result = default(T);
            IsCompleted = false;
            return this;
        }

        public NotifyAwaiter<T> GetAwaiter()
        {
            return this;
        }
    }
}
