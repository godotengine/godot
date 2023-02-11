using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace Godot
{
    public sealed class GodotSynchronizationContext : SynchronizationContext, IDisposable
    {
        private readonly BlockingCollection<(SendOrPostCallback Callback, object State)> _queue = new();

        public override void Send(SendOrPostCallback d, object state)
        {
            // Shortcut if we're already on this context
            // Also necessary to avoid a deadlock, since Send is blocking
            if (Current == this)
            {
                d(state);
                return;
            }

            var source = new TaskCompletionSource();

            _queue.Add((st =>
            {
                try
                {
                    d(st);
                }
                finally
                {
                    source.SetResult();
                }
            }, state));

            source.Task.Wait();
        }

        public override void Post(SendOrPostCallback d, object state)
        {
            _queue.Add((d, state));
        }

        /// <summary>
        /// Calls the Key method on each workItem object in the _queue to activate their callbacks.
        /// </summary>
        public void ExecutePendingContinuations()
        {
            while (_queue.TryTake(out var workItem))
            {
                workItem.Callback(workItem.State);
            }
        }

        public void Dispose()
        {
            _queue.Dispose();
        }
    }
}
