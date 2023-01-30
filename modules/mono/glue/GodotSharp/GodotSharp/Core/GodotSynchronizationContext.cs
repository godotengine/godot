using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace Godot
{
    public sealed class GodotSynchronizationContext : SynchronizationContext, IDisposable
    {
        private readonly BlockingCollection<KeyValuePair<SendOrPostCallback, object>> _queue = new();

        public override void Post(SendOrPostCallback d, object state)
        {
            _queue.Add(new KeyValuePair<SendOrPostCallback, object>(d, state));
        }

        /// <summary>
        /// Calls the Key method on each workItem object in the _queue to activate their callbacks.
        /// </summary>
        public void ExecutePendingContinuations()
        {
            while (_queue.TryTake(out var workItem))
            {
                workItem.Key(workItem.Value);
            }
        }

        public void Dispose()
        {
            _queue.Dispose();
        }
    }
}
