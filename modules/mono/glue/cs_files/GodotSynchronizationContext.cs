using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading;

namespace Godot
{
	public class GodotSynchronizationContext : SynchronizationContext
	{
		private readonly BlockingCollection<KeyValuePair<SendOrPostCallback, object>> queue = new BlockingCollection<KeyValuePair<SendOrPostCallback, object>>();

		public override void Post(SendOrPostCallback d, object state)
		{
			queue.Add(new KeyValuePair<SendOrPostCallback, object>(d, state));
		}

		public void ExecutePendingContinuations()
		{
			KeyValuePair<SendOrPostCallback, object> workItem;
			while (queue.TryTake(out workItem))
			{
				workItem.Key(workItem.Value);
			}
		}
	}
}
