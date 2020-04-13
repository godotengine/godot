using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Godot
{
    public class GodotTaskScheduler : TaskScheduler
    {
        internal GodotSynchronizationContext Context { get; }
        private readonly LinkedList<Task> _tasks = new LinkedList<Task>();

        public GodotTaskScheduler()
        {
            Context = new GodotSynchronizationContext();
            SynchronizationContext.SetSynchronizationContext(Context);
        }

        protected sealed override void QueueTask(Task task)
        {
            lock (_tasks)
            {
                _tasks.AddLast(task);
            }
        }

        protected sealed override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
        {
            if (SynchronizationContext.Current != Context)
                return false;

            if (taskWasPreviouslyQueued)
                TryDequeue(task);

            return TryExecuteTask(task);
        }

        protected sealed override bool TryDequeue(Task task)
        {
            lock (_tasks)
            {
                return _tasks.Remove(task);
            }
        }

        protected sealed override IEnumerable<Task> GetScheduledTasks()
        {
            lock (_tasks)
            {
                foreach (Task task in _tasks)
                    yield return task;
            }
        }

        public void Activate()
        {
            ExecuteQueuedTasks();
            Context.ExecutePendingContinuations();
        }

        private void ExecuteQueuedTasks()
        {
            while (true)
            {
                Task task;

                lock (_tasks)
                {
                    if (_tasks.Any())
                    {
                        task = _tasks.First.Value;
                        _tasks.RemoveFirst();
                    }
                    else
                    {
                        break;
                    }
                }

                if (task != null)
                {
                    if (!TryExecuteTask(task))
                    {
                        throw new InvalidOperationException();
                    }
                }
            }
        }
    }
}
