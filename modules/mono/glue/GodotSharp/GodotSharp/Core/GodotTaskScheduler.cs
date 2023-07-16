using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace Godot
{
    /// <summary>
    /// GodotTaskScheduler contains a linked list of tasks to perform as a queue. Methods
    /// within the class are used to control the queue and perform the contained tasks.
    /// </summary>
    public sealed class GodotTaskScheduler : TaskScheduler, IDisposable
    {
        /// <summary>
        /// The current synchronization context.
        /// </summary>
        internal GodotSynchronizationContext Context { get; }

        /// <summary>
        /// The queue of tasks for the task scheduler.
        /// </summary>
        private readonly LinkedList<Task> _tasks = new LinkedList<Task>();

        /// <summary>
        /// Constructs a new GodotTaskScheduler instance.
        /// </summary>
        public GodotTaskScheduler()
        {
            Context = new GodotSynchronizationContext();
            SynchronizationContext.SetSynchronizationContext(Context);
        }

        /// <summary>
        /// Enqueues a new Task.
        /// </summary>
        /// <param name="task">The task to queue up.</param>
        protected sealed override void QueueTask(Task task)
        {
            lock (_tasks)
            {
                _tasks.AddLast(task);
            }
        }

        /// <summary>
        /// Tries to execute a new task.
        /// </summary>
        /// <param name="task">The task to execute.</param>
        /// <param name="taskWasPreviouslyQueued">Whether or not the task
        /// was previously queued.</param>
        /// <returns><see langword="true"/> if the task was successfully
        /// executed; otherwise, <see langword="false"/>.</returns>
        protected sealed override bool TryExecuteTaskInline(Task task, bool taskWasPreviouslyQueued)
        {
            if (SynchronizationContext.Current != Context)
                return false;

            if (taskWasPreviouslyQueued)
                TryDequeue(task);

            return TryExecuteTask(task);
        }

        /// <summary>
        /// Tries to dequeue a task.
        /// </summary>
        /// <param name="task">The task to remove.</param>
        /// <returns><see langword="true"/> if the task was successfully
        /// dequeued; otherwise, <see langword="false"/>.</returns>
        protected sealed override bool TryDequeue(Task task)
        {
            lock (_tasks)
            {
                return _tasks.Remove(task);
            }
        }

        /// <summary>
        /// Get the currently scheduled tasks.
        /// </summary>
        /// <returns>The scheduled tasks.</returns>
        protected sealed override IEnumerable<Task> GetScheduledTasks()
        {
            lock (_tasks)
            {
                foreach (Task task in _tasks)
                    yield return task;
            }
        }

        /// <summary>
        /// Executes all queued tasks and pending tasks from the current context.
        /// </summary>
        public void Activate()
        {
            ExecuteQueuedTasks();
            Context.ExecutePendingContinuations();
        }

        /// <summary>
        /// Loops through and attempts to execute each task in _tasks.
        /// </summary>
        /// <exception cref="InvalidOperationException"></exception>
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

        /// <summary>
        /// Disposes of this <see cref="GodotTaskScheduler"/>.
        /// </summary>
        public void Dispose()
        {
            Context.Dispose();
        }
    }
}
