using System;
using System.Runtime.CompilerServices;
using Godot;

namespace GodotTools.Internals
{
    public class EditorProgress : IDisposable
    {
        public string Task { get; }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_Create(string task, string label, int amount, bool canCancel);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_Dispose(string task);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_Step(string task, string state, int step, bool forceRefresh);

        public EditorProgress(string task, string label, int amount, bool canCancel = false)
        {
            Task = task;
            internal_Create(task, label, amount, canCancel);
        }

        ~EditorProgress()
        {
            // Should never rely on the GC to dispose EditorProgress.
            // It should be disposed immediately when the task finishes.
            GD.PushError("EditorProgress disposed by the Garbage Collector");
            Dispose();
        }

        public void Dispose()
        {
            internal_Dispose(Task);
            GC.SuppressFinalize(this);
        }

        public void Step(string state, int step = -1, bool forceRefresh = true)
        {
            internal_Step(Task, state, step, forceRefresh);
        }

        public bool TryStep(string state, int step = -1, bool forceRefresh = true)
        {
            return internal_Step(Task, state, step, forceRefresh);
        }
    }
}
