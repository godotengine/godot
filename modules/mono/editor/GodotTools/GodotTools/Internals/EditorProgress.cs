using System;
using System.Runtime.CompilerServices;
using Godot;
using Godot.NativeInterop;

namespace GodotTools.Internals
{
    public class EditorProgress : IDisposable
    {
        public string Task { get; }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_Create(in godot_string task, in godot_string label, int amount,
            bool canCancel);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_Dispose(in godot_string task);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_Step(in godot_string task, in godot_string state, int step,
            bool forceRefresh);

        public EditorProgress(string task, string label, int amount, bool canCancel = false)
        {
            Task = task;
            using godot_string taskIn = Marshaling.mono_string_to_godot(task);
            using godot_string labelIn = Marshaling.mono_string_to_godot(label);
            internal_Create(taskIn, labelIn, amount, canCancel);
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
            using godot_string taskIn = Marshaling.mono_string_to_godot(Task);
            internal_Dispose(taskIn);
            GC.SuppressFinalize(this);
        }

        public void Step(string state, int step = -1, bool forceRefresh = true)
        {
            using godot_string taskIn = Marshaling.mono_string_to_godot(Task);
            using godot_string stateIn = Marshaling.mono_string_to_godot(state);
            internal_Step(taskIn, stateIn, step, forceRefresh);
        }

        public bool TryStep(string state, int step = -1, bool forceRefresh = true)
        {
            using godot_string taskIn = Marshaling.mono_string_to_godot(Task);
            using godot_string stateIn = Marshaling.mono_string_to_godot(state);
            return internal_Step(taskIn, stateIn, step, forceRefresh);
        }
    }
}
