using System.Runtime.CompilerServices;

namespace Godot
{
    public static class Dispatcher
    {
        /// <summary>
        /// Implements an external instance of GodotTaskScheduler.
        /// </summary>
        /// <returns>A GodotTaskScheduler instance.</returns>
        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern GodotTaskScheduler godot_icall_DefaultGodotTaskScheduler();

        /// <summary>
        /// Initializes the synchronization context as the context of the GodotTaskScheduler.
        /// </summary>
        public static GodotSynchronizationContext SynchronizationContext =>
            godot_icall_DefaultGodotTaskScheduler().Context;
    }
}
