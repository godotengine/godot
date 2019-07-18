using System.Runtime.CompilerServices;

namespace Godot
{
    public static class Dispatcher
    {
        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern GodotTaskScheduler godot_icall_DefaultGodotTaskScheduler();

        public static GodotSynchronizationContext SynchronizationContext =>
            godot_icall_DefaultGodotTaskScheduler().Context;
    }
}
