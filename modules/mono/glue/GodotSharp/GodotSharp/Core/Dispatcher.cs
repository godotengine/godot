namespace Godot
{
    public static class Dispatcher
    {
        internal static GodotTaskScheduler DefaultGodotTaskScheduler;

        private static void InitializeDefaultGodotTaskScheduler()
        {
            DefaultGodotTaskScheduler = new GodotTaskScheduler();
        }

        public static GodotSynchronizationContext SynchronizationContext => DefaultGodotTaskScheduler.Context;
    }
}
