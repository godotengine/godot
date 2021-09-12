using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    public static class Dispatcher
    {
        internal static GodotTaskScheduler DefaultGodotTaskScheduler;

        [UnmanagedCallersOnly]
        internal static void InitializeDefaultGodotTaskScheduler()
        {
            try
            {
                DefaultGodotTaskScheduler = new GodotTaskScheduler();
            }
            catch (Exception e)
            {
                ExceptionUtils.DebugUnhandledException(e);
            }
        }

        public static GodotSynchronizationContext SynchronizationContext => DefaultGodotTaskScheduler.Context;
    }
}
