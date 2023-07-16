using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    /// <summary>
    /// Provides a dispatcher to handle task scheduling and invocations.
    /// </summary>
    public static class Dispatcher
    {
        internal static GodotTaskScheduler DefaultGodotTaskScheduler;

        internal static void InitializeDefaultGodotTaskScheduler()
        {
            DefaultGodotTaskScheduler?.Dispose();
            DefaultGodotTaskScheduler = new GodotTaskScheduler();
        }

        /// <summary>
        /// Initializes the synchronization context as the context of the DefaultGodotTaskScheduler.
        /// </summary>
        public static GodotSynchronizationContext SynchronizationContext => DefaultGodotTaskScheduler.Context;
    }
}
