using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    public static partial class GD
    {
        [UnmanagedCallersOnly]
        internal static void OnCoreApiAssemblyLoaded(godot_bool isDebug)
        {
            try
            {
                Dispatcher.InitializeDefaultGodotTaskScheduler();

                if (isDebug.ToBool())
                {
                    DebuggingUtils.InstallTraceListener();

                    AppDomain.CurrentDomain.UnhandledException += (_, e) =>
                    {
                        // Exception.ToString() includes the inner exception
                        ExceptionUtils.LogUnhandledException((Exception)e.ExceptionObject);
                    };
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }
    }
}
