#if LIBGODOT_ENABLED
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

#if GODOT_WEB
using System.Runtime.InteropServices.JavaScript;
// Silence web build warnings for [JSImport]/[JSExport]
[assembly: System.Runtime.Versioning.SupportedOSPlatform("browser")]
#endif

namespace GodotPlugins.Game
{
    internal static partial class Initializer
    {
        // 2dog: shared builds register the managed initializer through this callback.
        [LibraryImport("libgodot")]
        private static partial void set_load_from_executable_fn(nint callback);

        [UnmanagedCallersOnly]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static nint LoadFromExecutable()
        {
#if TOOLS
            return nint.Zero;
#else
            return global::GodotPlugins.Game.Main.GetInitializePointer();
#endif
        }

#if !GODOT_WEB
        static unsafe int Main()
        {
            List<string> args = [.. Environment.GetCommandLineArgs()];
            var instance = LibGodot.CreateGodotInstance([.. args]);
            if (instance is null)
            {
                Console.Error.WriteLine("Error creating Godot instance");
                return 1;
            }

            set_load_from_executable_fn((nint)(delegate* unmanaged<nint>)&LoadFromExecutable);

            if (!instance.Start())
            {
                instance.Dispose();
                return 1;
            }
            while (!instance.Iteration()) { }
            instance.Dispose();

            return 0;
        }

#else // GODOT_WEB

        [DllImport("*")]
        private static extern void emscripten_set_main_loop(nint func, int fps, byte simulate_infinite_loop);
        [DllImport("*")]
        private static extern void emscripten_cancel_main_loop();
        [DllImport("*")]
        private static extern void emscripten_force_exit(int status);

        [DllImport("*")]
        private static unsafe extern void godot_js_os_finish_async(nint func);

        [LibraryImport("libgodot")]
        private static partial byte libgodot_web_iteration();

        private static GodotInstance? instance = null;
        private static bool shutdownComplete = false;

        [UnmanagedCallersOnly]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void ExitCallback()
        {
            if (!shutdownComplete)
            {
                return; // Still waiting.
            }
            if (instance is not null)
            {
                instance.Dispose();
                instance = null;
            }
            emscripten_cancel_main_loop();
            emscripten_force_exit(0);
        }

        [UnmanagedCallersOnly]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void CleanupAfterSync()
        {
            shutdownComplete = true;
        }

        private static unsafe void SetupExit()
        {
            emscripten_cancel_main_loop();
            emscripten_set_main_loop((nint)(delegate* unmanaged<void>)&ExitCallback, -1, 0);
            godot_js_os_finish_async((nint)(delegate* unmanaged<void>)&CleanupAfterSync);
        }


        [UnmanagedCallersOnly]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static void MainLoopCallback()
        {
            if (libgodot_web_iteration() != 0)
            {
                SetupExit();
            }
        }

        static unsafe int Main()
        {
            List<string> args = [.. Environment.GetCommandLineArgs()];
            instance = LibGodot.CreateGodotInstance([.. args]);
            if (instance is null)
            {
                Console.Error.WriteLine("Error creating Godot instance");
                return 1;
            }

            set_load_from_executable_fn((nint)(delegate* unmanaged<nint>)&LoadFromExecutable);
            if (!instance.Start())
            {
                instance.Dispose();
                instance = null;
                return 1;
            }

            emscripten_set_main_loop((nint)(delegate* unmanaged<void>)&MainLoopCallback, -1, 0);

            if (libgodot_web_iteration() != 0)
            {
                SetupExit();
                return 0;
            }

            return 0;
        }
#endif
    }
}
#endif
