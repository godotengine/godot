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
        // Set initialization getter, it needs to be this roundabout for godot as dll to work.
        // Static library could access [UnmanagedCallersOnly] with entry point directly.
        [LibraryImport("libgodot")]
        private static partial void set_load_from_executable_fn(nint callback);

        [UnmanagedCallersOnly]
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static nint LoadFromExecutable()
        {
            // Console.WriteLine("LoadFromExecutable called");
#if TOOLS
            // Use builtin dotnet loading for editor.
            return nint.Zero;
#else
            // Use NativeAOT loader for export build.s
            return global::GodotPlugins.Game.Main.GetInitializePointer();
#endif
        }

#if !GODOT_WEB
        static unsafe int Main()
        {
            // Console.WriteLine("LibGodot static main begin");
            List<string> args = [.. Environment.GetCommandLineArgs()];

            // Console.WriteLine($"Environment.CurrentDirectory: {Environment.CurrentDirectory}");
            var instance = LibGodot.CreateGodotInstance([.. args]);
            if (instance is null)
            {
                Console.Error.WriteLine("Error creating Godot instance");
                return 1;
            }

            set_load_from_executable_fn((nint)(delegate* unmanaged<nint>)&LoadFromExecutable);

            // Console.WriteLine("LibGodot before start");

            instance.Start();

            // Console.WriteLine("LibGodot before first iteration");
            while (!instance.Iteration()) { }

            // Console.WriteLine("LibGodot before destroy");
            instance.Dispose();

            return 0;
        }

#else // GODOT_WEB

        // Load emscriptens functions.
        [DllImport("*")]
        private static extern void emscripten_set_main_loop(nint func, int fps, byte simulate_infinite_loop);
        // [DllImport("*")]
        // private static extern byte emscripten_is_main_browser_thread();
        [DllImport("*")]
        private static extern void emscripten_cancel_main_loop();
        [DllImport("*")]
        private static extern void emscripten_force_exit(int status);

        // Load godot js libraries.
        [DllImport("*")]
        private static unsafe extern void godot_js_os_finish_async(nint func);

        // Custom web iteration.
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
                // Console.WriteLine("LibGodot before destroy");
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
            // Checking that we are not in the actual browser thread inside multithreaded main.
            // This makes it similar to enabled PROXY_TO_PTHREAD, which godot supports, so it's fine.
            // The only bad thing is that there is no automatic support for transferring offscreen canvas
            // to this main thread, which leads to a hack that adds support for it.
            // Console.WriteLine($"LibGodot is main browser thread: {emscripten_is_main_browser_thread() != 0}");
            // Console.WriteLine("LibGodot web main begin");
            List<string> args = [.. Environment.GetCommandLineArgs()];
            instance = LibGodot.CreateGodotInstance([.. args]);
            if (instance is null)
            {
                Console.Error.WriteLine("Error creating Godot instance");
                return 1;
            }

            set_load_from_executable_fn((nint)(delegate* unmanaged<nint>)&LoadFromExecutable);
            // Console.WriteLine("LibGodot web before start");

            instance.Start();
            // Console.WriteLine("LibGodot web start");

            emscripten_set_main_loop((nint)(delegate* unmanaged<void>)&MainLoopCallback, -1, 0);

            // Console.WriteLine("LibGodot before first iteration");
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
