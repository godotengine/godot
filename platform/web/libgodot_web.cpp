/**
 * libgodot_web.cpp
 * Web platform LibGodot entrypoint for C# (.NET) WASM exports.
 *
 * This file provides the static library entrypoint that allows the .NET
 * runtime (via Emscripten) to call into Godot's main loop when building
 * C# web exports. It mirrors the pattern used by libgodot_linuxbsd.cpp
 * and libgodot_macos.mm, adapted for the Emscripten/WASM environment.
 *
 * Build condition: Only compiled when both `platform=web` and
 * `module_mono_enabled=yes` are set, and `target` is editor or template.
 */

#include "core/extension/godot_instance.h"
#include "core/os/os.h"
#include "main/main.h"
#include "platform/web/os_web.h"

#include <emscripten.h>
#include <emscripten/html5.h>

// ---------------------------------------------------------------------------
// Forward declarations from web_main.cpp / web_runtime.cpp
// ---------------------------------------------------------------------------
extern "C" void godot_js_os_run_async(void);

// ---------------------------------------------------------------------------
// LibGodot Web C API
// Exported as C symbols so the .NET WASM runtime can call them via P/Invoke
// or Emscripten JS glue without C++ name mangling.
// ---------------------------------------------------------------------------
extern "C" {

/**
 * godot_libgodot_init
 *
 * Called once by the .NET host (GodotPlugins / Main.cs) after the Emscripten
 * Module is ready.  Mirrors the role of the native main() in web_main.cpp but
 * does NOT start the Emscripten main loop itself — that is the caller's
 * responsibility so that .NET's own event loop can drive it.
 *
 * @param argc  Argument count forwarded from JS bootstrapper.
 * @param argv  Argument vector forwarded from JS bootstrapper.
 * @return      0 on success, non-zero on error.
 */
EMSCRIPTEN_KEEPALIVE
int godot_libgodot_init(int argc, char *argv[]) {
    // OS_Web singleton is created in web_main.cpp before this is called.
    // If somehow it isn't, bail out safely.
    OS_Web *os = static_cast<OS_Web *>(OS::get_singleton());
    if (!os) {
        return ERR_UNAVAILABLE;
    }

    Error err = Main::setup(argv[0], argc - 1, &argv[1]);
    if (err != OK) {
        // Setup failed (e.g. --version / --help already handled).
        return (err == ERR_HELP) ? 0 : (int)err;
    }

    if (Main::start() != 0) {
        return ERR_CANT_CREATE;
    }

    return OK;
}

/**
 * godot_libgodot_iteration
 *
 * Advance the Godot main loop by exactly one frame.
 * Called from the Emscripten requestAnimationFrame callback that is set up
 * by the .NET bootstrapper (or directly from emscripten_set_main_loop).
 *
 * @return  true  if the engine wants to keep running.
 *          false if Main::iteration() signalled quit.
 */
EMSCRIPTEN_KEEPALIVE
bool godot_libgodot_iteration() {
    return !Main::iteration();
}

/**
 * godot_libgodot_terminate
 *
 * Cleanly shut down the Godot engine.  Must be called after the main loop
 * exits, before the Emscripten Module is destroyed.
 */
EMSCRIPTEN_KEEPALIVE
void godot_libgodot_terminate() {
    Main::cleanup();
}

} // extern "C"
