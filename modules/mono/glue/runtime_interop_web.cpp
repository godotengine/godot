/**************************************************************************/
/*  runtime_interop_web.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#ifdef WEB_ENABLED

#include "runtime_interop.h"
#include "core/os/os.h"
#include <emscripten.h>

extern "C" {
// Defined in platform/web/js/libs/library_godot_mono.js
extern int godot_mono_init();
}

namespace godotsharp {

void initialize_web_runtime() {
    print_verbose(".NET: Initializing WebAssembly runtime...");

    // Call the synchronous Emscripten wrapper
    // This expects that ASYNCIFY is enabled if the underlying dotnet setup blocks
    int success = godot_mono_init();
    if (!success) {
        ERR_PRINT(".NET: Failed to initialize the WASM .NET runtime.");
    } else {
        print_verbose(".NET: WebAssembly runtime initialized successfully.");
    }
}

} // namespace godotsharp

#endif // WEB_ENABLED
