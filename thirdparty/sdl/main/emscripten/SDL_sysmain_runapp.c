/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#ifdef SDL_PLATFORM_EMSCRIPTEN

#include <emscripten/emscripten.h>

EM_JS_DEPS(sdlrunapp, "$dynCall,$stringToNewUTF8");

int SDL_RunApp(int argc, char* argv[], SDL_main_func mainFunction, void * reserved)
{
    (void)reserved;

    // Move any URL params that start with "SDL_" over to environment
    //  variables, so the hint system can pick them up, etc, much like a user
    //  can set them from a shell prompt on a desktop machine. Ignore all
    //  other params, in case the app wants to use them for something.
    MAIN_THREAD_EM_ASM({
        var parms = new URLSearchParams(window.location.search);
        for (const [key, value] of parms) {
            if (key.startsWith("SDL_")) {
                var ckey = stringToNewUTF8(key);
                var cvalue = stringToNewUTF8(value);
                if ((ckey != 0) && (cvalue != 0)) {
                    //console.log("Setting SDL env var '" + key + "' to '" + value + "' ...");
                    dynCall('iiii', $0, [ckey, cvalue, 1]);
                }
                _free(ckey);  // these must use free(), not SDL_free()!
                _free(cvalue);
            }
        }
    }, SDL_setenv_unsafe);

    return mainFunction(argc, argv);
}

#endif
