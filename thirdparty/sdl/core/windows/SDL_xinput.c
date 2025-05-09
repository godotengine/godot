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

#include "SDL_xinput.h"

// Set up for C function definitions, even when using C++
#ifdef __cplusplus
extern "C" {
#endif

XInputGetState_t SDL_XInputGetState = NULL;
XInputSetState_t SDL_XInputSetState = NULL;
XInputGetCapabilities_t SDL_XInputGetCapabilities = NULL;
XInputGetCapabilitiesEx_t SDL_XInputGetCapabilitiesEx = NULL;
XInputGetBatteryInformation_t SDL_XInputGetBatteryInformation = NULL;

static HMODULE s_pXInputDLL = NULL;
static int s_XInputDLLRefCount = 0;

#if defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES)

bool WIN_LoadXInputDLL(void)
{
    /* Getting handles to system dlls (via LoadLibrary and its variants) is not
     * supported on Xbox, thus, pointers to XInput's functions can't be
     * retrieved via GetProcAddress.
     *
     * When on Xbox, assume that XInput is already loaded, and directly map
     * its XInput.h-declared functions to the SDL_XInput* set of function
     * pointers.
     */
    SDL_XInputGetState = (XInputGetState_t)XInputGetState;
    SDL_XInputSetState = (XInputSetState_t)XInputSetState;
    SDL_XInputGetCapabilities = (XInputGetCapabilities_t)XInputGetCapabilities;
    SDL_XInputGetBatteryInformation = (XInputGetBatteryInformation_t)XInputGetBatteryInformation;

    return true;
}

void WIN_UnloadXInputDLL(void)
{
}

#else // !(defined(SDL_PLATFORM_XBOXONE) || defined(SDL_PLATFORM_XBOXSERIES))

bool WIN_LoadXInputDLL(void)
{
    if (s_pXInputDLL) {
        SDL_assert(s_XInputDLLRefCount > 0);
        s_XInputDLLRefCount++;
        return true; // already loaded
    }

    /* NOTE: Don't load XinputUap.dll
     * This is XInput emulation over Windows.Gaming.Input, and has all the
     * limitations of that API (no devices at startup, no background input, etc.)
     */
    s_pXInputDLL = LoadLibrary(TEXT("XInput1_4.dll")); // 1.4 Ships with Windows 8.
    if (!s_pXInputDLL) {
        s_pXInputDLL = LoadLibrary(TEXT("XInput1_3.dll")); // 1.3 can be installed as a redistributable component.
    }
    if (!s_pXInputDLL) {
        // "9.1.0" Ships with Vista and Win7, and is more limited than 1.3+ (e.g. XInputGetStateEx is not available.)
        s_pXInputDLL = LoadLibrary(TEXT("XInput9_1_0.dll"));
    }
    if (!s_pXInputDLL) {
        return false;
    }

    SDL_assert(s_XInputDLLRefCount == 0);
    s_XInputDLLRefCount = 1;

    // 100 is the ordinal for _XInputGetStateEx, which returns the same struct as XinputGetState, but with extra data in wButtons for the guide button, we think...
    SDL_XInputGetState = (XInputGetState_t)GetProcAddress(s_pXInputDLL, (LPCSTR)100);
    if (!SDL_XInputGetState) {
        SDL_XInputGetState = (XInputGetState_t)GetProcAddress(s_pXInputDLL, "XInputGetState");
    }
    SDL_XInputSetState = (XInputSetState_t)GetProcAddress(s_pXInputDLL, "XInputSetState");
    SDL_XInputGetCapabilities = (XInputGetCapabilities_t)GetProcAddress(s_pXInputDLL, "XInputGetCapabilities");
    // 108 is the ordinal for _XInputGetCapabilitiesEx, which additionally returns VID/PID of the controller.
    SDL_XInputGetCapabilitiesEx = (XInputGetCapabilitiesEx_t)GetProcAddress(s_pXInputDLL, (LPCSTR)108);
    SDL_XInputGetBatteryInformation = (XInputGetBatteryInformation_t)GetProcAddress(s_pXInputDLL, "XInputGetBatteryInformation");
    if (!SDL_XInputGetState || !SDL_XInputSetState || !SDL_XInputGetCapabilities) {
        WIN_UnloadXInputDLL();
        return false;
    }

    return true;
}

void WIN_UnloadXInputDLL(void)
{
    if (s_pXInputDLL) {
        SDL_assert(s_XInputDLLRefCount > 0);
        if (--s_XInputDLLRefCount == 0) {
            FreeLibrary(s_pXInputDLL);
            s_pXInputDLL = NULL;
        }
    } else {
        SDL_assert(s_XInputDLLRefCount == 0);
    }
}

#endif

// Ends C function definitions when using C++
#ifdef __cplusplus
}
#endif
