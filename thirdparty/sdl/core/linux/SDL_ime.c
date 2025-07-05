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

#include "SDL_ime.h"
#include "SDL_ibus.h"
#include "SDL_fcitx.h"

typedef bool (*SDL_IME_Init_t)(void);
typedef void (*SDL_IME_Quit_t)(void);
typedef void (*SDL_IME_SetFocus_t)(bool);
typedef void (*SDL_IME_Reset_t)(void);
typedef bool (*SDL_IME_ProcessKeyEvent_t)(Uint32, Uint32, bool down);
typedef void (*SDL_IME_UpdateTextInputArea_t)(SDL_Window *window);
typedef void (*SDL_IME_PumpEvents_t)(void);

static SDL_IME_Init_t SDL_IME_Init_Real = NULL;
static SDL_IME_Quit_t SDL_IME_Quit_Real = NULL;
static SDL_IME_SetFocus_t SDL_IME_SetFocus_Real = NULL;
static SDL_IME_Reset_t SDL_IME_Reset_Real = NULL;
static SDL_IME_ProcessKeyEvent_t SDL_IME_ProcessKeyEvent_Real = NULL;
static SDL_IME_UpdateTextInputArea_t SDL_IME_UpdateTextInputArea_Real = NULL;
static SDL_IME_PumpEvents_t SDL_IME_PumpEvents_Real = NULL;

static void InitIME(void)
{
    static bool inited = false;
#ifdef HAVE_FCITX
    const char *im_module = SDL_getenv("SDL_IM_MODULE");
    const char *xmodifiers = SDL_getenv("XMODIFIERS");
#endif

    if (inited == true) {
        return;
    }

    inited = true;

    // See if fcitx IME support is being requested
#ifdef HAVE_FCITX
    if (!SDL_IME_Init_Real &&
        ((im_module && SDL_strcmp(im_module, "fcitx") == 0) ||
         (!im_module && xmodifiers && SDL_strstr(xmodifiers, "@im=fcitx") != NULL))) {
        SDL_IME_Init_Real = SDL_Fcitx_Init;
        SDL_IME_Quit_Real = SDL_Fcitx_Quit;
        SDL_IME_SetFocus_Real = SDL_Fcitx_SetFocus;
        SDL_IME_Reset_Real = SDL_Fcitx_Reset;
        SDL_IME_ProcessKeyEvent_Real = SDL_Fcitx_ProcessKeyEvent;
        SDL_IME_UpdateTextInputArea_Real = SDL_Fcitx_UpdateTextInputArea;
        SDL_IME_PumpEvents_Real = SDL_Fcitx_PumpEvents;
    }
#endif // HAVE_FCITX

    // default to IBus
#ifdef HAVE_IBUS_IBUS_H
    if (!SDL_IME_Init_Real) {
        SDL_IME_Init_Real = SDL_IBus_Init;
        SDL_IME_Quit_Real = SDL_IBus_Quit;
        SDL_IME_SetFocus_Real = SDL_IBus_SetFocus;
        SDL_IME_Reset_Real = SDL_IBus_Reset;
        SDL_IME_ProcessKeyEvent_Real = SDL_IBus_ProcessKeyEvent;
        SDL_IME_UpdateTextInputArea_Real = SDL_IBus_UpdateTextInputArea;
        SDL_IME_PumpEvents_Real = SDL_IBus_PumpEvents;
    }
#endif // HAVE_IBUS_IBUS_H
}

bool SDL_IME_Init(void)
{
    InitIME();

    if (SDL_IME_Init_Real) {
        if (SDL_IME_Init_Real()) {
            return true;
        }

        // uhoh, the IME implementation's init failed! Disable IME support.
        SDL_IME_Init_Real = NULL;
        SDL_IME_Quit_Real = NULL;
        SDL_IME_SetFocus_Real = NULL;
        SDL_IME_Reset_Real = NULL;
        SDL_IME_ProcessKeyEvent_Real = NULL;
        SDL_IME_UpdateTextInputArea_Real = NULL;
        SDL_IME_PumpEvents_Real = NULL;
    }

    return false;
}

void SDL_IME_Quit(void)
{
    if (SDL_IME_Quit_Real) {
        SDL_IME_Quit_Real();
    }
}

void SDL_IME_SetFocus(bool focused)
{
    if (SDL_IME_SetFocus_Real) {
        SDL_IME_SetFocus_Real(focused);
    }
}

void SDL_IME_Reset(void)
{
    if (SDL_IME_Reset_Real) {
        SDL_IME_Reset_Real();
    }
}

bool SDL_IME_ProcessKeyEvent(Uint32 keysym, Uint32 keycode, bool down)
{
    if (SDL_IME_ProcessKeyEvent_Real) {
        return SDL_IME_ProcessKeyEvent_Real(keysym, keycode, down);
    }

    return false;
}

void SDL_IME_UpdateTextInputArea(SDL_Window *window)
{
    if (SDL_IME_UpdateTextInputArea_Real) {
        SDL_IME_UpdateTextInputArea_Real(window);
    }
}

void SDL_IME_PumpEvents(void)
{
    if (SDL_IME_PumpEvents_Real) {
        SDL_IME_PumpEvents_Real();
    }
}
