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

#if defined(SDL_VIDEO_DRIVER_WAYLAND) || defined(SDL_VIDEO_DRIVER_X11)

#include "SDL_keyboard_c.h"
#include "SDL_keysym_to_scancode_c.h"
#include "imKStoUCS.h"


// Extended key code mappings
static const struct
{
    Uint32 keysym;
    SDL_Keycode keycode;
} keysym_to_keycode_table[] = {
    { 0xfe03, SDLK_MODE }, // XK_ISO_Level3_Shift
    { 0xfe11, SDLK_LEVEL5_SHIFT }, // XK_ISO_Level5_Shift
    { 0xfe20, SDLK_LEFT_TAB }, // XK_ISO_Left_Tab
    { 0xff20, SDLK_MULTI_KEY_COMPOSE }, // XK_Multi_key
    { 0xffe7, SDLK_LMETA }, // XK_Meta_L
    { 0xffe8, SDLK_RMETA }, // XK_Meta_R
    { 0xffed, SDLK_LHYPER }, // XK_Hyper_L
    { 0xffee, SDLK_RHYPER }, // XK_Hyper_R
};

SDL_Keycode SDL_GetKeyCodeFromKeySym(Uint32 keysym, Uint32 keycode, SDL_Keymod modifiers)
{
    SDL_Keycode sdl_keycode = SDL_KeySymToUcs4(keysym);

    if (!sdl_keycode) {
        for (int i = 0; i < SDL_arraysize(keysym_to_keycode_table); ++i) {
            if (keysym == keysym_to_keycode_table[i].keysym) {
                return keysym_to_keycode_table[i].keycode;
            }
        }
    }

    if (!sdl_keycode) {
        const SDL_Scancode scancode = SDL_GetScancodeFromKeySym(keysym, keycode);
        if (scancode != SDL_SCANCODE_UNKNOWN) {
            sdl_keycode = SDL_GetKeymapKeycode(NULL, scancode, modifiers);
        }
    }

    return sdl_keycode;
}

#endif // SDL_VIDEO_DRIVER_WAYLAND || SDL_VIDEO_DRIVER_X11
