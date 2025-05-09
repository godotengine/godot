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

#ifdef SDL_VIDEO_DRIVER_HAIKU

#include <SupportDefs.h>
#include <support/UTF8.h>

#ifdef __cplusplus
extern "C" {
#endif


#include "SDL_bkeyboard.h"


#define KEYMAP_SIZE 128


static SDL_Scancode keymap[KEYMAP_SIZE];

void HAIKU_InitOSKeymap(void)
{
        for ( uint i = 0; i < SDL_arraysize(keymap); ++i ) {
            keymap[i] = SDL_SCANCODE_UNKNOWN;
        }

        keymap[0x01]        = SDL_SCANCODE_ESCAPE;
        keymap[B_F1_KEY]    = SDL_SCANCODE_F1;
        keymap[B_F2_KEY]    = SDL_SCANCODE_F2;
        keymap[B_F3_KEY]    = SDL_SCANCODE_F3;
        keymap[B_F4_KEY]    = SDL_SCANCODE_F4;
        keymap[B_F5_KEY]    = SDL_SCANCODE_F5;
        keymap[B_F6_KEY]    = SDL_SCANCODE_F6;
        keymap[B_F7_KEY]    = SDL_SCANCODE_F7;
        keymap[B_F8_KEY]    = SDL_SCANCODE_F8;
        keymap[B_F9_KEY]    = SDL_SCANCODE_F9;
        keymap[B_F10_KEY]   = SDL_SCANCODE_F10;
        keymap[B_F11_KEY]   = SDL_SCANCODE_F11;
        keymap[B_F12_KEY]   = SDL_SCANCODE_F12;
        keymap[B_PRINT_KEY] = SDL_SCANCODE_PRINTSCREEN;
        keymap[B_SCROLL_KEY]= SDL_SCANCODE_SCROLLLOCK;
        keymap[B_PAUSE_KEY] = SDL_SCANCODE_PAUSE;
        keymap[0x11]        = SDL_SCANCODE_GRAVE;
        keymap[0x12]        = SDL_SCANCODE_1;
        keymap[0x13]        = SDL_SCANCODE_2;
        keymap[0x14]        = SDL_SCANCODE_3;
        keymap[0x15]        = SDL_SCANCODE_4;
        keymap[0x16]        = SDL_SCANCODE_5;
        keymap[0x17]        = SDL_SCANCODE_6;
        keymap[0x18]        = SDL_SCANCODE_7;
        keymap[0x19]        = SDL_SCANCODE_8;
        keymap[0x1a]        = SDL_SCANCODE_9;
        keymap[0x1b]        = SDL_SCANCODE_0;
        keymap[0x1c]        = SDL_SCANCODE_MINUS;
        keymap[0x1d]        = SDL_SCANCODE_EQUALS;
        keymap[0x1e]        = SDL_SCANCODE_BACKSPACE;
        keymap[0x1f]        = SDL_SCANCODE_INSERT;
        keymap[0x20]        = SDL_SCANCODE_HOME;
        keymap[0x21]        = SDL_SCANCODE_PAGEUP;
        keymap[0x22]        = SDL_SCANCODE_NUMLOCKCLEAR;
        keymap[0x23]        = SDL_SCANCODE_KP_DIVIDE;
        keymap[0x24]        = SDL_SCANCODE_KP_MULTIPLY;
        keymap[0x25]        = SDL_SCANCODE_KP_MINUS;
        keymap[0x26]        = SDL_SCANCODE_TAB;
        keymap[0x27]        = SDL_SCANCODE_Q;
        keymap[0x28]        = SDL_SCANCODE_W;
        keymap[0x29]        = SDL_SCANCODE_E;
        keymap[0x2a]        = SDL_SCANCODE_R;
        keymap[0x2b]        = SDL_SCANCODE_T;
        keymap[0x2c]        = SDL_SCANCODE_Y;
        keymap[0x2d]        = SDL_SCANCODE_U;
        keymap[0x2e]        = SDL_SCANCODE_I;
        keymap[0x2f]        = SDL_SCANCODE_O;
        keymap[0x30]        = SDL_SCANCODE_P;
        keymap[0x31]        = SDL_SCANCODE_LEFTBRACKET;
        keymap[0x32]        = SDL_SCANCODE_RIGHTBRACKET;
        keymap[0x33]        = SDL_SCANCODE_BACKSLASH;
        keymap[0x34]        = SDL_SCANCODE_DELETE;
        keymap[0x35]        = SDL_SCANCODE_END;
        keymap[0x36]        = SDL_SCANCODE_PAGEDOWN;
        keymap[0x37]        = SDL_SCANCODE_KP_7;
        keymap[0x38]        = SDL_SCANCODE_KP_8;
        keymap[0x39]        = SDL_SCANCODE_KP_9;
        keymap[0x3a]        = SDL_SCANCODE_KP_PLUS;
        keymap[0x3b]        = SDL_SCANCODE_CAPSLOCK;
        keymap[0x3c]        = SDL_SCANCODE_A;
        keymap[0x3d]        = SDL_SCANCODE_S;
        keymap[0x3e]        = SDL_SCANCODE_D;
        keymap[0x3f]        = SDL_SCANCODE_F;
        keymap[0x40]        = SDL_SCANCODE_G;
        keymap[0x41]        = SDL_SCANCODE_H;
        keymap[0x42]        = SDL_SCANCODE_J;
        keymap[0x43]        = SDL_SCANCODE_K;
        keymap[0x44]        = SDL_SCANCODE_L;
        keymap[0x45]        = SDL_SCANCODE_SEMICOLON;
        keymap[0x46]        = SDL_SCANCODE_APOSTROPHE;
        keymap[0x47]        = SDL_SCANCODE_RETURN;
        keymap[0x48]        = SDL_SCANCODE_KP_4;
        keymap[0x49]        = SDL_SCANCODE_KP_5;
        keymap[0x4a]        = SDL_SCANCODE_KP_6;
        keymap[0x4b]        = SDL_SCANCODE_LSHIFT;
        keymap[0x4c]        = SDL_SCANCODE_Z;
        keymap[0x4d]        = SDL_SCANCODE_X;
        keymap[0x4e]        = SDL_SCANCODE_C;
        keymap[0x4f]        = SDL_SCANCODE_V;
        keymap[0x50]        = SDL_SCANCODE_B;
        keymap[0x51]        = SDL_SCANCODE_N;
        keymap[0x52]        = SDL_SCANCODE_M;
        keymap[0x53]        = SDL_SCANCODE_COMMA;
        keymap[0x54]        = SDL_SCANCODE_PERIOD;
        keymap[0x55]        = SDL_SCANCODE_SLASH;
        keymap[0x56]        = SDL_SCANCODE_RSHIFT;
        keymap[0x57]        = SDL_SCANCODE_UP;
        keymap[0x58]        = SDL_SCANCODE_KP_1;
        keymap[0x59]        = SDL_SCANCODE_KP_2;
        keymap[0x5a]        = SDL_SCANCODE_KP_3;
        keymap[0x5b]        = SDL_SCANCODE_KP_ENTER;
        keymap[0x5c]        = SDL_SCANCODE_LCTRL;
        keymap[0x5d]        = SDL_SCANCODE_LALT;
        keymap[0x5e]        = SDL_SCANCODE_SPACE;
        keymap[0x5f]        = SDL_SCANCODE_RALT;
        keymap[0x60]        = SDL_SCANCODE_RCTRL;
        keymap[0x61]        = SDL_SCANCODE_LEFT;
        keymap[0x62]        = SDL_SCANCODE_DOWN;
        keymap[0x63]        = SDL_SCANCODE_RIGHT;
        keymap[0x64]        = SDL_SCANCODE_KP_0;
        keymap[0x65]        = SDL_SCANCODE_KP_PERIOD;
        keymap[0x66]        = SDL_SCANCODE_LGUI;
        keymap[0x67]        = SDL_SCANCODE_RGUI;
        keymap[0x68]        = SDL_SCANCODE_MENU;
        keymap[0x69]        = SDL_SCANCODE_2; // SDLK_EURO
        keymap[0x6a]        = SDL_SCANCODE_KP_EQUALS;
        keymap[0x6b]        = SDL_SCANCODE_POWER;
}

SDL_Scancode HAIKU_GetScancodeFromBeKey(int32 bkey) {
    if (bkey > 0 && bkey < (int32)SDL_arraysize(keymap)) {
        return keymap[bkey];
    } else {
        return SDL_SCANCODE_UNKNOWN;
    }
}

#ifdef __cplusplus
}
#endif

#endif // SDL_VIDEO_DRIVER_HAIKU
