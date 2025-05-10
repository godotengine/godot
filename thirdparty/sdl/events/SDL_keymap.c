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

#include "SDL_keymap_c.h"
#include "SDL_keyboard_c.h"

static SDL_Keycode SDL_GetDefaultKeyFromScancode(SDL_Scancode scancode, SDL_Keymod modstate);
static SDL_Scancode SDL_GetDefaultScancodeFromKey(SDL_Keycode key, SDL_Keymod *modstate);

SDL_Keymap *SDL_CreateKeymap(bool auto_release)
{
    SDL_Keymap *keymap = (SDL_Keymap *)SDL_calloc(1, sizeof(*keymap));
    if (!keymap) {
        return NULL;
    }

    keymap->auto_release = auto_release;
    keymap->scancode_to_keycode = SDL_CreateHashTable(256, false, SDL_HashID, SDL_KeyMatchID, NULL, NULL);
    keymap->keycode_to_scancode = SDL_CreateHashTable(256, false, SDL_HashID, SDL_KeyMatchID, NULL, NULL);
    if (!keymap->scancode_to_keycode || !keymap->keycode_to_scancode) {
        SDL_DestroyKeymap(keymap);
        return NULL;
    }
    return keymap;
}

static SDL_Keymod NormalizeModifierStateForKeymap(SDL_Keymod modstate)
{
    // The modifiers that affect the keymap are: SHIFT, CAPS, ALT, MODE, and LEVEL5
    modstate &= (SDL_KMOD_SHIFT | SDL_KMOD_CAPS | SDL_KMOD_ALT | SDL_KMOD_MODE | SDL_KMOD_LEVEL5);

    // If either right or left Shift are set, set both in the output
    if (modstate & SDL_KMOD_SHIFT) {
        modstate |= SDL_KMOD_SHIFT;
    }

    // If either right or left Alt are set, set both in the output
    if (modstate & SDL_KMOD_ALT) {
        modstate |= SDL_KMOD_ALT;
    }

    return modstate;
}

void SDL_SetKeymapEntry(SDL_Keymap *keymap, SDL_Scancode scancode, SDL_Keymod modstate, SDL_Keycode keycode)
{
    if (!keymap) {
        return;
    }

    modstate = NormalizeModifierStateForKeymap(modstate);
    Uint32 key = ((Uint32)modstate << 16) | scancode;
    const void *value;
    if (SDL_FindInHashTable(keymap->scancode_to_keycode, (void *)(uintptr_t)key, &value)) {
        const SDL_Keycode existing_keycode = (SDL_Keycode)(uintptr_t)value;
        if (existing_keycode == keycode) {
            // We already have this mapping
            return;
        }
        // InsertIntoHashTable will replace the existing entry in the keymap atomically.
    }
    SDL_InsertIntoHashTable(keymap->scancode_to_keycode, (void *)(uintptr_t)key, (void *)(uintptr_t)keycode, true);

    bool update_keycode = true;
    if (SDL_FindInHashTable(keymap->keycode_to_scancode, (void *)(uintptr_t)keycode, &value)) {
        const Uint32 existing_value = (Uint32)(uintptr_t)value;
        const SDL_Keymod existing_modstate = (SDL_Keymod)(existing_value >> 16);

        // Keep the simplest combination of scancode and modifiers to generate this keycode
        if (existing_modstate <= modstate) {
            update_keycode = false;
        }
    }
    if (update_keycode) {
        SDL_InsertIntoHashTable(keymap->keycode_to_scancode, (void *)(uintptr_t)keycode, (void *)(uintptr_t)key, true);
    }
}

SDL_Keycode SDL_GetKeymapKeycode(SDL_Keymap *keymap, SDL_Scancode scancode, SDL_Keymod modstate)
{
    SDL_Keycode keycode;

    const Uint32 key = ((Uint32)NormalizeModifierStateForKeymap(modstate) << 16) | scancode;
    const void *value;
    if (keymap && SDL_FindInHashTable(keymap->scancode_to_keycode, (void *)(uintptr_t)key, &value)) {
        keycode = (SDL_Keycode)(uintptr_t)value;
    } else {
        keycode = SDL_GetDefaultKeyFromScancode(scancode, modstate);
    }
    return keycode;
}

SDL_Scancode SDL_GetKeymapScancode(SDL_Keymap *keymap, SDL_Keycode keycode, SDL_Keymod *modstate)
{
    SDL_Scancode scancode;

    const void *value;
    if (keymap && SDL_FindInHashTable(keymap->keycode_to_scancode, (void *)(uintptr_t)keycode, &value)) {
        scancode = (SDL_Scancode)((uintptr_t)value & 0xFFFF);
        if (modstate) {
            *modstate = (SDL_Keymod)((uintptr_t)value >> 16);
        }
    } else {
        scancode = SDL_GetDefaultScancodeFromKey(keycode, modstate);
    }
    return scancode;
}

void SDL_DestroyKeymap(SDL_Keymap *keymap)
{
    if (!keymap) {
        return;
    }

    SDL_DestroyHashTable(keymap->scancode_to_keycode);
    SDL_DestroyHashTable(keymap->keycode_to_scancode);
    SDL_free(keymap);
}

static const SDL_Keycode normal_default_symbols[] = {
    SDLK_1,
    SDLK_2,
    SDLK_3,
    SDLK_4,
    SDLK_5,
    SDLK_6,
    SDLK_7,
    SDLK_8,
    SDLK_9,
    SDLK_0,
    SDLK_RETURN,
    SDLK_ESCAPE,
    SDLK_BACKSPACE,
    SDLK_TAB,
    SDLK_SPACE,
    SDLK_MINUS,
    SDLK_EQUALS,
    SDLK_LEFTBRACKET,
    SDLK_RIGHTBRACKET,
    SDLK_BACKSLASH,
    SDLK_HASH,
    SDLK_SEMICOLON,
    SDLK_APOSTROPHE,
    SDLK_GRAVE,
    SDLK_COMMA,
    SDLK_PERIOD,
    SDLK_SLASH,
};

static const SDL_Keycode shifted_default_symbols[] = {
    SDLK_EXCLAIM,
    SDLK_AT,
    SDLK_HASH,
    SDLK_DOLLAR,
    SDLK_PERCENT,
    SDLK_CARET,
    SDLK_AMPERSAND,
    SDLK_ASTERISK,
    SDLK_LEFTPAREN,
    SDLK_RIGHTPAREN,
    SDLK_RETURN,
    SDLK_ESCAPE,
    SDLK_BACKSPACE,
    SDLK_TAB,
    SDLK_SPACE,
    SDLK_UNDERSCORE,
    SDLK_PLUS,
    SDLK_LEFTBRACE,
    SDLK_RIGHTBRACE,
    SDLK_PIPE,
    SDLK_HASH,
    SDLK_COLON,
    SDLK_DBLAPOSTROPHE,
    SDLK_TILDE,
    SDLK_LESS,
    SDLK_GREATER,
    SDLK_QUESTION
};

static const struct
{
    SDL_Keycode keycode;
    SDL_Scancode scancode;
} extended_default_symbols[] = {
    { SDLK_LEFT_TAB, SDL_SCANCODE_TAB },
    { SDLK_MULTI_KEY_COMPOSE, SDL_SCANCODE_APPLICATION }, // Sun keyboards
    { SDLK_LMETA, SDL_SCANCODE_LGUI },
    { SDLK_RMETA, SDL_SCANCODE_RGUI },
    { SDLK_RHYPER, SDL_SCANCODE_APPLICATION }
};

static SDL_Keycode SDL_GetDefaultKeyFromScancode(SDL_Scancode scancode, SDL_Keymod modstate)
{
    if (((int)scancode) < SDL_SCANCODE_UNKNOWN || scancode >= SDL_SCANCODE_COUNT) {
        SDL_InvalidParamError("scancode");
        return SDLK_UNKNOWN;
    }

    if (scancode < SDL_SCANCODE_A) {
        return SDLK_UNKNOWN;
    }

    if (scancode < SDL_SCANCODE_1) {
        bool shifted = (modstate & SDL_KMOD_SHIFT) ? true : false;
#ifdef SDL_PLATFORM_APPLE
        // Apple maps to upper case for either shift or capslock inclusive
        if (modstate & SDL_KMOD_CAPS) {
            shifted = true;
        }
#else
        if (modstate & SDL_KMOD_CAPS) {
            shifted = !shifted;
        }
#endif
        if (modstate & SDL_KMOD_MODE) {
            return SDLK_UNKNOWN;
        }
        if (!shifted) {
            return (SDL_Keycode)('a' + scancode - SDL_SCANCODE_A);
        } else {
            return (SDL_Keycode)('A' + scancode - SDL_SCANCODE_A);
        }
    }

    if (scancode < SDL_SCANCODE_CAPSLOCK) {
        bool shifted = (modstate & SDL_KMOD_SHIFT) ? true : false;

        if (modstate & SDL_KMOD_MODE) {
            return SDLK_UNKNOWN;
        }
        if (!shifted) {
            return normal_default_symbols[scancode - SDL_SCANCODE_1];
        } else {
            return shifted_default_symbols[scancode - SDL_SCANCODE_1];
        }
    }

    // These scancodes are not mapped to printable keycodes
    switch (scancode) {
    case SDL_SCANCODE_DELETE:
        return SDLK_DELETE;
    case SDL_SCANCODE_CAPSLOCK:
        return SDLK_CAPSLOCK;
    case SDL_SCANCODE_F1:
        return SDLK_F1;
    case SDL_SCANCODE_F2:
        return SDLK_F2;
    case SDL_SCANCODE_F3:
        return SDLK_F3;
    case SDL_SCANCODE_F4:
        return SDLK_F4;
    case SDL_SCANCODE_F5:
        return SDLK_F5;
    case SDL_SCANCODE_F6:
        return SDLK_F6;
    case SDL_SCANCODE_F7:
        return SDLK_F7;
    case SDL_SCANCODE_F8:
        return SDLK_F8;
    case SDL_SCANCODE_F9:
        return SDLK_F9;
    case SDL_SCANCODE_F10:
        return SDLK_F10;
    case SDL_SCANCODE_F11:
        return SDLK_F11;
    case SDL_SCANCODE_F12:
        return SDLK_F12;
    case SDL_SCANCODE_PRINTSCREEN:
        return SDLK_PRINTSCREEN;
    case SDL_SCANCODE_SCROLLLOCK:
        return SDLK_SCROLLLOCK;
    case SDL_SCANCODE_PAUSE:
        return SDLK_PAUSE;
    case SDL_SCANCODE_INSERT:
        return SDLK_INSERT;
    case SDL_SCANCODE_HOME:
        return SDLK_HOME;
    case SDL_SCANCODE_PAGEUP:
        return SDLK_PAGEUP;
    case SDL_SCANCODE_END:
        return SDLK_END;
    case SDL_SCANCODE_PAGEDOWN:
        return SDLK_PAGEDOWN;
    case SDL_SCANCODE_RIGHT:
        return SDLK_RIGHT;
    case SDL_SCANCODE_LEFT:
        return SDLK_LEFT;
    case SDL_SCANCODE_DOWN:
        return SDLK_DOWN;
    case SDL_SCANCODE_UP:
        return SDLK_UP;
    case SDL_SCANCODE_NUMLOCKCLEAR:
        return SDLK_NUMLOCKCLEAR;
    case SDL_SCANCODE_KP_DIVIDE:
        return SDLK_KP_DIVIDE;
    case SDL_SCANCODE_KP_MULTIPLY:
        return SDLK_KP_MULTIPLY;
    case SDL_SCANCODE_KP_MINUS:
        return SDLK_KP_MINUS;
    case SDL_SCANCODE_KP_PLUS:
        return SDLK_KP_PLUS;
    case SDL_SCANCODE_KP_ENTER:
        return SDLK_KP_ENTER;
    case SDL_SCANCODE_KP_1:
        return SDLK_KP_1;
    case SDL_SCANCODE_KP_2:
        return SDLK_KP_2;
    case SDL_SCANCODE_KP_3:
        return SDLK_KP_3;
    case SDL_SCANCODE_KP_4:
        return SDLK_KP_4;
    case SDL_SCANCODE_KP_5:
        return SDLK_KP_5;
    case SDL_SCANCODE_KP_6:
        return SDLK_KP_6;
    case SDL_SCANCODE_KP_7:
        return SDLK_KP_7;
    case SDL_SCANCODE_KP_8:
        return SDLK_KP_8;
    case SDL_SCANCODE_KP_9:
        return SDLK_KP_9;
    case SDL_SCANCODE_KP_0:
        return SDLK_KP_0;
    case SDL_SCANCODE_KP_PERIOD:
        return SDLK_KP_PERIOD;
    case SDL_SCANCODE_APPLICATION:
        return SDLK_APPLICATION;
    case SDL_SCANCODE_POWER:
        return SDLK_POWER;
    case SDL_SCANCODE_KP_EQUALS:
        return SDLK_KP_EQUALS;
    case SDL_SCANCODE_F13:
        return SDLK_F13;
    case SDL_SCANCODE_F14:
        return SDLK_F14;
    case SDL_SCANCODE_F15:
        return SDLK_F15;
    case SDL_SCANCODE_F16:
        return SDLK_F16;
    case SDL_SCANCODE_F17:
        return SDLK_F17;
    case SDL_SCANCODE_F18:
        return SDLK_F18;
    case SDL_SCANCODE_F19:
        return SDLK_F19;
    case SDL_SCANCODE_F20:
        return SDLK_F20;
    case SDL_SCANCODE_F21:
        return SDLK_F21;
    case SDL_SCANCODE_F22:
        return SDLK_F22;
    case SDL_SCANCODE_F23:
        return SDLK_F23;
    case SDL_SCANCODE_F24:
        return SDLK_F24;
    case SDL_SCANCODE_EXECUTE:
        return SDLK_EXECUTE;
    case SDL_SCANCODE_HELP:
        return SDLK_HELP;
    case SDL_SCANCODE_MENU:
        return SDLK_MENU;
    case SDL_SCANCODE_SELECT:
        return SDLK_SELECT;
    case SDL_SCANCODE_STOP:
        return SDLK_STOP;
    case SDL_SCANCODE_AGAIN:
        return SDLK_AGAIN;
    case SDL_SCANCODE_UNDO:
        return SDLK_UNDO;
    case SDL_SCANCODE_CUT:
        return SDLK_CUT;
    case SDL_SCANCODE_COPY:
        return SDLK_COPY;
    case SDL_SCANCODE_PASTE:
        return SDLK_PASTE;
    case SDL_SCANCODE_FIND:
        return SDLK_FIND;
    case SDL_SCANCODE_MUTE:
        return SDLK_MUTE;
    case SDL_SCANCODE_VOLUMEUP:
        return SDLK_VOLUMEUP;
    case SDL_SCANCODE_VOLUMEDOWN:
        return SDLK_VOLUMEDOWN;
    case SDL_SCANCODE_KP_COMMA:
        return SDLK_KP_COMMA;
    case SDL_SCANCODE_KP_EQUALSAS400:
        return SDLK_KP_EQUALSAS400;
    case SDL_SCANCODE_ALTERASE:
        return SDLK_ALTERASE;
    case SDL_SCANCODE_SYSREQ:
        return SDLK_SYSREQ;
    case SDL_SCANCODE_CANCEL:
        return SDLK_CANCEL;
    case SDL_SCANCODE_CLEAR:
        return SDLK_CLEAR;
    case SDL_SCANCODE_PRIOR:
        return SDLK_PRIOR;
    case SDL_SCANCODE_RETURN2:
        return SDLK_RETURN2;
    case SDL_SCANCODE_SEPARATOR:
        return SDLK_SEPARATOR;
    case SDL_SCANCODE_OUT:
        return SDLK_OUT;
    case SDL_SCANCODE_OPER:
        return SDLK_OPER;
    case SDL_SCANCODE_CLEARAGAIN:
        return SDLK_CLEARAGAIN;
    case SDL_SCANCODE_CRSEL:
        return SDLK_CRSEL;
    case SDL_SCANCODE_EXSEL:
        return SDLK_EXSEL;
    case SDL_SCANCODE_KP_00:
        return SDLK_KP_00;
    case SDL_SCANCODE_KP_000:
        return SDLK_KP_000;
    case SDL_SCANCODE_THOUSANDSSEPARATOR:
        return SDLK_THOUSANDSSEPARATOR;
    case SDL_SCANCODE_DECIMALSEPARATOR:
        return SDLK_DECIMALSEPARATOR;
    case SDL_SCANCODE_CURRENCYUNIT:
        return SDLK_CURRENCYUNIT;
    case SDL_SCANCODE_CURRENCYSUBUNIT:
        return SDLK_CURRENCYSUBUNIT;
    case SDL_SCANCODE_KP_LEFTPAREN:
        return SDLK_KP_LEFTPAREN;
    case SDL_SCANCODE_KP_RIGHTPAREN:
        return SDLK_KP_RIGHTPAREN;
    case SDL_SCANCODE_KP_LEFTBRACE:
        return SDLK_KP_LEFTBRACE;
    case SDL_SCANCODE_KP_RIGHTBRACE:
        return SDLK_KP_RIGHTBRACE;
    case SDL_SCANCODE_KP_TAB:
        return SDLK_KP_TAB;
    case SDL_SCANCODE_KP_BACKSPACE:
        return SDLK_KP_BACKSPACE;
    case SDL_SCANCODE_KP_A:
        return SDLK_KP_A;
    case SDL_SCANCODE_KP_B:
        return SDLK_KP_B;
    case SDL_SCANCODE_KP_C:
        return SDLK_KP_C;
    case SDL_SCANCODE_KP_D:
        return SDLK_KP_D;
    case SDL_SCANCODE_KP_E:
        return SDLK_KP_E;
    case SDL_SCANCODE_KP_F:
        return SDLK_KP_F;
    case SDL_SCANCODE_KP_XOR:
        return SDLK_KP_XOR;
    case SDL_SCANCODE_KP_POWER:
        return SDLK_KP_POWER;
    case SDL_SCANCODE_KP_PERCENT:
        return SDLK_KP_PERCENT;
    case SDL_SCANCODE_KP_LESS:
        return SDLK_KP_LESS;
    case SDL_SCANCODE_KP_GREATER:
        return SDLK_KP_GREATER;
    case SDL_SCANCODE_KP_AMPERSAND:
        return SDLK_KP_AMPERSAND;
    case SDL_SCANCODE_KP_DBLAMPERSAND:
        return SDLK_KP_DBLAMPERSAND;
    case SDL_SCANCODE_KP_VERTICALBAR:
        return SDLK_KP_VERTICALBAR;
    case SDL_SCANCODE_KP_DBLVERTICALBAR:
        return SDLK_KP_DBLVERTICALBAR;
    case SDL_SCANCODE_KP_COLON:
        return SDLK_KP_COLON;
    case SDL_SCANCODE_KP_HASH:
        return SDLK_KP_HASH;
    case SDL_SCANCODE_KP_SPACE:
        return SDLK_KP_SPACE;
    case SDL_SCANCODE_KP_AT:
        return SDLK_KP_AT;
    case SDL_SCANCODE_KP_EXCLAM:
        return SDLK_KP_EXCLAM;
    case SDL_SCANCODE_KP_MEMSTORE:
        return SDLK_KP_MEMSTORE;
    case SDL_SCANCODE_KP_MEMRECALL:
        return SDLK_KP_MEMRECALL;
    case SDL_SCANCODE_KP_MEMCLEAR:
        return SDLK_KP_MEMCLEAR;
    case SDL_SCANCODE_KP_MEMADD:
        return SDLK_KP_MEMADD;
    case SDL_SCANCODE_KP_MEMSUBTRACT:
        return SDLK_KP_MEMSUBTRACT;
    case SDL_SCANCODE_KP_MEMMULTIPLY:
        return SDLK_KP_MEMMULTIPLY;
    case SDL_SCANCODE_KP_MEMDIVIDE:
        return SDLK_KP_MEMDIVIDE;
    case SDL_SCANCODE_KP_PLUSMINUS:
        return SDLK_KP_PLUSMINUS;
    case SDL_SCANCODE_KP_CLEAR:
        return SDLK_KP_CLEAR;
    case SDL_SCANCODE_KP_CLEARENTRY:
        return SDLK_KP_CLEARENTRY;
    case SDL_SCANCODE_KP_BINARY:
        return SDLK_KP_BINARY;
    case SDL_SCANCODE_KP_OCTAL:
        return SDLK_KP_OCTAL;
    case SDL_SCANCODE_KP_DECIMAL:
        return SDLK_KP_DECIMAL;
    case SDL_SCANCODE_KP_HEXADECIMAL:
        return SDLK_KP_HEXADECIMAL;
    case SDL_SCANCODE_LCTRL:
        return SDLK_LCTRL;
    case SDL_SCANCODE_LSHIFT:
        return SDLK_LSHIFT;
    case SDL_SCANCODE_LALT:
        return SDLK_LALT;
    case SDL_SCANCODE_LGUI:
        return SDLK_LGUI;
    case SDL_SCANCODE_RCTRL:
        return SDLK_RCTRL;
    case SDL_SCANCODE_RSHIFT:
        return SDLK_RSHIFT;
    case SDL_SCANCODE_RALT:
        return SDLK_RALT;
    case SDL_SCANCODE_RGUI:
        return SDLK_RGUI;
    case SDL_SCANCODE_MODE:
        return SDLK_MODE;
    case SDL_SCANCODE_SLEEP:
        return SDLK_SLEEP;
    case SDL_SCANCODE_WAKE:
        return SDLK_WAKE;
    case SDL_SCANCODE_CHANNEL_INCREMENT:
        return SDLK_CHANNEL_INCREMENT;
    case SDL_SCANCODE_CHANNEL_DECREMENT:
        return SDLK_CHANNEL_DECREMENT;
    case SDL_SCANCODE_MEDIA_PLAY:
        return SDLK_MEDIA_PLAY;
    case SDL_SCANCODE_MEDIA_PAUSE:
        return SDLK_MEDIA_PAUSE;
    case SDL_SCANCODE_MEDIA_RECORD:
        return SDLK_MEDIA_RECORD;
    case SDL_SCANCODE_MEDIA_FAST_FORWARD:
        return SDLK_MEDIA_FAST_FORWARD;
    case SDL_SCANCODE_MEDIA_REWIND:
        return SDLK_MEDIA_REWIND;
    case SDL_SCANCODE_MEDIA_NEXT_TRACK:
        return SDLK_MEDIA_NEXT_TRACK;
    case SDL_SCANCODE_MEDIA_PREVIOUS_TRACK:
        return SDLK_MEDIA_PREVIOUS_TRACK;
    case SDL_SCANCODE_MEDIA_STOP:
        return SDLK_MEDIA_STOP;
    case SDL_SCANCODE_MEDIA_EJECT:
        return SDLK_MEDIA_EJECT;
    case SDL_SCANCODE_MEDIA_PLAY_PAUSE:
        return SDLK_MEDIA_PLAY_PAUSE;
    case SDL_SCANCODE_MEDIA_SELECT:
        return SDLK_MEDIA_SELECT;
    case SDL_SCANCODE_AC_NEW:
        return SDLK_AC_NEW;
    case SDL_SCANCODE_AC_OPEN:
        return SDLK_AC_OPEN;
    case SDL_SCANCODE_AC_CLOSE:
        return SDLK_AC_CLOSE;
    case SDL_SCANCODE_AC_EXIT:
        return SDLK_AC_EXIT;
    case SDL_SCANCODE_AC_SAVE:
        return SDLK_AC_SAVE;
    case SDL_SCANCODE_AC_PRINT:
        return SDLK_AC_PRINT;
    case SDL_SCANCODE_AC_PROPERTIES:
        return SDLK_AC_PROPERTIES;
    case SDL_SCANCODE_AC_SEARCH:
        return SDLK_AC_SEARCH;
    case SDL_SCANCODE_AC_HOME:
        return SDLK_AC_HOME;
    case SDL_SCANCODE_AC_BACK:
        return SDLK_AC_BACK;
    case SDL_SCANCODE_AC_FORWARD:
        return SDLK_AC_FORWARD;
    case SDL_SCANCODE_AC_STOP:
        return SDLK_AC_STOP;
    case SDL_SCANCODE_AC_REFRESH:
        return SDLK_AC_REFRESH;
    case SDL_SCANCODE_AC_BOOKMARKS:
        return SDLK_AC_BOOKMARKS;
    case SDL_SCANCODE_SOFTLEFT:
        return SDLK_SOFTLEFT;
    case SDL_SCANCODE_SOFTRIGHT:
        return SDLK_SOFTRIGHT;
    case SDL_SCANCODE_CALL:
        return SDLK_CALL;
    case SDL_SCANCODE_ENDCALL:
        return SDLK_ENDCALL;
    default:
        return SDLK_UNKNOWN;
    }
}

static SDL_Scancode SDL_GetDefaultScancodeFromKey(SDL_Keycode key, SDL_Keymod *modstate)
{
    if (modstate) {
        *modstate = SDL_KMOD_NONE;
    }

    if (key == SDLK_UNKNOWN) {
        return SDL_SCANCODE_UNKNOWN;
    }

    if (key & SDLK_EXTENDED_MASK) {
        for (int i = 0; i < SDL_arraysize(extended_default_symbols); ++i) {
            if (extended_default_symbols[i].keycode == key) {
                return extended_default_symbols[i].scancode;
            }
        }

        return SDL_SCANCODE_UNKNOWN;
    }

    if (key & SDLK_SCANCODE_MASK) {
        return (SDL_Scancode)(key & ~SDLK_SCANCODE_MASK);
    }

    if (key >= SDLK_A && key <= SDLK_Z) {
        return (SDL_Scancode)(SDL_SCANCODE_A + key - SDLK_A);
    }

    if (key >= 'A' && key <= 'Z') {
        if (modstate) {
            *modstate = SDL_KMOD_SHIFT;
        }
        return (SDL_Scancode)(SDL_SCANCODE_A + key - 'A');
    }

    for (int i = 0; i < SDL_arraysize(normal_default_symbols); ++i) {
        if (key == normal_default_symbols[i]) {
            return(SDL_Scancode)(SDL_SCANCODE_1 + i);
        }
    }

    for (int i = 0; i < SDL_arraysize(shifted_default_symbols); ++i) {
        if (key == shifted_default_symbols[i]) {
            if (modstate) {
                *modstate = SDL_KMOD_SHIFT;
            }
            return(SDL_Scancode)(SDL_SCANCODE_1 + i);
        }
    }

    if (key == SDLK_DELETE) {
        return SDL_SCANCODE_DELETE;
    }

    return SDL_SCANCODE_UNKNOWN;
}

static const char *SDL_scancode_names[SDL_SCANCODE_COUNT] =
{
    /* 0 */ NULL,
    /* 1 */ NULL,
    /* 2 */ NULL,
    /* 3 */ NULL,
    /* 4 */ "A",
    /* 5 */ "B",
    /* 6 */ "C",
    /* 7 */ "D",
    /* 8 */ "E",
    /* 9 */ "F",
    /* 10 */ "G",
    /* 11 */ "H",
    /* 12 */ "I",
    /* 13 */ "J",
    /* 14 */ "K",
    /* 15 */ "L",
    /* 16 */ "M",
    /* 17 */ "N",
    /* 18 */ "O",
    /* 19 */ "P",
    /* 20 */ "Q",
    /* 21 */ "R",
    /* 22 */ "S",
    /* 23 */ "T",
    /* 24 */ "U",
    /* 25 */ "V",
    /* 26 */ "W",
    /* 27 */ "X",
    /* 28 */ "Y",
    /* 29 */ "Z",
    /* 30 */ "1",
    /* 31 */ "2",
    /* 32 */ "3",
    /* 33 */ "4",
    /* 34 */ "5",
    /* 35 */ "6",
    /* 36 */ "7",
    /* 37 */ "8",
    /* 38 */ "9",
    /* 39 */ "0",
    /* 40 */ "Return",
    /* 41 */ "Escape",
    /* 42 */ "Backspace",
    /* 43 */ "Tab",
    /* 44 */ "Space",
    /* 45 */ "-",
    /* 46 */ "=",
    /* 47 */ "[",
    /* 48 */ "]",
    /* 49 */ "\\",
    /* 50 */ "#",
    /* 51 */ ";",
    /* 52 */ "'",
    /* 53 */ "`",
    /* 54 */ ",",
    /* 55 */ ".",
    /* 56 */ "/",
    /* 57 */ "CapsLock",
    /* 58 */ "F1",
    /* 59 */ "F2",
    /* 60 */ "F3",
    /* 61 */ "F4",
    /* 62 */ "F5",
    /* 63 */ "F6",
    /* 64 */ "F7",
    /* 65 */ "F8",
    /* 66 */ "F9",
    /* 67 */ "F10",
    /* 68 */ "F11",
    /* 69 */ "F12",
    /* 70 */ "PrintScreen",
    /* 71 */ "ScrollLock",
    /* 72 */ "Pause",
    /* 73 */ "Insert",
    /* 74 */ "Home",
    /* 75 */ "PageUp",
    /* 76 */ "Delete",
    /* 77 */ "End",
    /* 78 */ "PageDown",
    /* 79 */ "Right",
    /* 80 */ "Left",
    /* 81 */ "Down",
    /* 82 */ "Up",
    /* 83 */ "Numlock",
    /* 84 */ "Keypad /",
    /* 85 */ "Keypad *",
    /* 86 */ "Keypad -",
    /* 87 */ "Keypad +",
    /* 88 */ "Keypad Enter",
    /* 89 */ "Keypad 1",
    /* 90 */ "Keypad 2",
    /* 91 */ "Keypad 3",
    /* 92 */ "Keypad 4",
    /* 93 */ "Keypad 5",
    /* 94 */ "Keypad 6",
    /* 95 */ "Keypad 7",
    /* 96 */ "Keypad 8",
    /* 97 */ "Keypad 9",
    /* 98 */ "Keypad 0",
    /* 99 */ "Keypad .",
    /* 100 */ "NonUSBackslash",
    /* 101 */ "Application",
    /* 102 */ "Power",
    /* 103 */ "Keypad =",
    /* 104 */ "F13",
    /* 105 */ "F14",
    /* 106 */ "F15",
    /* 107 */ "F16",
    /* 108 */ "F17",
    /* 109 */ "F18",
    /* 110 */ "F19",
    /* 111 */ "F20",
    /* 112 */ "F21",
    /* 113 */ "F22",
    /* 114 */ "F23",
    /* 115 */ "F24",
    /* 116 */ "Execute",
    /* 117 */ "Help",
    /* 118 */ "Menu",
    /* 119 */ "Select",
    /* 120 */ "Stop",
    /* 121 */ "Again",
    /* 122 */ "Undo",
    /* 123 */ "Cut",
    /* 124 */ "Copy",
    /* 125 */ "Paste",
    /* 126 */ "Find",
    /* 127 */ "Mute",
    /* 128 */ "VolumeUp",
    /* 129 */ "VolumeDown",
    /* 130 */ NULL,
    /* 131 */ NULL,
    /* 132 */ NULL,
    /* 133 */ "Keypad ,",
    /* 134 */ "Keypad = (AS400)",
    /* 135 */ "International 1",
    /* 136 */ "International 2",
    /* 137 */ "International 3",
    /* 138 */ "International 4",
    /* 139 */ "International 5",
    /* 140 */ "International 6",
    /* 141 */ "International 7",
    /* 142 */ "International 8",
    /* 143 */ "International 9",
    /* 144 */ "Language 1",
    /* 145 */ "Language 2",
    /* 146 */ "Language 3",
    /* 147 */ "Language 4",
    /* 148 */ "Language 5",
    /* 149 */ "Language 6",
    /* 150 */ "Language 7",
    /* 151 */ "Language 8",
    /* 152 */ "Language 9",
    /* 153 */ "AltErase",
    /* 154 */ "SysReq",
    /* 155 */ "Cancel",
    /* 156 */ "Clear",
    /* 157 */ "Prior",
    /* 158 */ "Return",
    /* 159 */ "Separator",
    /* 160 */ "Out",
    /* 161 */ "Oper",
    /* 162 */ "Clear / Again",
    /* 163 */ "CrSel",
    /* 164 */ "ExSel",
    /* 165 */ NULL,
    /* 166 */ NULL,
    /* 167 */ NULL,
    /* 168 */ NULL,
    /* 169 */ NULL,
    /* 170 */ NULL,
    /* 171 */ NULL,
    /* 172 */ NULL,
    /* 173 */ NULL,
    /* 174 */ NULL,
    /* 175 */ NULL,
    /* 176 */ "Keypad 00",
    /* 177 */ "Keypad 000",
    /* 178 */ "ThousandsSeparator",
    /* 179 */ "DecimalSeparator",
    /* 180 */ "CurrencyUnit",
    /* 181 */ "CurrencySubUnit",
    /* 182 */ "Keypad (",
    /* 183 */ "Keypad )",
    /* 184 */ "Keypad {",
    /* 185 */ "Keypad }",
    /* 186 */ "Keypad Tab",
    /* 187 */ "Keypad Backspace",
    /* 188 */ "Keypad A",
    /* 189 */ "Keypad B",
    /* 190 */ "Keypad C",
    /* 191 */ "Keypad D",
    /* 192 */ "Keypad E",
    /* 193 */ "Keypad F",
    /* 194 */ "Keypad XOR",
    /* 195 */ "Keypad ^",
    /* 196 */ "Keypad %",
    /* 197 */ "Keypad <",
    /* 198 */ "Keypad >",
    /* 199 */ "Keypad &",
    /* 200 */ "Keypad &&",
    /* 201 */ "Keypad |",
    /* 202 */ "Keypad ||",
    /* 203 */ "Keypad :",
    /* 204 */ "Keypad #",
    /* 205 */ "Keypad Space",
    /* 206 */ "Keypad @",
    /* 207 */ "Keypad !",
    /* 208 */ "Keypad MemStore",
    /* 209 */ "Keypad MemRecall",
    /* 210 */ "Keypad MemClear",
    /* 211 */ "Keypad MemAdd",
    /* 212 */ "Keypad MemSubtract",
    /* 213 */ "Keypad MemMultiply",
    /* 214 */ "Keypad MemDivide",
    /* 215 */ "Keypad +/-",
    /* 216 */ "Keypad Clear",
    /* 217 */ "Keypad ClearEntry",
    /* 218 */ "Keypad Binary",
    /* 219 */ "Keypad Octal",
    /* 220 */ "Keypad Decimal",
    /* 221 */ "Keypad Hexadecimal",
    /* 222 */ NULL,
    /* 223 */ NULL,
    /* 224 */ "Left Ctrl",
    /* 225 */ "Left Shift",
    /* 226 */ "Left Alt",
    /* 227 */ "Left GUI",
    /* 228 */ "Right Ctrl",
    /* 229 */ "Right Shift",
    /* 230 */ "Right Alt",
    /* 231 */ "Right GUI",
    /* 232 */ NULL,
    /* 233 */ NULL,
    /* 234 */ NULL,
    /* 235 */ NULL,
    /* 236 */ NULL,
    /* 237 */ NULL,
    /* 238 */ NULL,
    /* 239 */ NULL,
    /* 240 */ NULL,
    /* 241 */ NULL,
    /* 242 */ NULL,
    /* 243 */ NULL,
    /* 244 */ NULL,
    /* 245 */ NULL,
    /* 246 */ NULL,
    /* 247 */ NULL,
    /* 248 */ NULL,
    /* 249 */ NULL,
    /* 250 */ NULL,
    /* 251 */ NULL,
    /* 252 */ NULL,
    /* 253 */ NULL,
    /* 254 */ NULL,
    /* 255 */ NULL,
    /* 256 */ NULL,
    /* 257 */ "ModeSwitch",
    /* 258 */ "Sleep",
    /* 259 */ "Wake",
    /* 260 */ "ChannelUp",
    /* 261 */ "ChannelDown",
    /* 262 */ "MediaPlay",
    /* 263 */ "MediaPause",
    /* 264 */ "MediaRecord",
    /* 265 */ "MediaFastForward",
    /* 266 */ "MediaRewind",
    /* 267 */ "MediaTrackNext",
    /* 268 */ "MediaTrackPrevious",
    /* 269 */ "MediaStop",
    /* 270 */ "Eject",
    /* 271 */ "MediaPlayPause",
    /* 272 */ "MediaSelect",
    /* 273 */ "AC New",
    /* 274 */ "AC Open",
    /* 275 */ "AC Close",
    /* 276 */ "AC Exit",
    /* 277 */ "AC Save",
    /* 278 */ "AC Print",
    /* 279 */ "AC Properties",
    /* 280 */ "AC Search",
    /* 281 */ "AC Home",
    /* 282 */ "AC Back",
    /* 283 */ "AC Forward",
    /* 284 */ "AC Stop",
    /* 285 */ "AC Refresh",
    /* 286 */ "AC Bookmarks",
    /* 287 */ "SoftLeft",
    /* 288 */ "SoftRight",
    /* 289 */ "Call",
    /* 290 */ "EndCall",
};

static const char *SDL_extended_key_names[] = {
    "LeftTab",         /* 0x01 SDLK_LEFT_TAB */
    "Level5Shift",     /* 0x02 SDLK_LEVEL5_SHIFT */
    "MultiKeyCompose", /* 0x03 SDLK_MULTI_KEY_COMPOSE */
    "Left Meta",       /* 0x04 SDLK_LMETA */
    "Right Meta",      /* 0x05 SDLK_RMETA */
    "Left Hyper",      /* 0x06 SDLK_LHYPER */
    "Right Hyper"      /* 0x07 SDLK_RHYPER */
};

bool SDL_SetScancodeName(SDL_Scancode scancode, const char *name)
{
    if (((int)scancode) < SDL_SCANCODE_UNKNOWN || scancode >= SDL_SCANCODE_COUNT) {
        return SDL_InvalidParamError("scancode");
    }

    SDL_scancode_names[scancode] = name;
    return true;
}

const char *SDL_GetScancodeName(SDL_Scancode scancode)
{
    const char *name;
    if (((int)scancode) < SDL_SCANCODE_UNKNOWN || scancode >= SDL_SCANCODE_COUNT) {
        SDL_InvalidParamError("scancode");
        return "";
    }

    name = SDL_scancode_names[scancode];
    if (!name) {
        name = "";
    }
    // This is pointing to static memory or application managed memory
    return name;
}

SDL_Scancode SDL_GetScancodeFromName(const char *name)
{
    int i;

    if (!name || !*name) {
        SDL_InvalidParamError("name");
        return SDL_SCANCODE_UNKNOWN;
    }

    for (i = 0; i < SDL_arraysize(SDL_scancode_names); ++i) {
        if (!SDL_scancode_names[i]) {
            continue;
        }
        if (SDL_strcasecmp(name, SDL_scancode_names[i]) == 0) {
            return (SDL_Scancode)i;
        }
    }

    SDL_InvalidParamError("name");
    return SDL_SCANCODE_UNKNOWN;
}

const char *SDL_GetKeyName(SDL_Keycode key)
{
    const bool uppercase = true;
    char name[8];
    char *end;

    if (key & SDLK_SCANCODE_MASK) {
        return SDL_GetScancodeName((SDL_Scancode)(key & ~SDLK_SCANCODE_MASK));
    }

    if (key & SDLK_EXTENDED_MASK) {
        const SDL_Keycode idx = (key & ~SDLK_EXTENDED_MASK);
        if (idx > 0 && (idx - 1) < SDL_arraysize(SDL_extended_key_names)) {
            return SDL_extended_key_names[idx - 1];
        }

        // Key out of name index bounds.
        SDL_InvalidParamError("key");
        return "";
    }

    switch (key) {
    case SDLK_RETURN:
        return SDL_GetScancodeName(SDL_SCANCODE_RETURN);
    case SDLK_ESCAPE:
        return SDL_GetScancodeName(SDL_SCANCODE_ESCAPE);
    case SDLK_BACKSPACE:
        return SDL_GetScancodeName(SDL_SCANCODE_BACKSPACE);
    case SDLK_TAB:
        return SDL_GetScancodeName(SDL_SCANCODE_TAB);
    case SDLK_SPACE:
        return SDL_GetScancodeName(SDL_SCANCODE_SPACE);
    case SDLK_DELETE:
        return SDL_GetScancodeName(SDL_SCANCODE_DELETE);
    default:
        if (uppercase) {
            // SDL_Keycode is defined as the unshifted key on the keyboard,
            // but the key name is defined as the letter printed on that key,
            // which is usually the shifted capital letter.
            if (key > 0x7F || (key >= 'a' && key <= 'z')) {
                SDL_Keymap *keymap = SDL_GetCurrentKeymap();
                SDL_Keymod modstate;
                SDL_Scancode scancode = SDL_GetKeymapScancode(keymap, key, &modstate);
                if (scancode != SDL_SCANCODE_UNKNOWN && !(modstate & SDL_KMOD_SHIFT)) {
                    SDL_Keycode capital = SDL_GetKeymapKeycode(keymap, scancode, SDL_KMOD_SHIFT);
                    if (capital > 0x7F || (capital >= 'A' && capital <= 'Z')) {
                        key = capital;
                    }
                }
            }
        }

        end = SDL_UCS4ToUTF8(key, name);
        *end = '\0';
        return SDL_GetPersistentString(name);
    }
}

SDL_Keycode SDL_GetKeyFromName(const char *name)
{
    const bool uppercase = true;
    SDL_Keycode key;

    // Check input
    if (!name) {
        return SDLK_UNKNOWN;
    }

    // If it's a single UTF-8 character, then that's the keycode itself
    key = *(const unsigned char *)name;
    if (key >= 0xF0) {
        if (SDL_strlen(name) == 4) {
            int i = 0;
            key = (Uint16)(name[i] & 0x07) << 18;
            key |= (Uint16)(name[++i] & 0x3F) << 12;
            key |= (Uint16)(name[++i] & 0x3F) << 6;
            key |= (Uint16)(name[++i] & 0x3F);
        } else {
            key = SDLK_UNKNOWN;
        }
    } else if (key >= 0xE0) {
        if (SDL_strlen(name) == 3) {
            int i = 0;
            key = (Uint16)(name[i] & 0x0F) << 12;
            key |= (Uint16)(name[++i] & 0x3F) << 6;
            key |= (Uint16)(name[++i] & 0x3F);
        } else {
            key = SDLK_UNKNOWN;
        }
    } else if (key >= 0xC0) {
        if (SDL_strlen(name) == 2) {
            int i = 0;
            key = (Uint16)(name[i] & 0x1F) << 6;
            key |= (Uint16)(name[++i] & 0x3F);
        } else {
            key = SDLK_UNKNOWN;
        }
    } else {
        if (SDL_strlen(name) != 1) {
            key = SDLK_UNKNOWN;
        }
    }

    if (key != SDLK_UNKNOWN) {
        if (uppercase) {
            // SDL_Keycode is defined as the unshifted key on the keyboard,
            // but the key name is defined as the letter printed on that key,
            // which is usually the shifted capital letter.
            SDL_Keymap *keymap = SDL_GetCurrentKeymap();
            SDL_Keymod modstate;
            SDL_Scancode scancode = SDL_GetKeymapScancode(keymap, key, &modstate);
            if (scancode != SDL_SCANCODE_UNKNOWN && (modstate & (SDL_KMOD_SHIFT | SDL_KMOD_CAPS))) {
                key = SDL_GetKeymapKeycode(keymap, scancode, SDL_KMOD_NONE);
            }
        }
        return key;
    }

    // Check the extended key names
    for (SDL_Keycode i = 0; i < SDL_arraysize(SDL_extended_key_names); ++i) {
        if (SDL_strcasecmp(name, SDL_extended_key_names[i]) == 0) {
            return (i + 1) | SDLK_EXTENDED_MASK;
        }
    }

    return SDL_GetKeyFromScancode(SDL_GetScancodeFromName(name), SDL_KMOD_NONE, false);
}
