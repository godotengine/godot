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

// General keyboard handling code for SDL

#include "SDL_events_c.h"
#include "SDL_keymap_c.h"
#include "../video/SDL_sysvideo.h"

#if 0
#define DEBUG_KEYBOARD
#endif

// Global keyboard information

#define KEYBOARD_HARDWARE        0x01
#define KEYBOARD_VIRTUAL         0x02
#define KEYBOARD_AUTORELEASE     0x04
#define KEYBOARD_IGNOREMODIFIERS 0x08

#define KEYBOARD_SOURCE_MASK (KEYBOARD_HARDWARE | KEYBOARD_AUTORELEASE)

#define KEYCODE_OPTION_HIDE_NUMPAD      0x01
#define KEYCODE_OPTION_FRENCH_NUMBERS   0x02
#define KEYCODE_OPTION_LATIN_LETTERS    0x04
#define DEFAULT_KEYCODE_OPTIONS         (KEYCODE_OPTION_FRENCH_NUMBERS | KEYCODE_OPTION_LATIN_LETTERS)

typedef struct SDL_KeyboardInstance
{
    SDL_KeyboardID instance_id;
    char *name;
} SDL_KeyboardInstance;

typedef struct SDL_Keyboard
{
    // Data common to all keyboards
    SDL_Window *focus;
    SDL_Keymod modstate;
    Uint8 keysource[SDL_SCANCODE_COUNT];
    bool keystate[SDL_SCANCODE_COUNT];
    SDL_Keymap *keymap;
    Uint32 keycode_options;
    bool autorelease_pending;
    Uint64 hardware_timestamp;
    int next_reserved_scancode;
} SDL_Keyboard;

static SDL_Keyboard SDL_keyboard;
static int SDL_keyboard_count;
static SDL_KeyboardInstance *SDL_keyboards;

static void SDLCALL SDL_KeycodeOptionsChanged(void *userdata, const char *name, const char *oldValue, const char *hint)
{
    SDL_Keyboard *keyboard = (SDL_Keyboard *)userdata;

    if (hint && *hint) {
        keyboard->keycode_options = 0;
        if (!SDL_strstr(hint, "none")) {
            if (SDL_strstr(hint, "hide_numpad")) {
                keyboard->keycode_options |= KEYCODE_OPTION_HIDE_NUMPAD;
            }
            if (SDL_strstr(hint, "french_numbers")) {
                keyboard->keycode_options |= KEYCODE_OPTION_FRENCH_NUMBERS;
            }
            if (SDL_strstr(hint, "latin_letters")) {
                keyboard->keycode_options |= KEYCODE_OPTION_LATIN_LETTERS;
            }
        }
    } else {
        keyboard->keycode_options = DEFAULT_KEYCODE_OPTIONS;
    }
}

// Public functions
bool SDL_InitKeyboard(void)
{
    SDL_AddHintCallback(SDL_HINT_KEYCODE_OPTIONS,
                        SDL_KeycodeOptionsChanged, &SDL_keyboard);
    return true;
}

bool SDL_IsKeyboard(Uint16 vendor, Uint16 product, int num_keys)
{
    const int REAL_KEYBOARD_KEY_COUNT = 50;
    if (num_keys > 0 && num_keys < REAL_KEYBOARD_KEY_COUNT) {
        return false;
    }

    // Eventually we'll have a blacklist of devices that enumerate as keyboards but aren't really
    return true;
}

static int SDL_GetKeyboardIndex(SDL_KeyboardID keyboardID)
{
    for (int i = 0; i < SDL_keyboard_count; ++i) {
        if (keyboardID == SDL_keyboards[i].instance_id) {
            return i;
        }
    }
    return -1;
}

void SDL_AddKeyboard(SDL_KeyboardID keyboardID, const char *name, bool send_event)
{
    int keyboard_index = SDL_GetKeyboardIndex(keyboardID);
    if (keyboard_index >= 0) {
        // We already know about this keyboard
        return;
    }

    SDL_assert(keyboardID != 0);

    SDL_KeyboardInstance *keyboards = (SDL_KeyboardInstance *)SDL_realloc(SDL_keyboards, (SDL_keyboard_count + 1) * sizeof(*keyboards));
    if (!keyboards) {
        return;
    }
    SDL_KeyboardInstance *instance = &keyboards[SDL_keyboard_count];
    instance->instance_id = keyboardID;
    instance->name = SDL_strdup(name ? name : "");
    SDL_keyboards = keyboards;
    ++SDL_keyboard_count;

    if (send_event) {
        SDL_Event event;
        SDL_zero(event);
        event.type = SDL_EVENT_KEYBOARD_ADDED;
        event.kdevice.which = keyboardID;
        SDL_PushEvent(&event);
    }
}

void SDL_RemoveKeyboard(SDL_KeyboardID keyboardID, bool send_event)
{
    int keyboard_index = SDL_GetKeyboardIndex(keyboardID);
    if (keyboard_index < 0) {
        // We don't know about this keyboard
        return;
    }

    SDL_free(SDL_keyboards[keyboard_index].name);

    if (keyboard_index != SDL_keyboard_count - 1) {
        SDL_memmove(&SDL_keyboards[keyboard_index], &SDL_keyboards[keyboard_index + 1], (SDL_keyboard_count - keyboard_index - 1) * sizeof(SDL_keyboards[keyboard_index]));
    }
    --SDL_keyboard_count;

    if (send_event) {
        SDL_Event event;
        SDL_zero(event);
        event.type = SDL_EVENT_KEYBOARD_REMOVED;
        event.kdevice.which = keyboardID;
        SDL_PushEvent(&event);
    }
}

void SDL_SetKeyboardName(SDL_KeyboardID keyboardID, const char *name)
{
    SDL_assert(keyboardID != 0);

    const int keyboard_index = SDL_GetKeyboardIndex(keyboardID);

    if (keyboard_index >= 0) {
        SDL_KeyboardInstance *instance = &SDL_keyboards[keyboard_index];
        SDL_free(instance->name);
        instance->name = SDL_strdup(name ? name : "");
    }
}

bool SDL_HasKeyboard(void)
{
    return (SDL_keyboard_count > 0);
}

SDL_KeyboardID *SDL_GetKeyboards(int *count)
{
    int i;
    SDL_KeyboardID *keyboards;

    keyboards = (SDL_JoystickID *)SDL_malloc((SDL_keyboard_count + 1) * sizeof(*keyboards));
    if (keyboards) {
        if (count) {
            *count = SDL_keyboard_count;
        }

        for (i = 0; i < SDL_keyboard_count; ++i) {
            keyboards[i] = SDL_keyboards[i].instance_id;
        }
        keyboards[i] = 0;
    } else {
        if (count) {
            *count = 0;
        }
    }

    return keyboards;
}

const char *SDL_GetKeyboardNameForID(SDL_KeyboardID instance_id)
{
    int keyboard_index = SDL_GetKeyboardIndex(instance_id);
    if (keyboard_index < 0) {
        SDL_SetError("Keyboard %" SDL_PRIu32 " not found", instance_id);
        return NULL;
    }
    return SDL_GetPersistentString(SDL_keyboards[keyboard_index].name);
}

void SDL_ResetKeyboard(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    int scancode;

#ifdef DEBUG_KEYBOARD
    SDL_Log("Resetting keyboard");
#endif
    for (scancode = SDL_SCANCODE_UNKNOWN; scancode < SDL_SCANCODE_COUNT; ++scancode) {
        if (keyboard->keystate[scancode]) {
            SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, (SDL_Scancode)scancode, false);
        }
    }
}

SDL_Keymap *SDL_GetCurrentKeymap(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Keymap *keymap = SDL_keyboard.keymap;

    if (keymap && keymap->thai_keyboard) {
        // Thai keyboards are QWERTY plus Thai characters, use the default QWERTY keymap
        return NULL;
    }

    if ((keyboard->keycode_options & KEYCODE_OPTION_LATIN_LETTERS) &&
        keymap && !keymap->latin_letters) {
        // We'll use the default QWERTY keymap
        return NULL;
    }

    return keyboard->keymap;
}

void SDL_SetKeymap(SDL_Keymap *keymap, bool send_event)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (keyboard->keymap && keyboard->keymap->auto_release) {
        SDL_DestroyKeymap(keyboard->keymap);
    }

    keyboard->keymap = keymap;

    if (keymap && !keymap->layout_determined) {
        keymap->layout_determined = true;

        // Detect French number row (all symbols)
        keymap->french_numbers = true;
        for (int i = SDL_SCANCODE_1; i <= SDL_SCANCODE_0; ++i) {
            if (SDL_isdigit(SDL_GetKeymapKeycode(keymap, (SDL_Scancode)i, SDL_KMOD_NONE)) ||
                !SDL_isdigit(SDL_GetKeymapKeycode(keymap, (SDL_Scancode)i, SDL_KMOD_SHIFT))) {
                keymap->french_numbers = false;
                break;
                }
        }

        // Detect non-Latin keymap
        keymap->thai_keyboard = false;
        keymap->latin_letters = false;
        for (int i = SDL_SCANCODE_A; i <= SDL_SCANCODE_D; ++i) {
            SDL_Keycode key = SDL_GetKeymapKeycode(keymap, (SDL_Scancode)i, SDL_KMOD_NONE);
            if (key <= 0xFF) {
                keymap->latin_letters = true;
                break;
            }

            if (key >= 0x0E00 && key <= 0x0E7F) {
                keymap->thai_keyboard = true;
                break;
            }
        }
    }

    if (send_event) {
        SDL_SendKeymapChangedEvent();
    }
}

static SDL_Scancode GetNextReservedScancode(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Scancode scancode;

    if (keyboard->next_reserved_scancode && keyboard->next_reserved_scancode < SDL_SCANCODE_RESERVED + 100) {
        scancode = (SDL_Scancode)keyboard->next_reserved_scancode;
    } else {
        scancode = SDL_SCANCODE_RESERVED;
    }
    keyboard->next_reserved_scancode = (int)scancode + 1;

    return scancode;
}

static void SetKeymapEntry(SDL_Scancode scancode, SDL_Keymod modstate, SDL_Keycode keycode)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (!keyboard->keymap) {
        keyboard->keymap = SDL_CreateKeymap(true);
    }

    SDL_SetKeymapEntry(keyboard->keymap, scancode, modstate, keycode);
}

SDL_Window *SDL_GetKeyboardFocus(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    return keyboard->focus;
}

bool SDL_SetKeyboardFocus(SDL_Window *window)
{
    SDL_VideoDevice *video = SDL_GetVideoDevice();
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Mouse *mouse = SDL_GetMouse();

    if (window) {
        if (!SDL_ObjectValid(window, SDL_OBJECT_TYPE_WINDOW) || window->is_destroying) {
            return SDL_SetError("Invalid window");
        }
    }

    if (keyboard->focus && !window) {
        // We won't get anymore keyboard messages, so reset keyboard state
        SDL_ResetKeyboard();

        // Also leave mouse relative mode
        if (mouse->relative_mode) {
            SDL_SetRelativeMouseMode(false);

            SDL_Window *focus = keyboard->focus;
            if ((focus->flags & SDL_WINDOW_MINIMIZED) != 0) {
                // We can't warp the mouse within minimized windows, so manually restore the position
                float x = focus->x + mouse->x;
                float y = focus->y + mouse->y;
                SDL_WarpMouseGlobal(x, y);
            }
        }
    }

    // See if the current window has lost focus
    if (keyboard->focus && keyboard->focus != window) {
        SDL_SendWindowEvent(keyboard->focus, SDL_EVENT_WINDOW_FOCUS_LOST, 0, 0);

        // Ensures IME compositions are committed
        if (SDL_TextInputActive(keyboard->focus)) {
            if (video && video->StopTextInput) {
                video->StopTextInput(video, keyboard->focus);
            }
        }
    }

    keyboard->focus = window;

    if (keyboard->focus) {
        SDL_SendWindowEvent(keyboard->focus, SDL_EVENT_WINDOW_FOCUS_GAINED, 0, 0);

        if (SDL_TextInputActive(keyboard->focus)) {
            if (video && video->StartTextInput) {
                video->StartTextInput(video, keyboard->focus, keyboard->focus->text_input_props);
            }
        }
    }

    SDL_UpdateRelativeMouseMode();

    return true;
}

static SDL_Keycode SDL_ConvertNumpadKeycode(SDL_Keycode keycode, bool numlock)
{
    switch (keycode) {
    case SDLK_KP_DIVIDE:
        return SDLK_SLASH;
    case SDLK_KP_MULTIPLY:
        return SDLK_ASTERISK;
    case SDLK_KP_MINUS:
        return SDLK_MINUS;
    case SDLK_KP_PLUS:
        return SDLK_PLUS;
    case SDLK_KP_ENTER:
        return SDLK_RETURN;
    case SDLK_KP_1:
        return numlock ? SDLK_1 : SDLK_END;
    case SDLK_KP_2:
        return numlock ? SDLK_2 : SDLK_DOWN;
    case SDLK_KP_3:
        return numlock ? SDLK_3 : SDLK_PAGEDOWN;
    case SDLK_KP_4:
        return numlock ? SDLK_4 : SDLK_LEFT;
    case SDLK_KP_5:
        return numlock ? SDLK_5 : SDLK_CLEAR;
    case SDLK_KP_6:
        return numlock ? SDLK_6 : SDLK_RIGHT;
    case SDLK_KP_7:
        return numlock ? SDLK_7 : SDLK_HOME;
    case SDLK_KP_8:
        return numlock ? SDLK_8 : SDLK_UP;
    case SDLK_KP_9:
        return numlock ? SDLK_9 : SDLK_PAGEUP;
    case SDLK_KP_0:
        return numlock ? SDLK_0 : SDLK_INSERT;
    case SDLK_KP_PERIOD:
        return numlock ? SDLK_PERIOD : SDLK_DELETE;
    case SDLK_KP_EQUALS:
        return SDLK_EQUALS;
    case SDLK_KP_COMMA:
        return SDLK_COMMA;
    case SDLK_KP_EQUALSAS400:
        return SDLK_EQUALS;
    case SDLK_KP_LEFTPAREN:
        return SDLK_LEFTPAREN;
    case SDLK_KP_RIGHTPAREN:
        return SDLK_RIGHTPAREN;
    case SDLK_KP_LEFTBRACE:
        return SDLK_LEFTBRACE;
    case SDLK_KP_RIGHTBRACE:
        return SDLK_RIGHTBRACE;
    case SDLK_KP_TAB:
        return SDLK_TAB;
    case SDLK_KP_BACKSPACE:
        return SDLK_BACKSPACE;
    case SDLK_KP_A:
        return SDLK_A;
    case SDLK_KP_B:
        return SDLK_B;
    case SDLK_KP_C:
        return SDLK_C;
    case SDLK_KP_D:
        return SDLK_D;
    case SDLK_KP_E:
        return SDLK_E;
    case SDLK_KP_F:
        return SDLK_F;
    case SDLK_KP_PERCENT:
        return SDLK_PERCENT;
    case SDLK_KP_LESS:
        return SDLK_LESS;
    case SDLK_KP_GREATER:
        return SDLK_GREATER;
    case SDLK_KP_AMPERSAND:
        return SDLK_AMPERSAND;
    case SDLK_KP_COLON:
        return SDLK_COLON;
    case SDLK_KP_HASH:
        return SDLK_HASH;
    case SDLK_KP_SPACE:
        return SDLK_SPACE;
    case SDLK_KP_AT:
        return SDLK_AT;
    case SDLK_KP_EXCLAM:
        return SDLK_EXCLAIM;
    case SDLK_KP_PLUSMINUS:
        return SDLK_PLUSMINUS;
    default:
        return keycode;
    }
}

SDL_Keycode SDL_GetKeyFromScancode(SDL_Scancode scancode, SDL_Keymod modstate, bool key_event)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (key_event) {
        SDL_Keymap *keymap = SDL_GetCurrentKeymap();
        bool numlock = (modstate & SDL_KMOD_NUM) != 0;
        SDL_Keycode keycode;

        // We won't be applying any modifiers by default
        modstate = SDL_KMOD_NONE;

        if ((keyboard->keycode_options & KEYCODE_OPTION_FRENCH_NUMBERS) &&
            keymap && keymap->french_numbers &&
            (scancode >= SDL_SCANCODE_1 && scancode <= SDL_SCANCODE_0)) {
            // Add the shift state to generate a numeric keycode
            modstate |= SDL_KMOD_SHIFT;
        }

        keycode = SDL_GetKeymapKeycode(keymap, scancode, modstate);

        if (keyboard->keycode_options & KEYCODE_OPTION_HIDE_NUMPAD) {
            keycode = SDL_ConvertNumpadKeycode(keycode, numlock);
        }
        return keycode;
    }

    return SDL_GetKeymapKeycode(keyboard->keymap, scancode, modstate);
}

SDL_Scancode SDL_GetScancodeFromKey(SDL_Keycode key, SDL_Keymod *modstate)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    return SDL_GetKeymapScancode(keyboard->keymap, key, modstate);
}

static bool SDL_SendKeyboardKeyInternal(Uint64 timestamp, Uint32 flags, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, bool down)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    bool posted = false;
    SDL_Keycode keycode = SDLK_UNKNOWN;
    Uint32 type;
    bool repeat = false;
    const Uint8 source = flags & KEYBOARD_SOURCE_MASK;

#ifdef DEBUG_KEYBOARD
    SDL_Log("The '%s' key has been %s", SDL_GetScancodeName(scancode), down ? "pressed" : "released");
#endif

    // Figure out what type of event this is
    if (down) {
        type = SDL_EVENT_KEY_DOWN;
    } else {
        type = SDL_EVENT_KEY_UP;
    }

    if (scancode > SDL_SCANCODE_UNKNOWN && scancode < SDL_SCANCODE_COUNT) {
        // Drop events that don't change state
        if (down) {
            if (keyboard->keystate[scancode]) {
                if (!(keyboard->keysource[scancode] & source)) {
                    keyboard->keysource[scancode] |= source;
                    return false;
                }
                repeat = true;
            }
            keyboard->keysource[scancode] |= source;
        } else {
            if (!keyboard->keystate[scancode]) {
                return false;
            }
            keyboard->keysource[scancode] = 0;
        }

        // Update internal keyboard state
        keyboard->keystate[scancode] = down;

        keycode = SDL_GetKeyFromScancode(scancode, keyboard->modstate, true);

    } else if (rawcode == 0) {
        // Nothing to do!
        return false;
    }

    if (source == KEYBOARD_HARDWARE) {
        keyboard->hardware_timestamp = SDL_GetTicks();
    } else if (source == KEYBOARD_AUTORELEASE) {
        keyboard->autorelease_pending = true;
    }

    // Update modifiers state if applicable
    if (!(flags & KEYBOARD_IGNOREMODIFIERS) && !repeat) {
        SDL_Keymod modifier;

        switch (keycode) {
        case SDLK_LCTRL:
            modifier = SDL_KMOD_LCTRL;
            break;
        case SDLK_RCTRL:
            modifier = SDL_KMOD_RCTRL;
            break;
        case SDLK_LSHIFT:
            modifier = SDL_KMOD_LSHIFT;
            break;
        case SDLK_RSHIFT:
            modifier = SDL_KMOD_RSHIFT;
            break;
        case SDLK_LALT:
            modifier = SDL_KMOD_LALT;
            break;
        case SDLK_RALT:
            modifier = SDL_KMOD_RALT;
            break;
        case SDLK_LGUI:
            modifier = SDL_KMOD_LGUI;
            break;
        case SDLK_RGUI:
            modifier = SDL_KMOD_RGUI;
            break;
        case SDLK_MODE:
            modifier = SDL_KMOD_MODE;
            break;
        default:
            modifier = SDL_KMOD_NONE;
            break;
        }
        if (SDL_EVENT_KEY_DOWN == type) {
            switch (keycode) {
            case SDLK_NUMLOCKCLEAR:
                keyboard->modstate ^= SDL_KMOD_NUM;
                break;
            case SDLK_CAPSLOCK:
                keyboard->modstate ^= SDL_KMOD_CAPS;
                break;
            case SDLK_SCROLLLOCK:
                keyboard->modstate ^= SDL_KMOD_SCROLL;
                break;
            default:
                keyboard->modstate |= modifier;
                break;
            }
        } else {
            keyboard->modstate &= ~modifier;
        }
    }

    // Post the event, if desired
    if (SDL_EventEnabled(type)) {
        SDL_Event event;
        event.type = type;
        event.common.timestamp = timestamp;
        event.key.scancode = scancode;
        event.key.key = keycode;
        event.key.mod = keyboard->modstate;
        event.key.raw = (Uint16)rawcode;
        event.key.down = down;
        event.key.repeat = repeat;
        event.key.windowID = keyboard->focus ? keyboard->focus->id : 0;
        event.key.which = keyboardID;
        posted = SDL_PushEvent(&event);
    }

    /* If the keyboard is grabbed and the grabbed window is in full-screen,
       minimize the window when we receive Alt+Tab, unless the application
       has explicitly opted out of this behavior. */
    if (keycode == SDLK_TAB && down &&
        (keyboard->modstate & SDL_KMOD_ALT) &&
        keyboard->focus &&
        (keyboard->focus->flags & SDL_WINDOW_KEYBOARD_GRABBED) &&
        (keyboard->focus->flags & SDL_WINDOW_FULLSCREEN) &&
        SDL_GetHintBoolean(SDL_HINT_ALLOW_ALT_TAB_WHILE_GRABBED, true)) {
        /* We will temporarily forfeit our grab by minimizing our window,
           allowing the user to escape the application */
        SDL_MinimizeWindow(keyboard->focus);
    }

    return posted;
}

void SDL_SendKeyboardUnicodeKey(Uint64 timestamp, Uint32 ch)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    SDL_Keymod modstate = SDL_KMOD_NONE;
    SDL_Scancode scancode;

    if (ch == '\n') {
        ch = SDLK_RETURN;
    }
    scancode = SDL_GetKeymapScancode(keyboard->keymap, ch, &modstate);

    // Make sure we have this keycode in our keymap
    if (scancode == SDL_SCANCODE_UNKNOWN && ch < SDLK_SCANCODE_MASK) {
        scancode = GetNextReservedScancode();
        SetKeymapEntry(scancode, modstate, ch);
    }

    if (modstate & SDL_KMOD_SHIFT) {
        // If the character uses shift, press shift down
        SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_VIRTUAL, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_LSHIFT, true);
    }

    // Send a keydown and keyup for the character
    SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_VIRTUAL, SDL_GLOBAL_KEYBOARD_ID, 0, scancode, true);
    SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_VIRTUAL, SDL_GLOBAL_KEYBOARD_ID, 0, scancode, false);

    if (modstate & SDL_KMOD_SHIFT) {
        // If the character uses shift, release shift
        SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_VIRTUAL, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_LSHIFT, false);
    }
}

bool SDL_SendKeyboardKey(Uint64 timestamp, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, bool down)
{
    return SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_HARDWARE, keyboardID, rawcode, scancode, down);
}

bool SDL_SendKeyboardKeyAndKeycode(Uint64 timestamp, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, SDL_Keycode keycode, bool down)
{
    if (down) {
        // Make sure we have this keycode in our keymap
        SetKeymapEntry(scancode, SDL_GetModState(), keycode);
    }

    return SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_HARDWARE, keyboardID, rawcode, scancode, down);
}

bool SDL_SendKeyboardKeyIgnoreModifiers(Uint64 timestamp, SDL_KeyboardID keyboardID, int rawcode, SDL_Scancode scancode, bool down)
{
    return SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_HARDWARE | KEYBOARD_IGNOREMODIFIERS, keyboardID, rawcode, scancode, down);
}

bool SDL_SendKeyboardKeyAutoRelease(Uint64 timestamp, SDL_Scancode scancode)
{
    return SDL_SendKeyboardKeyInternal(timestamp, KEYBOARD_AUTORELEASE, SDL_GLOBAL_KEYBOARD_ID, 0, scancode, true);
}

void SDL_ReleaseAutoReleaseKeys(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    int scancode;

    if (keyboard->autorelease_pending) {
        for (scancode = SDL_SCANCODE_UNKNOWN; scancode < SDL_SCANCODE_COUNT; ++scancode) {
            if (keyboard->keysource[scancode] == KEYBOARD_AUTORELEASE) {
                SDL_SendKeyboardKeyInternal(0, KEYBOARD_AUTORELEASE, SDL_GLOBAL_KEYBOARD_ID, 0, (SDL_Scancode)scancode, false);
            }
        }
        keyboard->autorelease_pending = false;
    }

    if (keyboard->hardware_timestamp) {
        // Keep hardware keyboard "active" for 250 ms
        if (SDL_GetTicks() >= keyboard->hardware_timestamp + 250) {
            keyboard->hardware_timestamp = 0;
        }
    }
}

bool SDL_HardwareKeyboardKeyPressed(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    int scancode;

    for (scancode = SDL_SCANCODE_UNKNOWN; scancode < SDL_SCANCODE_COUNT; ++scancode) {
        if (keyboard->keysource[scancode] & KEYBOARD_HARDWARE) {
            return true;
        }
    }

    return keyboard->hardware_timestamp ? true : false;
}

void SDL_SendKeyboardText(const char *text)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (!keyboard->focus || !SDL_TextInputActive(keyboard->focus)) {
        return;
    }

    if (!text || !*text) {
        return;
    }

    // Don't post text events for unprintable characters
    if (SDL_iscntrl((unsigned char)*text)) {
        return;
    }

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_TEXT_INPUT)) {
        SDL_Event event;
        event.type = SDL_EVENT_TEXT_INPUT;
        event.common.timestamp = 0;
        event.text.windowID = keyboard->focus ? keyboard->focus->id : 0;
        event.text.text = SDL_CreateTemporaryString(text);
        if (!event.text.text) {
            return;
        }
        SDL_PushEvent(&event);
    }
}

void SDL_SendEditingText(const char *text, int start, int length)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (!keyboard->focus || !SDL_TextInputActive(keyboard->focus)) {
        return;
    }

    if (!text) {
        return;
    }

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_TEXT_EDITING)) {
        SDL_Event event;

        event.type = SDL_EVENT_TEXT_EDITING;
        event.common.timestamp = 0;
        event.edit.windowID = keyboard->focus ? keyboard->focus->id : 0;
        event.edit.start = start;
        event.edit.length = length;
        event.edit.text = SDL_CreateTemporaryString(text);
        if (!event.edit.text) {
            return;
        }
        SDL_PushEvent(&event);
    }
}

static const char * const *CreateCandidatesForEvent(char **candidates, int num_candidates)
{
    const char **event_candidates;
    int i;
    char *ptr;
    size_t total_length = (num_candidates + 1) * sizeof(*event_candidates);

    for (i = 0; i < num_candidates; ++i) {
        size_t length = SDL_strlen(candidates[i]) + 1;

        total_length += length;
    }

    event_candidates = (const char **)SDL_AllocateTemporaryMemory(total_length);
    if (!event_candidates) {
        return NULL;
    }
    ptr = (char *)(event_candidates + (num_candidates + 1));

    for (i = 0; i < num_candidates; ++i) {
        size_t length = SDL_strlen(candidates[i]) + 1;

        event_candidates[i] = ptr;
        SDL_memcpy(ptr, candidates[i], length);
        ptr += length;
    }
    event_candidates[i] = NULL;

    return event_candidates;
}

void SDL_SendEditingTextCandidates(char **candidates, int num_candidates, int selected_candidate, bool horizontal)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (!keyboard->focus || !SDL_TextInputActive(keyboard->focus)) {
        return;
    }

    // Post the event, if desired
    if (SDL_EventEnabled(SDL_EVENT_TEXT_EDITING_CANDIDATES)) {
        SDL_Event event;

        event.type = SDL_EVENT_TEXT_EDITING_CANDIDATES;
        event.common.timestamp = 0;
        event.edit.windowID = keyboard->focus ? keyboard->focus->id : 0;
        if (num_candidates > 0) {
            const char * const *event_candidates = CreateCandidatesForEvent(candidates, num_candidates);
            if (!event_candidates) {
                return;
            }
            event.edit_candidates.candidates = event_candidates;
            event.edit_candidates.num_candidates = num_candidates;
            event.edit_candidates.selected_candidate = selected_candidate;
            event.edit_candidates.horizontal = horizontal;
        } else {
            event.edit_candidates.candidates = NULL;
            event.edit_candidates.num_candidates = 0;
            event.edit_candidates.selected_candidate = -1;
            event.edit_candidates.horizontal = false;
        }
        SDL_PushEvent(&event);
    }
}

void SDL_QuitKeyboard(void)
{
    for (int i = SDL_keyboard_count; i--;) {
        SDL_RemoveKeyboard(SDL_keyboards[i].instance_id, false);
    }
    SDL_free(SDL_keyboards);
    SDL_keyboards = NULL;

    if (SDL_keyboard.keymap && SDL_keyboard.keymap->auto_release) {
        SDL_DestroyKeymap(SDL_keyboard.keymap);
        SDL_keyboard.keymap = NULL;
    }

    SDL_RemoveHintCallback(SDL_HINT_KEYCODE_OPTIONS,
                        SDL_KeycodeOptionsChanged, &SDL_keyboard);
}

const bool *SDL_GetKeyboardState(int *numkeys)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    if (numkeys != (int *)0) {
        *numkeys = SDL_SCANCODE_COUNT;
    }
    return keyboard->keystate;
}

SDL_Keymod SDL_GetModState(void)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    return keyboard->modstate;
}

void SDL_SetModState(SDL_Keymod modstate)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;

    keyboard->modstate = modstate;
}

// Note that SDL_ToggleModState() is not a public API. SDL_SetModState() is.
void SDL_ToggleModState(SDL_Keymod modstate, bool toggle)
{
    SDL_Keyboard *keyboard = &SDL_keyboard;
    if (toggle) {
        keyboard->modstate |= modstate;
    } else {
        keyboard->modstate &= ~modstate;
    }
}

