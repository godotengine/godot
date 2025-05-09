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

#ifdef SDL_VIDEO_DRIVER_EMSCRIPTEN

#include <emscripten/html5.h>
#include <emscripten/dom_pk_codes.h>

#include "../../events/SDL_dropevents_c.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_keyboard_c.h"
#include "../../events/SDL_touch_c.h"

#include "SDL_emscriptenevents.h"
#include "SDL_emscriptenvideo.h"

/*
Emscripten PK code to scancode
https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent
https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/code
*/
static const SDL_Scancode emscripten_scancode_table[] = {
    /* 0x00 "Unidentified"   */ SDL_SCANCODE_UNKNOWN,
    /* 0x01 "Escape"         */ SDL_SCANCODE_ESCAPE,
    /* 0x02 "Digit0"         */ SDL_SCANCODE_0,
    /* 0x03 "Digit1"         */ SDL_SCANCODE_1,
    /* 0x04 "Digit2"         */ SDL_SCANCODE_2,
    /* 0x05 "Digit3"         */ SDL_SCANCODE_3,
    /* 0x06 "Digit4"         */ SDL_SCANCODE_4,
    /* 0x07 "Digit5"         */ SDL_SCANCODE_5,
    /* 0x08 "Digit6"         */ SDL_SCANCODE_6,
    /* 0x09 "Digit7"         */ SDL_SCANCODE_7,
    /* 0x0A "Digit8"         */ SDL_SCANCODE_8,
    /* 0x0B "Digit9"         */ SDL_SCANCODE_9,
    /* 0x0C "Minus"          */ SDL_SCANCODE_MINUS,
    /* 0x0D "Equal"          */ SDL_SCANCODE_EQUALS,
    /* 0x0E "Backspace"      */ SDL_SCANCODE_BACKSPACE,
    /* 0x0F "Tab"            */ SDL_SCANCODE_TAB,
    /* 0x10 "KeyQ"           */ SDL_SCANCODE_Q,
    /* 0x11 "KeyW"           */ SDL_SCANCODE_W,
    /* 0x12 "KeyE"           */ SDL_SCANCODE_E,
    /* 0x13 "KeyR"           */ SDL_SCANCODE_R,
    /* 0x14 "KeyT"           */ SDL_SCANCODE_T,
    /* 0x15 "KeyY"           */ SDL_SCANCODE_Y,
    /* 0x16 "KeyU"           */ SDL_SCANCODE_U,
    /* 0x17 "KeyI"           */ SDL_SCANCODE_I,
    /* 0x18 "KeyO"           */ SDL_SCANCODE_O,
    /* 0x19 "KeyP"           */ SDL_SCANCODE_P,
    /* 0x1A "BracketLeft"    */ SDL_SCANCODE_LEFTBRACKET,
    /* 0x1B "BracketRight"   */ SDL_SCANCODE_RIGHTBRACKET,
    /* 0x1C "Enter"          */ SDL_SCANCODE_RETURN,
    /* 0x1D "ControlLeft"    */ SDL_SCANCODE_LCTRL,
    /* 0x1E "KeyA"           */ SDL_SCANCODE_A,
    /* 0x1F "KeyS"           */ SDL_SCANCODE_S,
    /* 0x20 "KeyD"           */ SDL_SCANCODE_D,
    /* 0x21 "KeyF"           */ SDL_SCANCODE_F,
    /* 0x22 "KeyG"           */ SDL_SCANCODE_G,
    /* 0x23 "KeyH"           */ SDL_SCANCODE_H,
    /* 0x24 "KeyJ"           */ SDL_SCANCODE_J,
    /* 0x25 "KeyK"           */ SDL_SCANCODE_K,
    /* 0x26 "KeyL"           */ SDL_SCANCODE_L,
    /* 0x27 "Semicolon"      */ SDL_SCANCODE_SEMICOLON,
    /* 0x28 "Quote"          */ SDL_SCANCODE_APOSTROPHE,
    /* 0x29 "Backquote"      */ SDL_SCANCODE_GRAVE,
    /* 0x2A "ShiftLeft"      */ SDL_SCANCODE_LSHIFT,
    /* 0x2B "Backslash"      */ SDL_SCANCODE_BACKSLASH,
    /* 0x2C "KeyZ"           */ SDL_SCANCODE_Z,
    /* 0x2D "KeyX"           */ SDL_SCANCODE_X,
    /* 0x2E "KeyC"           */ SDL_SCANCODE_C,
    /* 0x2F "KeyV"           */ SDL_SCANCODE_V,
    /* 0x30 "KeyB"           */ SDL_SCANCODE_B,
    /* 0x31 "KeyN"           */ SDL_SCANCODE_N,
    /* 0x32 "KeyM"           */ SDL_SCANCODE_M,
    /* 0x33 "Comma"          */ SDL_SCANCODE_COMMA,
    /* 0x34 "Period"         */ SDL_SCANCODE_PERIOD,
    /* 0x35 "Slash"          */ SDL_SCANCODE_SLASH,
    /* 0x36 "ShiftRight"     */ SDL_SCANCODE_RSHIFT,
    /* 0x37 "NumpadMultiply" */ SDL_SCANCODE_KP_MULTIPLY,
    /* 0x38 "AltLeft"        */ SDL_SCANCODE_LALT,
    /* 0x39 "Space"          */ SDL_SCANCODE_SPACE,
    /* 0x3A "CapsLock"       */ SDL_SCANCODE_CAPSLOCK,
    /* 0x3B "F1"             */ SDL_SCANCODE_F1,
    /* 0x3C "F2"             */ SDL_SCANCODE_F2,
    /* 0x3D "F3"             */ SDL_SCANCODE_F3,
    /* 0x3E "F4"             */ SDL_SCANCODE_F4,
    /* 0x3F "F5"             */ SDL_SCANCODE_F5,
    /* 0x40 "F6"             */ SDL_SCANCODE_F6,
    /* 0x41 "F7"             */ SDL_SCANCODE_F7,
    /* 0x42 "F8"             */ SDL_SCANCODE_F8,
    /* 0x43 "F9"             */ SDL_SCANCODE_F9,
    /* 0x44 "F10"            */ SDL_SCANCODE_F10,
    /* 0x45 "Pause"          */ SDL_SCANCODE_PAUSE,
    /* 0x46 "ScrollLock"     */ SDL_SCANCODE_SCROLLLOCK,
    /* 0x47 "Numpad7"        */ SDL_SCANCODE_KP_7,
    /* 0x48 "Numpad8"        */ SDL_SCANCODE_KP_8,
    /* 0x49 "Numpad9"        */ SDL_SCANCODE_KP_9,
    /* 0x4A "NumpadSubtract" */ SDL_SCANCODE_KP_MINUS,
    /* 0x4B "Numpad4"        */ SDL_SCANCODE_KP_4,
    /* 0x4C "Numpad5"        */ SDL_SCANCODE_KP_5,
    /* 0x4D "Numpad6"        */ SDL_SCANCODE_KP_6,
    /* 0x4E "NumpadAdd"      */ SDL_SCANCODE_KP_PLUS,
    /* 0x4F "Numpad1"        */ SDL_SCANCODE_KP_1,
    /* 0x50 "Numpad2"        */ SDL_SCANCODE_KP_2,
    /* 0x51 "Numpad3"        */ SDL_SCANCODE_KP_3,
    /* 0x52 "Numpad0"        */ SDL_SCANCODE_KP_0,
    /* 0x53 "NumpadDecimal"  */ SDL_SCANCODE_KP_PERIOD,
    /* 0x54 "PrintScreen"    */ SDL_SCANCODE_PRINTSCREEN,
    /* 0x55                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x56 "IntlBackslash"  */ SDL_SCANCODE_NONUSBACKSLASH,
    /* 0x57 "F11"            */ SDL_SCANCODE_F11,
    /* 0x58 "F12"            */ SDL_SCANCODE_F12,
    /* 0x59 "NumpadEqual"    */ SDL_SCANCODE_KP_EQUALS,
    /* 0x5A                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x5B                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x5C                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x5D                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x5E                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x5F                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x60                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x61                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x62                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x63                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x64 "F13"            */ SDL_SCANCODE_F13,
    /* 0x65 "F14"            */ SDL_SCANCODE_F14,
    /* 0x66 "F15"            */ SDL_SCANCODE_F15,
    /* 0x67 "F16"            */ SDL_SCANCODE_F16,
    /* 0x68 "F17"            */ SDL_SCANCODE_F17,
    /* 0x69 "F18"            */ SDL_SCANCODE_F18,
    /* 0x6A "F19"            */ SDL_SCANCODE_F19,
    /* 0x6B "F20"            */ SDL_SCANCODE_F20,
    /* 0x6C "F21"            */ SDL_SCANCODE_F21,
    /* 0x6D "F22"            */ SDL_SCANCODE_F22,
    /* 0x6E "F23"            */ SDL_SCANCODE_F23,
    /* 0x6F                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x70 "KanaMode"       */ SDL_SCANCODE_INTERNATIONAL2,
    /* 0x71 "Lang2"          */ SDL_SCANCODE_LANG2,
    /* 0x72 "Lang1"          */ SDL_SCANCODE_LANG1,
    /* 0x73 "IntlRo"         */ SDL_SCANCODE_INTERNATIONAL1,
    /* 0x74                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x75                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x76 "F24"            */ SDL_SCANCODE_F24,
    /* 0x77                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x78                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x79 "Convert"        */ SDL_SCANCODE_INTERNATIONAL4,
    /* 0x7A                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x7B "NonConvert"     */ SDL_SCANCODE_INTERNATIONAL5,
    /* 0x7C                  */ SDL_SCANCODE_UNKNOWN,
    /* 0x7D "IntlYen"        */ SDL_SCANCODE_INTERNATIONAL3,
    /* 0x7E "NumpadComma"    */ SDL_SCANCODE_KP_COMMA
};

static SDL_Scancode Emscripten_MapScanCode(const char *code)
{
    const DOM_PK_CODE_TYPE pk_code = emscripten_compute_dom_pk_code(code);
    if (pk_code < SDL_arraysize(emscripten_scancode_table)) {
        return emscripten_scancode_table[pk_code];
    }

    switch (pk_code) {
    case DOM_PK_PASTE:
        return SDL_SCANCODE_PASTE;
    case DOM_PK_MEDIA_TRACK_PREVIOUS:
        return SDL_SCANCODE_MEDIA_PREVIOUS_TRACK;
    case DOM_PK_CUT:
        return SDL_SCANCODE_CUT;
    case DOM_PK_COPY:
        return SDL_SCANCODE_COPY;
    case DOM_PK_MEDIA_TRACK_NEXT:
        return SDL_SCANCODE_MEDIA_NEXT_TRACK;
    case DOM_PK_NUMPAD_ENTER:
        return SDL_SCANCODE_KP_ENTER;
    case DOM_PK_CONTROL_RIGHT:
        return SDL_SCANCODE_RCTRL;
    case DOM_PK_AUDIO_VOLUME_MUTE:
        return SDL_SCANCODE_MUTE;
    case DOM_PK_MEDIA_PLAY_PAUSE:
        return SDL_SCANCODE_MEDIA_PLAY_PAUSE;
    case DOM_PK_MEDIA_STOP:
        return SDL_SCANCODE_MEDIA_STOP;
    case DOM_PK_EJECT:
        return SDL_SCANCODE_MEDIA_EJECT;
    case DOM_PK_AUDIO_VOLUME_DOWN:
        return SDL_SCANCODE_VOLUMEDOWN;
    case DOM_PK_AUDIO_VOLUME_UP:
        return SDL_SCANCODE_VOLUMEUP;
    case DOM_PK_BROWSER_HOME:
        return SDL_SCANCODE_AC_HOME;
    case DOM_PK_NUMPAD_DIVIDE:
        return SDL_SCANCODE_KP_DIVIDE;
    case DOM_PK_ALT_RIGHT:
        return SDL_SCANCODE_RALT;
    case DOM_PK_HELP:
        return SDL_SCANCODE_HELP;
    case DOM_PK_NUM_LOCK:
        return SDL_SCANCODE_NUMLOCKCLEAR;
    case DOM_PK_HOME:
        return SDL_SCANCODE_HOME;
    case DOM_PK_ARROW_UP:
        return SDL_SCANCODE_UP;
    case DOM_PK_PAGE_UP:
        return SDL_SCANCODE_PAGEUP;
    case DOM_PK_ARROW_LEFT:
        return SDL_SCANCODE_LEFT;
    case DOM_PK_ARROW_RIGHT:
        return SDL_SCANCODE_RIGHT;
    case DOM_PK_END:
        return SDL_SCANCODE_END;
    case DOM_PK_ARROW_DOWN:
        return SDL_SCANCODE_DOWN;
    case DOM_PK_PAGE_DOWN:
        return SDL_SCANCODE_PAGEDOWN;
    case DOM_PK_INSERT:
        return SDL_SCANCODE_INSERT;
    case DOM_PK_DELETE:
        return SDL_SCANCODE_DELETE;
    case DOM_PK_META_LEFT:
        return SDL_SCANCODE_LGUI;
    case DOM_PK_META_RIGHT:
        return SDL_SCANCODE_RGUI;
    case DOM_PK_CONTEXT_MENU:
        return SDL_SCANCODE_APPLICATION;
    case DOM_PK_POWER:
        return SDL_SCANCODE_POWER;
    case DOM_PK_BROWSER_SEARCH:
        return SDL_SCANCODE_AC_SEARCH;
    case DOM_PK_BROWSER_FAVORITES:
        return SDL_SCANCODE_AC_BOOKMARKS;
    case DOM_PK_BROWSER_REFRESH:
        return SDL_SCANCODE_AC_REFRESH;
    case DOM_PK_BROWSER_STOP:
        return SDL_SCANCODE_AC_STOP;
    case DOM_PK_BROWSER_FORWARD:
        return SDL_SCANCODE_AC_FORWARD;
    case DOM_PK_BROWSER_BACK:
        return SDL_SCANCODE_AC_BACK;
    case DOM_PK_MEDIA_SELECT:
        return SDL_SCANCODE_MEDIA_SELECT;
    }

    return SDL_SCANCODE_UNKNOWN;
}

static SDL_Window *Emscripten_GetFocusedWindow(SDL_VideoDevice *device)
{
    SDL_Window *window;
    for (window = device->windows; window; window = window->next) {
        SDL_WindowData *wdata = window->internal;

        const int focused = MAIN_THREAD_EM_ASM_INT({
            var id = UTF8ToString($0);
            try
            {
                var canvas = document.querySelector(id);
                if (canvas) {
                    return canvas === document.activeElement;
                }
            }
            catch (e)
            {
                // querySelector throws if not a valid selector
            }
            return false;
        }, wdata->canvas_id);

        if (focused) {
            break;
        }
    }
    return window;
}

static EM_BOOL Emscripten_HandlePointerLockChange(int eventType, const EmscriptenPointerlockChangeEvent *changeEvent, void *userData)
{
    SDL_WindowData *window_data = (SDL_WindowData *)userData;
    // keep track of lock losses, so we can regrab if/when appropriate.
    window_data->has_pointer_lock = changeEvent->isActive;
    return 0;
}

static EM_BOOL Emscripten_HandlePointerLockChangeGlobal(int eventType, const EmscriptenPointerlockChangeEvent *changeEvent, void *userData)
{
    SDL_VideoDevice *device = userData;
    bool prevent_default = false;
    SDL_Window *window;

    for (window = device->windows; window; window = window->next) {
        prevent_default |= Emscripten_HandlePointerLockChange(eventType, changeEvent, window->internal);
    }

    return prevent_default;
}

static EM_BOOL Emscripten_HandleMouseMove(int eventType, const EmscriptenMouseEvent *mouseEvent, void *userData)
{
    SDL_WindowData *window_data = userData;
    const bool isPointerLocked = window_data->has_pointer_lock;
    float mx, my;

    // rescale (in case canvas is being scaled)
    double client_w, client_h, xscale, yscale;
    emscripten_get_element_css_size(window_data->canvas_id, &client_w, &client_h);
    xscale = window_data->window->w / client_w;
    yscale = window_data->window->h / client_h;

    if (isPointerLocked) {
        mx = (float)(mouseEvent->movementX * xscale);
        my = (float)(mouseEvent->movementY * yscale);
    } else {
        mx = (float)(mouseEvent->targetX * xscale);
        my = (float)(mouseEvent->targetY * yscale);
    }

    SDL_SendMouseMotion(0, window_data->window, SDL_DEFAULT_MOUSE_ID, isPointerLocked, mx, my);
    return 0;
}

static EM_BOOL Emscripten_HandleMouseButton(int eventType, const EmscriptenMouseEvent *mouseEvent, void *userData)
{
    SDL_WindowData *window_data = userData;
    Uint8 sdl_button;
    bool sdl_button_state;
    double css_w, css_h;
    bool prevent_default = false; // needed for iframe implementation in Chrome-based browsers.

    switch (mouseEvent->button) {
    case 0:
        sdl_button = SDL_BUTTON_LEFT;
        break;
    case 1:
        sdl_button = SDL_BUTTON_MIDDLE;
        break;
    case 2:
        sdl_button = SDL_BUTTON_RIGHT;
        break;
    default:
        return 0;
    }

    const SDL_Mouse *mouse = SDL_GetMouse();
    SDL_assert(mouse != NULL);

    if (eventType == EMSCRIPTEN_EVENT_MOUSEDOWN) {
        if (mouse->relative_mode && !window_data->has_pointer_lock) {
            emscripten_request_pointerlock(window_data->canvas_id, 0); // try to regrab lost pointer lock.
        }
        sdl_button_state = true;
    } else {
        sdl_button_state = false;
        prevent_default = SDL_EventEnabled(SDL_EVENT_MOUSE_BUTTON_UP);
    }

    SDL_SendMouseButton(0, window_data->window, SDL_DEFAULT_MOUSE_ID, sdl_button, sdl_button_state);

    // We have an imaginary mouse capture, because we need SDL to not drop our imaginary mouse focus when we leave the canvas.
    if (mouse->auto_capture) {
        if (SDL_GetMouseState(NULL, NULL) != 0) {
            window_data->window->flags |= SDL_WINDOW_MOUSE_CAPTURE;
        } else {
            window_data->window->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
        }
    }

    if ((eventType == EMSCRIPTEN_EVENT_MOUSEUP) && window_data->mouse_focus_loss_pending) {
        window_data->mouse_focus_loss_pending = (window_data->window->flags & SDL_WINDOW_MOUSE_CAPTURE) != 0;
        if (!window_data->mouse_focus_loss_pending) {
            SDL_SetMouseFocus(NULL);
        }
    } else {
        // Do not consume the event if the mouse is outside of the canvas.
        emscripten_get_element_css_size(window_data->canvas_id, &css_w, &css_h);
        if (mouseEvent->targetX < 0 || mouseEvent->targetX >= css_w ||
            mouseEvent->targetY < 0 || mouseEvent->targetY >= css_h) {
            return 0;
        }
    }

    return prevent_default;
}

static EM_BOOL Emscripten_HandleMouseButtonGlobal(int eventType, const EmscriptenMouseEvent *mouseEvent, void *userData)
{
    SDL_VideoDevice *device = userData;
    bool prevent_default = false;
    SDL_Window *window;

    for (window = device->windows; window; window = window->next) {
        prevent_default |= Emscripten_HandleMouseButton(eventType, mouseEvent, window->internal);
    }

    return prevent_default;
}

static EM_BOOL Emscripten_HandleMouseFocus(int eventType, const EmscriptenMouseEvent *mouseEvent, void *userData)
{
    SDL_WindowData *window_data = userData;

    const bool isPointerLocked = window_data->has_pointer_lock;

    if (!isPointerLocked) {
        // rescale (in case canvas is being scaled)
        float mx, my;
        double client_w, client_h;
        emscripten_get_element_css_size(window_data->canvas_id, &client_w, &client_h);

        mx = (float)(mouseEvent->targetX * (window_data->window->w / client_w));
        my = (float)(mouseEvent->targetY * (window_data->window->h / client_h));
        SDL_SendMouseMotion(0, window_data->window, SDL_GLOBAL_MOUSE_ID, isPointerLocked, mx, my);
    }

    const bool isenter = (eventType == EMSCRIPTEN_EVENT_MOUSEENTER);
    if (isenter && window_data->mouse_focus_loss_pending) {
        window_data->mouse_focus_loss_pending = false;  // just drop the state, but don't send the enter event.
    } else if (!isenter && (window_data->window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        window_data->mouse_focus_loss_pending = true;  // waiting on a mouse button to let go before we send the mouse focus update.
    } else {
        SDL_SetMouseFocus(isenter ? window_data->window : NULL);
    }

    return SDL_EventEnabled(SDL_EVENT_MOUSE_MOTION);  // !!! FIXME: should this be MOUSE_MOTION or something else?
}

static EM_BOOL Emscripten_HandleWheel(int eventType, const EmscriptenWheelEvent *wheelEvent, void *userData)
{
    SDL_WindowData *window_data = userData;

    float deltaY = wheelEvent->deltaY;
    float deltaX = wheelEvent->deltaX;

    switch (wheelEvent->deltaMode) {
    case DOM_DELTA_PIXEL:
        deltaX /= 100; // 100 pixels make up a step
        deltaY /= 100; // 100 pixels make up a step
        break;
    case DOM_DELTA_LINE:
        deltaX /= 3; // 3 lines make up a step
        deltaY /= 3; // 3 lines make up a step
        break;
    case DOM_DELTA_PAGE:
        deltaX *= 80; // A page makes up 80 steps
        deltaY *= 80; // A page makes up 80 steps
        break;
    }

    SDL_SendMouseWheel(0, window_data->window, SDL_DEFAULT_MOUSE_ID, deltaX, -deltaY, SDL_MOUSEWHEEL_NORMAL);
    return SDL_EventEnabled(SDL_EVENT_MOUSE_WHEEL);
}

static EM_BOOL Emscripten_HandleFocus(int eventType, const EmscriptenFocusEvent *focusEvent, void *userData)
{
    SDL_VideoDevice *device = userData;
    SDL_Window *window = Emscripten_GetFocusedWindow(device);

    SDL_EventType sdl_event_type;

    /* If the user switches away while keys are pressed (such as
     * via Alt+Tab), key release events won't be received. */
    if (eventType == EMSCRIPTEN_EVENT_BLUR) {
        SDL_ResetKeyboard();
    }

    sdl_event_type = (eventType == EMSCRIPTEN_EVENT_FOCUS) ? SDL_EVENT_WINDOW_FOCUS_GAINED : SDL_EVENT_WINDOW_FOCUS_LOST;
    SDL_SetKeyboardFocus(sdl_event_type == SDL_EVENT_WINDOW_FOCUS_GAINED ? window : NULL);
    return SDL_EventEnabled(sdl_event_type);
}

static EM_BOOL Emscripten_HandleTouch(int eventType, const EmscriptenTouchEvent *touchEvent, void *userData)
{
    SDL_WindowData *window_data = (SDL_WindowData *)userData;
    int i;
    double client_w, client_h;
    int preventDefault = 0;

    const SDL_TouchID deviceId = 1;
    if (SDL_AddTouch(deviceId, SDL_TOUCH_DEVICE_DIRECT, "") < 0) {
        return 0;
    }

    emscripten_get_element_css_size(window_data->canvas_id, &client_w, &client_h);

    for (i = 0; i < touchEvent->numTouches; i++) {
        SDL_FingerID id;
        float x, y;

        if (!touchEvent->touches[i].isChanged) {
            continue;
        }

        id = touchEvent->touches[i].identifier + 1;
        if (client_w <= 1) {
            x = 0.5f;
        } else {
            x = touchEvent->touches[i].targetX / (client_w - 1);
        }
        if (client_h <= 1) {
            y = 0.5f;
        } else {
            y = touchEvent->touches[i].targetY / (client_h - 1);
        }

        if (eventType == EMSCRIPTEN_EVENT_TOUCHSTART) {
            SDL_SendTouch(0, deviceId, id, window_data->window, SDL_EVENT_FINGER_DOWN, x, y, 1.0f);

            // disable browser scrolling/pinch-to-zoom if app handles touch events
            if (!preventDefault && SDL_EventEnabled(SDL_EVENT_FINGER_DOWN)) {
                preventDefault = 1;
            }
        } else if (eventType == EMSCRIPTEN_EVENT_TOUCHMOVE) {
            SDL_SendTouchMotion(0, deviceId, id, window_data->window, x, y, 1.0f);
        } else if (eventType == EMSCRIPTEN_EVENT_TOUCHEND) {
            SDL_SendTouch(0, deviceId, id, window_data->window, SDL_EVENT_FINGER_UP, x, y, 1.0f);

            // block browser's simulated mousedown/mouseup on touchscreen devices
            preventDefault = 1;
        } else if (eventType == EMSCRIPTEN_EVENT_TOUCHCANCEL) {
            SDL_SendTouch(0, deviceId, id, window_data->window, SDL_EVENT_FINGER_CANCELED, x, y, 1.0f);
        }
    }

    return preventDefault;
}

static bool IsFunctionKey(SDL_Scancode scancode)
{
    if (scancode >= SDL_SCANCODE_F1 && scancode <= SDL_SCANCODE_F12) {
        return true;
    }
    if (scancode >= SDL_SCANCODE_F13 && scancode <= SDL_SCANCODE_F24) {
        return true;
    }
    return false;
}

/* This is a great tool to see web keyboard events live:
 * https://w3c.github.io/uievents/tools/key-event-viewer.html
 */
static EM_BOOL Emscripten_HandleKey(int eventType, const EmscriptenKeyboardEvent *keyEvent, void *userData)
{
    SDL_WindowData *window_data = (SDL_WindowData *)userData;
    SDL_Scancode scancode = Emscripten_MapScanCode(keyEvent->code);
    SDL_Keycode keycode = SDLK_UNKNOWN;
    bool prevent_default = false;
    bool is_nav_key = false;

    if (scancode == SDL_SCANCODE_UNKNOWN) {
        if (SDL_strcmp(keyEvent->key, "Sleep") == 0) {
            scancode = SDL_SCANCODE_SLEEP;
        } else if (SDL_strcmp(keyEvent->key, "ChannelUp") == 0) {
            scancode = SDL_SCANCODE_CHANNEL_INCREMENT;
        } else if (SDL_strcmp(keyEvent->key, "ChannelDown") == 0) {
            scancode = SDL_SCANCODE_CHANNEL_DECREMENT;
        } else if (SDL_strcmp(keyEvent->key, "MediaPlay") == 0) {
            scancode = SDL_SCANCODE_MEDIA_PLAY;
        } else if (SDL_strcmp(keyEvent->key, "MediaPause") == 0) {
            scancode = SDL_SCANCODE_MEDIA_PAUSE;
        } else if (SDL_strcmp(keyEvent->key, "MediaRecord") == 0) {
            scancode = SDL_SCANCODE_MEDIA_RECORD;
        } else if (SDL_strcmp(keyEvent->key, "MediaFastForward") == 0) {
            scancode = SDL_SCANCODE_MEDIA_FAST_FORWARD;
        } else if (SDL_strcmp(keyEvent->key, "MediaRewind") == 0) {
            scancode = SDL_SCANCODE_MEDIA_REWIND;
        } else if (SDL_strcmp(keyEvent->key, "Close") == 0) {
            scancode = SDL_SCANCODE_AC_CLOSE;
        } else if (SDL_strcmp(keyEvent->key, "New") == 0) {
            scancode = SDL_SCANCODE_AC_NEW;
        } else if (SDL_strcmp(keyEvent->key, "Open") == 0) {
            scancode = SDL_SCANCODE_AC_OPEN;
        } else if (SDL_strcmp(keyEvent->key, "Print") == 0) {
            scancode = SDL_SCANCODE_AC_PRINT;
        } else if (SDL_strcmp(keyEvent->key, "Save") == 0) {
            scancode = SDL_SCANCODE_AC_SAVE;
        } else if (SDL_strcmp(keyEvent->key, "Props") == 0) {
            scancode = SDL_SCANCODE_AC_PROPERTIES;
        }
    }

    if (scancode == SDL_SCANCODE_UNKNOWN) {
        // KaiOS Left Soft Key and Right Soft Key, they act as OK/Next/Menu and Cancel/Back/Clear
        if (SDL_strcmp(keyEvent->key, "SoftLeft") == 0) {
            scancode = SDL_SCANCODE_AC_FORWARD;
        } else if (SDL_strcmp(keyEvent->key, "SoftRight") == 0) {
            scancode = SDL_SCANCODE_AC_BACK;
        }
    }

    if (keyEvent->location == 0 && SDL_utf8strlen(keyEvent->key) == 1) {
        const char *key = keyEvent->key;
        keycode = SDL_StepUTF8(&key, NULL);
        if (keycode == SDL_INVALID_UNICODE_CODEPOINT) {
            keycode = SDLK_UNKNOWN;
        }
    }

    if (keycode != SDLK_UNKNOWN) {
        prevent_default = SDL_SendKeyboardKeyAndKeycode(0, SDL_DEFAULT_KEYBOARD_ID, 0, scancode, keycode, (eventType == EMSCRIPTEN_EVENT_KEYDOWN));
    } else {
        prevent_default = SDL_SendKeyboardKey(0, SDL_DEFAULT_KEYBOARD_ID, 0, scancode, (eventType == EMSCRIPTEN_EVENT_KEYDOWN));
    }

    /* if TEXTINPUT events are enabled we can't prevent keydown or we won't get keypress
     * we need to ALWAYS prevent backspace and tab otherwise chrome takes action and does bad navigation UX
     */
    if ((scancode == SDL_SCANCODE_BACKSPACE) ||
        (scancode == SDL_SCANCODE_TAB) ||
        (scancode == SDL_SCANCODE_LEFT) ||
        (scancode == SDL_SCANCODE_UP) ||
        (scancode == SDL_SCANCODE_RIGHT) ||
        (scancode == SDL_SCANCODE_DOWN) ||
        IsFunctionKey(scancode) ||
        keyEvent->ctrlKey) {
        is_nav_key = true;
    }

    if ((eventType == EMSCRIPTEN_EVENT_KEYDOWN) && SDL_TextInputActive(window_data->window) && !is_nav_key) {
        prevent_default = false;
    }

    return prevent_default;
}

static EM_BOOL Emscripten_HandleKeyPress(int eventType, const EmscriptenKeyboardEvent *keyEvent, void *userData)
{
    SDL_WindowData *window_data = (SDL_WindowData *)userData;

    if (SDL_TextInputActive(window_data->window)) {
        char text[5];
        char *end = SDL_UCS4ToUTF8(keyEvent->charCode, text);
        *end = '\0';
        SDL_SendKeyboardText(text);
        return EM_TRUE;
    }
    return EM_FALSE;
}

static EM_BOOL Emscripten_HandleFullscreenChange(int eventType, const EmscriptenFullscreenChangeEvent *fullscreenChangeEvent, void *userData)
{
    SDL_WindowData *window_data = userData;

    if (fullscreenChangeEvent->isFullscreen) {
        SDL_SendWindowEvent(window_data->window, SDL_EVENT_WINDOW_ENTER_FULLSCREEN, 0, 0);
        window_data->fullscreen_mode_flags = 0;
    } else {
        SDL_SendWindowEvent(window_data->window, SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);
    }

    SDL_UpdateFullscreenMode(window_data->window, fullscreenChangeEvent->isFullscreen, false);

    return 0;
}

static EM_BOOL Emscripten_HandleFullscreenChangeGlobal(int eventType, const EmscriptenFullscreenChangeEvent *fullscreenChangeEvent, void *userData)
{
    SDL_VideoDevice *device = userData;
    SDL_Window *window = Emscripten_GetFocusedWindow(device);
    if (window) {
        return Emscripten_HandleFullscreenChange(eventType, fullscreenChangeEvent, window->internal);
    }
    return EM_FALSE;
}

static EM_BOOL Emscripten_HandleResize(int eventType, const EmscriptenUiEvent *uiEvent, void *userData)
{
    SDL_WindowData *window_data = userData;
    bool force = false;

    // update pixel ratio
    if (window_data->window->flags & SDL_WINDOW_HIGH_PIXEL_DENSITY) {
        if (window_data->pixel_ratio != emscripten_get_device_pixel_ratio()) {
            window_data->pixel_ratio = emscripten_get_device_pixel_ratio();
            force = true;
        }
    }

    if (!(window_data->window->flags & SDL_WINDOW_FULLSCREEN)) {
        // this will only work if the canvas size is set through css
        if (window_data->window->flags & SDL_WINDOW_RESIZABLE) {
            double w = window_data->window->w;
            double h = window_data->window->h;

            if (window_data->external_size) {
                emscripten_get_element_css_size(window_data->canvas_id, &w, &h);
            }

            emscripten_set_canvas_element_size(window_data->canvas_id, SDL_lroundf(w * window_data->pixel_ratio), SDL_lroundf(h * window_data->pixel_ratio));

            // set_canvas_size unsets this
            if (!window_data->external_size && window_data->pixel_ratio != 1.0f) {
                emscripten_set_element_css_size(window_data->canvas_id, w, h);
            }

            if (force) {
                // force the event to trigger, so pixel ratio changes can be handled
                window_data->window->w = 0;
                window_data->window->h = 0;
            }

            SDL_SendWindowEvent(window_data->window, SDL_EVENT_WINDOW_RESIZED, SDL_lroundf(w), SDL_lroundf(h));
        }
    }

    return 0;
}

static EM_BOOL Emscripten_HandleResizeGlobal(int eventType, const EmscriptenUiEvent *uiEvent, void *userData)
{
    SDL_VideoDevice *device = userData;
    bool prevent_default = false;
    SDL_Window *window;

    for (window = device->windows; window; window = window->next) {
        prevent_default |= Emscripten_HandleResize(eventType, uiEvent, window->internal);
    }

    return prevent_default;
}

EM_BOOL
Emscripten_HandleCanvasResize(int eventType, const void *reserved, void *userData)
{
    // this is used during fullscreen changes
    SDL_WindowData *window_data = userData;

    if (window_data->fullscreen_resize) {
        double css_w, css_h;
        emscripten_get_element_css_size(window_data->canvas_id, &css_w, &css_h);
        SDL_SendWindowEvent(window_data->window, SDL_EVENT_WINDOW_RESIZED, SDL_lroundf(css_w), SDL_lroundf(css_h));
    }

    return 0;
}

static EM_BOOL Emscripten_HandleVisibilityChange(int eventType, const EmscriptenVisibilityChangeEvent *visEvent, void *userData)
{
    SDL_WindowData *window_data = userData;
    SDL_SendWindowEvent(window_data->window, visEvent->hidden ? SDL_EVENT_WINDOW_HIDDEN : SDL_EVENT_WINDOW_SHOWN, 0, 0);
    return 0;
}

static const char *Emscripten_HandleBeforeUnload(int eventType, const void *reserved, void *userData)
{
    /* This event will need to be handled synchronously, e.g. using
       SDL_AddEventWatch, as the page is being closed *now*. */
    // No need to send a SDL_EVENT_QUIT, the app won't get control again.
    SDL_SendAppEvent(SDL_EVENT_TERMINATING);
    return ""; // don't trigger confirmation dialog
}

static EM_BOOL Emscripten_HandleOrientationChange(int eventType, const EmscriptenOrientationChangeEvent *orientationChangeEvent, void *userData)
{
    SDL_DisplayOrientation orientation;
    switch (orientationChangeEvent->orientationIndex) {
        #define CHECK_ORIENTATION(emsdk, sdl) case EMSCRIPTEN_ORIENTATION_##emsdk: orientation = SDL_ORIENTATION_##sdl; break
        CHECK_ORIENTATION(LANDSCAPE_PRIMARY, LANDSCAPE);
        CHECK_ORIENTATION(LANDSCAPE_SECONDARY, LANDSCAPE_FLIPPED);
        CHECK_ORIENTATION(PORTRAIT_PRIMARY, PORTRAIT);
        CHECK_ORIENTATION(PORTRAIT_SECONDARY, PORTRAIT_FLIPPED);
        #undef CHECK_ORIENTATION
        default: orientation = SDL_ORIENTATION_UNKNOWN; break;
    }

    SDL_WindowData *window_data = (SDL_WindowData *) userData;
    SDL_SendDisplayEvent(SDL_GetVideoDisplayForWindow(window_data->window), SDL_EVENT_DISPLAY_ORIENTATION, orientation, 0);

    return 0;
}

// IF YOU CHANGE THIS STRUCTURE, YOU NEED TO UPDATE THE JAVASCRIPT THAT FILLS IT IN: makePointerEventCStruct, below.
typedef struct Emscripten_PointerEvent
{
    int pointerid;
    int button;
    int buttons;
    float movementX;
    float movementY;
    float targetX;
    float targetY;
    float pressure;
    float tangential_pressure;
    float tiltx;
    float tilty;
    float rotation;
} Emscripten_PointerEvent;

static void Emscripten_UpdatePointerFromEvent(SDL_WindowData *window_data, const Emscripten_PointerEvent *event)
{
    const SDL_PenID pen = SDL_FindPenByHandle((void *) (size_t) event->pointerid);
    if (pen) {
        // rescale (in case canvas is being scaled)
        double client_w, client_h;
        emscripten_get_element_css_size(window_data->canvas_id, &client_w, &client_h);
        const double xscale = window_data->window->w / client_w;
        const double yscale = window_data->window->h / client_h;

        const bool isPointerLocked = window_data->has_pointer_lock;
        float mx, my;
        if (isPointerLocked) {
            mx = (float)(event->movementX * xscale);
            my = (float)(event->movementY * yscale);
        } else {
            mx = (float)(event->targetX * xscale);
            my = (float)(event->targetY * yscale);
        }

        SDL_SendPenMotion(0, pen, window_data->window, mx, my);

        if (event->button == 0) {  // pen touch
            bool down = ((event->buttons & 1) != 0);
            SDL_SendPenTouch(0, pen, window_data->window, false, down);
        } else if (event->button == 5) {  // eraser touch...? Not sure if this is right...
            bool down = ((event->buttons & 32) != 0);
            SDL_SendPenTouch(0, pen, window_data->window, true, down);
        } else if (event->button == 1) {
            bool down = ((event->buttons & 4) != 0);
            SDL_SendPenButton(0, pen, window_data->window, 2, down);
        } else if (event->button == 2) {
            bool down = ((event->buttons & 2) != 0);
            SDL_SendPenButton(0, pen, window_data->window, 1, down);
        }

        SDL_SendPenAxis(0, pen, window_data->window, SDL_PEN_AXIS_PRESSURE, event->pressure);
        SDL_SendPenAxis(0, pen, window_data->window, SDL_PEN_AXIS_TANGENTIAL_PRESSURE, event->tangential_pressure);
        SDL_SendPenAxis(0, pen, window_data->window, SDL_PEN_AXIS_XTILT, event->tiltx);
        SDL_SendPenAxis(0, pen, window_data->window, SDL_PEN_AXIS_YTILT, event->tilty);
        SDL_SendPenAxis(0, pen, window_data->window, SDL_PEN_AXIS_ROTATION, event->rotation);
    }
}

EMSCRIPTEN_KEEPALIVE void Emscripten_HandlePointerEnter(SDL_WindowData *window_data, const Emscripten_PointerEvent *event)
{
    // Web browsers offer almost none of this information as specifics, but can without warning offer any of these specific things.
    SDL_PenInfo peninfo;
    SDL_zero(peninfo);
    peninfo.capabilities = SDL_PEN_CAPABILITY_PRESSURE | SDL_PEN_CAPABILITY_ROTATION | SDL_PEN_CAPABILITY_XTILT | SDL_PEN_CAPABILITY_YTILT | SDL_PEN_CAPABILITY_TANGENTIAL_PRESSURE | SDL_PEN_CAPABILITY_ERASER;
    peninfo.max_tilt = 90.0f;
    peninfo.num_buttons = 2;
    peninfo.subtype = SDL_PEN_TYPE_PEN;
    SDL_AddPenDevice(0, NULL, &peninfo, (void *) (size_t) event->pointerid);
    Emscripten_UpdatePointerFromEvent(window_data, event);
}

EMSCRIPTEN_KEEPALIVE void Emscripten_HandlePointerLeave(SDL_WindowData *window_data, const Emscripten_PointerEvent *event)
{
    const SDL_PenID pen = SDL_FindPenByHandle((void *) (size_t) event->pointerid);
    if (pen) {
        Emscripten_UpdatePointerFromEvent(window_data, event);  // last data updates?
        SDL_RemovePenDevice(0, pen);
    }
}

EMSCRIPTEN_KEEPALIVE void Emscripten_HandlePointerGeneric(SDL_WindowData *window_data, const Emscripten_PointerEvent *event)
{
    Emscripten_UpdatePointerFromEvent(window_data, event);
}

static void Emscripten_set_pointer_event_callbacks(SDL_WindowData *data)
{
    MAIN_THREAD_EM_ASM({
        var target = document.querySelector(UTF8ToString($1));
        if (target) {
            var data = $0;

            if (typeof(Module['SDL3']) === 'undefined') {
                Module['SDL3'] = {};
            }
            var SDL3 = Module['SDL3'];

            var makePointerEventCStruct = function(event) {
                var ptr = 0;
                if (event.pointerType == "pen") {
                    ptr = _SDL_malloc($2);
                    if (ptr != 0) {
                        var rect = target.getBoundingClientRect();
                        var idx = ptr >> 2;
                        HEAP32[idx++] = event.pointerId;
                        HEAP32[idx++] = (typeof(event.button) !== "undefined") ? event.button : -1;
                        HEAP32[idx++] = event.buttons;
                        HEAPF32[idx++] = event.movementX;
                        HEAPF32[idx++] = event.movementY;
                        HEAPF32[idx++] = event.clientX - rect.left;
                        HEAPF32[idx++] = event.clientY - rect.top;
                        HEAPF32[idx++] = event.pressure;
                        HEAPF32[idx++] = event.tangentialPressure;
                        HEAPF32[idx++] = event.tiltX;
                        HEAPF32[idx++] = event.tiltY;
                        HEAPF32[idx++] = event.twist;
                    }
                }
                return ptr;
            };

            SDL3.eventHandlerPointerEnter = function(event) {
                var d = makePointerEventCStruct(event); if (d != 0) { _Emscripten_HandlePointerEnter(data, d); _SDL_free(d); }
            };
            target.addEventListener("pointerenter", SDL3.eventHandlerPointerEnter);

            SDL3.eventHandlerPointerLeave = function(event) {
                var d = makePointerEventCStruct(event); if (d != 0) { _Emscripten_HandlePointerLeave(data, d); _SDL_free(d); }
            };
            target.addEventListener("pointerleave", SDL3.eventHandlerPointerLeave);
            target.addEventListener("pointercancel", SDL3.eventHandlerPointerLeave);  // catch this, just in case.

            SDL3.eventHandlerPointerGeneric = function(event) {
                var d = makePointerEventCStruct(event); if (d != 0) { _Emscripten_HandlePointerGeneric(data, d); _SDL_free(d); }
            };
            target.addEventListener("pointerdown", SDL3.eventHandlerPointerGeneric);
            target.addEventListener("pointerup", SDL3.eventHandlerPointerGeneric);
            target.addEventListener("pointermove", SDL3.eventHandlerPointerGeneric);
        }
    }, data, data->canvas_id, sizeof (Emscripten_PointerEvent));
}

static void Emscripten_unset_pointer_event_callbacks(SDL_WindowData *data)
{
    MAIN_THREAD_EM_ASM({
        var target = document.querySelector(UTF8ToString($0));
        if (target) {
            var SDL3 = Module['SDL3'];
            target.removeEventListener("pointerenter", SDL3.eventHandlerPointerEnter);
            target.removeEventListener("pointerleave", SDL3.eventHandlerPointerLeave);
            target.removeEventListener("pointercancel", SDL3.eventHandlerPointerLeave);
            target.removeEventListener("pointerdown", SDL3.eventHandlerPointerGeneric);
            target.removeEventListener("pointerup", SDL3.eventHandlerPointerGeneric);
            target.removeEventListener("pointermove", SDL3.eventHandlerPointerGeneric);
            SDL3.eventHandlerPointerEnter = undefined;
            SDL3.eventHandlerPointerLeave = undefined;
            SDL3.eventHandlerPointerGeneric = undefined;
        }
    }, data->canvas_id);
}

// IF YOU CHANGE THIS STRUCTURE, YOU NEED TO UPDATE THE JAVASCRIPT THAT FILLS IT IN: makeDropEventCStruct, below.
typedef struct Emscripten_DropEvent
{
    int x;
    int y;
} Emscripten_DropEvent;

EMSCRIPTEN_KEEPALIVE void Emscripten_SendDragEvent(SDL_WindowData *window_data, const Emscripten_DropEvent *event)
{
    SDL_SendDropPosition(window_data->window, event->x, event->y);
}

EMSCRIPTEN_KEEPALIVE void Emscripten_SendDragCompleteEvent(SDL_WindowData *window_data)
{
    SDL_SendDropComplete(window_data->window);
}

EMSCRIPTEN_KEEPALIVE void Emscripten_SendDragTextEvent(SDL_WindowData *window_data, char *text)
{
    SDL_SendDropText(window_data->window, text);
}

EMSCRIPTEN_KEEPALIVE void Emscripten_SendDragFileEvent(SDL_WindowData *window_data, char *filename)
{
    SDL_SendDropFile(window_data->window, NULL, filename);
}

EM_JS_DEPS(dragndrop, "$writeArrayToMemory");

static void Emscripten_set_drag_event_callbacks(SDL_WindowData *data)
{
    MAIN_THREAD_EM_ASM({
        var target = document.querySelector(UTF8ToString($1));
        if (target) {
            var data = $0;

            if (typeof(Module['SDL3']) === 'undefined') {
                Module['SDL3'] = {};
            }
            var SDL3 = Module['SDL3'];

            var makeDropEventCStruct = function(event) {
                var ptr = 0;
                ptr = _SDL_malloc($2);
                if (ptr != 0) {
                    var idx = ptr >> 2;
                    var rect = target.getBoundingClientRect();
                    HEAP32[idx++] = event.clientX - rect.left;
                    HEAP32[idx++] = event.clientY - rect.top;
                }
                return ptr;
            };

            SDL3.eventHandlerDropDragover = function(event) {
                event.preventDefault();
                var d = makeDropEventCStruct(event); if (d != 0) { _Emscripten_SendDragEvent(data, d); _SDL_free(d); }
            };
            target.addEventListener("dragover", SDL3.eventHandlerDropDragover);

            SDL3.drop_count = 0;
            FS.mkdir("/tmp/filedrop");
            SDL3.eventHandlerDropDrop = function(event) {
                event.preventDefault();
                if (event.dataTransfer.types.includes("text/plain")) {
                    let plain_text = stringToNewUTF8(event.dataTransfer.getData("text/plain"));
                    _Emscripten_SendDragTextEvent(data, plain_text);
                    _free(plain_text);
                } else if (event.dataTransfer.types.includes("Files")) {
                    for (let i = 0; i < event.dataTransfer.files.length; i++) {
                        const file = event.dataTransfer.files.item(i);
                        const file_reader = new FileReader();
                        file_reader.readAsArrayBuffer(file);
                        file_reader.onload = function(event) {
                            const fs_dropdir = `/tmp/filedrop/${SDL3.drop_count}`;
                            SDL3.drop_count += 1;

                            const fs_filepath = `${fs_dropdir}/${file.name}`;
                            const c_fs_filepath = stringToNewUTF8(fs_filepath);
                            const contents_array8 = new Uint8Array(event.target.result);

                            FS.mkdir(fs_dropdir);
                            var stream = FS.open(fs_filepath, "w");
                            FS.write(stream, contents_array8, 0, contents_array8.length, 0);
                            FS.close(stream);

                            _Emscripten_SendDragFileEvent(data, c_fs_filepath);
                            _free(c_fs_filepath);
                            _Emscripten_SendDragCompleteEvent(data);
                        };
                    }
                }
                _Emscripten_SendDragCompleteEvent(data);
            };
            target.addEventListener("drop", SDL3.eventHandlerDropDrop);

            SDL3.eventHandlerDropDragend = function(event) {
                event.preventDefault();
                _Emscripten_SendDragCompleteEvent(data);
            };
            target.addEventListener("dragend", SDL3.eventHandlerDropDragend);
            target.addEventListener("dragleave", SDL3.eventHandlerDropDragend);
        }
    }, data, data->canvas_id, sizeof (Emscripten_DropEvent));
}

static void Emscripten_unset_drag_event_callbacks(SDL_WindowData *data)
{
    MAIN_THREAD_EM_ASM({
        var target = document.querySelector(UTF8ToString($0));
        if (target) {
            var SDL3 = Module['SDL3'];
            target.removeEventListener("dragleave", SDL3.eventHandlerDropDragend);
            target.removeEventListener("dragend", SDL3.eventHandlerDropDragend);
            target.removeEventListener("drop", SDL3.eventHandlerDropDrop);
            SDL3.drop_count = undefined;

            function recursive_remove(dirpath) {
                FS.readdir(dirpath).forEach((filename) => {
                    const p = `${dirpath}/${filename}`;
                    const p_s = FS.stat(p);
                    if (FS.isFile(p_s.mode)) {
                        FS.unlink(p);
                    } else if (FS.isDir(p)) {
                        recursive_remove(p);
                    }
                });
                FS.rmdir(dirpath);
            }("/tmp/filedrop");

            FS.rmdir("/tmp/filedrop");
            target.removeEventListener("dragover", SDL3.eventHandlerDropDragover);
            SDL3.eventHandlerDropDragover = undefined;
            SDL3.eventHandlerDropDrop = undefined;
            SDL3.eventHandlerDropDragend = undefined;
        }
    }, data->canvas_id);
}

static const char *Emscripten_GetKeyboardTargetElement(const char *target)
{
    if (SDL_strcmp(target, "#none") == 0) {
        return NULL;
    } else if (SDL_strcmp(target, "#window") == 0) {
        return EMSCRIPTEN_EVENT_TARGET_WINDOW;
    } else if (SDL_strcmp(target, "#document") == 0) {
        return EMSCRIPTEN_EVENT_TARGET_DOCUMENT;
    } else if (SDL_strcmp(target, "#screen") == 0) {
        return EMSCRIPTEN_EVENT_TARGET_SCREEN;
    }

    return target;
}

void Emscripten_RegisterGlobalEventHandlers(SDL_VideoDevice *device)
{
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, device, 0, Emscripten_HandleMouseButtonGlobal);

    emscripten_set_focus_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, device, 0, Emscripten_HandleFocus);
    emscripten_set_blur_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, device, 0, Emscripten_HandleFocus);

    emscripten_set_pointerlockchange_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, device, 0, Emscripten_HandlePointerLockChangeGlobal);

    emscripten_set_fullscreenchange_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, device, 0, Emscripten_HandleFullscreenChangeGlobal);

    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, device, 0, Emscripten_HandleResizeGlobal);
}

void Emscripten_UnregisterGlobalEventHandlers(SDL_VideoDevice *device)
{
    emscripten_set_mouseup_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, NULL, 0, NULL);

    emscripten_set_focus_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, NULL);
    emscripten_set_blur_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, NULL);

    emscripten_set_pointerlockchange_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, NULL, 0, NULL);

    emscripten_set_fullscreenchange_callback(EMSCRIPTEN_EVENT_TARGET_DOCUMENT, NULL, 0, NULL);

    emscripten_set_resize_callback(EMSCRIPTEN_EVENT_TARGET_WINDOW, NULL, 0, NULL);
}

void Emscripten_RegisterEventHandlers(SDL_WindowData *data)
{
    const char *keyElement;

    // There is only one window and that window is the canvas
    emscripten_set_mousemove_callback(data->canvas_id, data, 0, Emscripten_HandleMouseMove);

    emscripten_set_mousedown_callback(data->canvas_id, data, 0, Emscripten_HandleMouseButton);

    emscripten_set_mouseenter_callback(data->canvas_id, data, 0, Emscripten_HandleMouseFocus);
    emscripten_set_mouseleave_callback(data->canvas_id, data, 0, Emscripten_HandleMouseFocus);

    emscripten_set_wheel_callback(data->canvas_id, data, 0, Emscripten_HandleWheel);

    emscripten_set_orientationchange_callback(data, 0, Emscripten_HandleOrientationChange);

    emscripten_set_touchstart_callback(data->canvas_id, data, 0, Emscripten_HandleTouch);
    emscripten_set_touchend_callback(data->canvas_id, data, 0, Emscripten_HandleTouch);
    emscripten_set_touchmove_callback(data->canvas_id, data, 0, Emscripten_HandleTouch);
    emscripten_set_touchcancel_callback(data->canvas_id, data, 0, Emscripten_HandleTouch);

    keyElement = Emscripten_GetKeyboardTargetElement(data->keyboard_element);
    if (keyElement) {
        emscripten_set_keydown_callback(keyElement, data, 0, Emscripten_HandleKey);
        emscripten_set_keyup_callback(keyElement, data, 0, Emscripten_HandleKey);
        emscripten_set_keypress_callback(keyElement, data, 0, Emscripten_HandleKeyPress);
    }

    emscripten_set_visibilitychange_callback(data, 0, Emscripten_HandleVisibilityChange);

    emscripten_set_beforeunload_callback(data, Emscripten_HandleBeforeUnload);

    // !!! FIXME: currently Emscripten doesn't have a Pointer Events functions like emscripten_set_*_callback, but we should use those when they do:
    // !!! FIXME:  https://github.com/emscripten-core/emscripten/issues/7278#issuecomment-2280024621
    Emscripten_set_pointer_event_callbacks(data);

    // !!! FIXME: currently Emscripten doesn't have a Drop Events functions like emscripten_set_*_callback, but we should use those when they do:
    Emscripten_set_drag_event_callbacks(data);
}

void Emscripten_UnregisterEventHandlers(SDL_WindowData *data)
{
    const char *keyElement;

    // !!! FIXME: currently Emscripten doesn't have a Drop Events functions like emscripten_set_*_callback, but we should use those when they do:
    Emscripten_unset_drag_event_callbacks(data);

    // !!! FIXME: currently Emscripten doesn't have a Pointer Events functions like emscripten_set_*_callback, but we should use those when they do:
    // !!! FIXME:  https://github.com/emscripten-core/emscripten/issues/7278#issuecomment-2280024621
    Emscripten_unset_pointer_event_callbacks(data);

    // only works due to having one window
    emscripten_set_mousemove_callback(data->canvas_id, NULL, 0, NULL);

    emscripten_set_mousedown_callback(data->canvas_id, NULL, 0, NULL);

    emscripten_set_mouseenter_callback(data->canvas_id, NULL, 0, NULL);
    emscripten_set_mouseleave_callback(data->canvas_id, NULL, 0, NULL);

    emscripten_set_wheel_callback(data->canvas_id, NULL, 0, NULL);

    emscripten_set_orientationchange_callback(NULL, 0, NULL);

    emscripten_set_touchstart_callback(data->canvas_id, NULL, 0, NULL);
    emscripten_set_touchend_callback(data->canvas_id, NULL, 0, NULL);
    emscripten_set_touchmove_callback(data->canvas_id, NULL, 0, NULL);
    emscripten_set_touchcancel_callback(data->canvas_id, NULL, 0, NULL);

    keyElement = Emscripten_GetKeyboardTargetElement(data->keyboard_element);
    if (keyElement) {
        emscripten_set_keydown_callback(keyElement, NULL, 0, NULL);
        emscripten_set_keyup_callback(keyElement, NULL, 0, NULL);
        emscripten_set_keypress_callback(keyElement, NULL, 0, NULL);
    }

    emscripten_set_visibilitychange_callback(NULL, 0, NULL);

    emscripten_set_beforeunload_callback(NULL, NULL);
}

#endif // SDL_VIDEO_DRIVER_EMSCRIPTEN
