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

#ifdef SDL_VIDEO_DRIVER_WINDOWS

#include "SDL_windowsvideo.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_touch_c.h"
#include "../../events/scancodes_windows.h"
#include "../../main/SDL_main_callbacks.h"
#include "../../core/windows/SDL_hid.h"

// Dropfile support
#include <shellapi.h>

// Device names
#include <setupapi.h>

// For GET_X_LPARAM, GET_Y_LPARAM.
#include <windowsx.h>

// For WM_TABLET_QUERYSYSTEMGESTURESTATUS et. al.
#ifdef HAVE_TPCSHRD_H
#include <tpcshrd.h>
#endif // HAVE_TPCSHRD_H

#if 0
#define WMMSG_DEBUG
#endif
#ifdef WMMSG_DEBUG
#include <stdio.h>
#include "wmmsg.h"
#endif

#ifdef HAVE_SHOBJIDL_CORE_H
#include <shobjidl_core.h>
#endif

#ifdef SDL_PLATFORM_GDK
#include "../../core/gdk/SDL_gdk.h"
#endif

// #define HIGHDPI_DEBUG

// Make sure XBUTTON stuff is defined that isn't in older Platform SDKs...
#ifndef WM_XBUTTONDOWN
#define WM_XBUTTONDOWN 0x020B
#endif
#ifndef WM_XBUTTONUP
#define WM_XBUTTONUP 0x020C
#endif
#ifndef GET_XBUTTON_WPARAM
#define GET_XBUTTON_WPARAM(w) (HIWORD(w))
#endif
#ifndef WM_INPUT
#define WM_INPUT 0x00ff
#endif
#ifndef WM_TOUCH
#define WM_TOUCH 0x0240
#endif
#ifndef WM_MOUSEHWHEEL
#define WM_MOUSEHWHEEL 0x020E
#endif
#ifndef RI_MOUSE_HWHEEL
#define RI_MOUSE_HWHEEL 0x0800
#endif
#ifndef WM_POINTERUPDATE
#define WM_POINTERUPDATE 0x0245
#endif
#ifndef WM_POINTERDOWN
#define WM_POINTERDOWN 0x0246
#endif
#ifndef WM_POINTERUP
#define WM_POINTERUP 0x0247
#endif
#ifndef WM_POINTERENTER
#define WM_POINTERENTER 0x0249
#endif
#ifndef WM_POINTERLEAVE
#define WM_POINTERLEAVE 0x024A
#endif
#ifndef WM_POINTERCAPTURECHANGED
#define WM_POINTERCAPTURECHANGED 0x024C
#endif
#ifndef WM_UNICHAR
#define WM_UNICHAR 0x0109
#endif
#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0
#endif
#ifndef WM_GETDPISCALEDSIZE
#define WM_GETDPISCALEDSIZE 0x02E4
#endif
#ifndef TOUCHEVENTF_PEN
#define TOUCHEVENTF_PEN 0x0040
#endif

#ifndef MAPVK_VK_TO_VSC_EX
#define MAPVK_VK_TO_VSC_EX 4
#endif

#ifndef WC_ERR_INVALID_CHARS
#define WC_ERR_INVALID_CHARS 0x00000080
#endif

#ifndef IS_HIGH_SURROGATE
#define IS_HIGH_SURROGATE(x) (((x) >= 0xd800) && ((x) <= 0xdbff))
#endif

#ifndef USER_TIMER_MINIMUM
#define USER_TIMER_MINIMUM 0x0000000A
#endif

// Used to compare Windows message timestamps
#define SDL_TICKS_PASSED(A, B) ((Sint32)((B) - (A)) <= 0)

#ifdef _WIN64
typedef Uint64 QWORD; // Needed for NEXTRAWINPUTBLOCK()
#endif

static bool SDL_processing_messages;
static DWORD message_tick;
static Uint64 timestamp_offset;

static void WIN_SetMessageTick(DWORD tick)
{
    message_tick = tick;
}

static Uint64 WIN_GetEventTimestamp(void)
{
    const Uint64 TIMESTAMP_WRAP_OFFSET = SDL_MS_TO_NS(0x100000000LL);
    Uint64 timestamp, now;

    if (!SDL_processing_messages) {
        // message_tick isn't valid, just use the current time
        return 0;
    }

    now = SDL_GetTicksNS();
    timestamp = SDL_MS_TO_NS(message_tick);
    timestamp += timestamp_offset;
    if (!timestamp_offset) {
        // Initializing timestamp offset
        //SDL_Log("Initializing timestamp offset");
        timestamp_offset = (now - timestamp);
        timestamp = now;
    } else if ((Sint64)(now - timestamp - TIMESTAMP_WRAP_OFFSET) >= 0) {
        // The windows message tick wrapped
        //SDL_Log("Adjusting timestamp offset for wrapping tick");
        timestamp_offset += TIMESTAMP_WRAP_OFFSET;
        timestamp += TIMESTAMP_WRAP_OFFSET;
    } else if (timestamp > now) {
        // We got a newer timestamp, but it can't be newer than now, so adjust our offset
        //SDL_Log("Adjusting timestamp offset, %.2f ms newer", (double)(timestamp - now) / SDL_NS_PER_MS);
        timestamp_offset -= (timestamp - now);
        timestamp = now;
    }
    return timestamp;
}

// A message hook called before TranslateMessage()
static SDL_WindowsMessageHook g_WindowsMessageHook = NULL;
static void *g_WindowsMessageHookData = NULL;

void SDL_SetWindowsMessageHook(SDL_WindowsMessageHook callback, void *userdata)
{
    g_WindowsMessageHook = callback;
    g_WindowsMessageHookData = userdata;
}

static SDL_Scancode WindowsScanCodeToSDLScanCode(LPARAM lParam, WPARAM wParam, Uint16 *rawcode, bool *virtual_key)
{
    SDL_Scancode code;
    Uint8 index;
    Uint16 keyFlags = HIWORD(lParam);
    Uint16 scanCode = LOBYTE(keyFlags);

    /* On-Screen Keyboard can send wrong scan codes with high-order bit set (key break code).
     * Strip high-order bit. */
    scanCode &= ~0x80;

    *virtual_key = (scanCode == 0);

    if (scanCode != 0) {
        if ((keyFlags & KF_EXTENDED) == KF_EXTENDED) {
            scanCode = MAKEWORD(scanCode, 0xe0);
        } else if (scanCode == 0x45) {
            // Pause
            scanCode = 0xe046;
        }
    } else {
        Uint16 vkCode = LOWORD(wParam);

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        /* Windows may not report scan codes for some buttons (multimedia buttons etc).
         * Get scan code from the VK code.*/
        scanCode = LOWORD(MapVirtualKey(vkCode, WIN_IsWindowsXP() ? MAPVK_VK_TO_VSC : MAPVK_VK_TO_VSC_EX));
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

        /* Pause/Break key have a special scan code with 0xe1 prefix.
         * Use Pause scan code that is used in Win32. */
        if (scanCode == 0xe11d) {
            scanCode = 0xe046;
        }
    }

    // Pack scan code into one byte to make the index.
    index = LOBYTE(scanCode) | (HIBYTE(scanCode) ? 0x80 : 0x00);
    code = windows_scancode_table[index];
    *rawcode = scanCode;

    return code;
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static bool WIN_ShouldIgnoreFocusClick(SDL_WindowData *data)
{
    return !SDL_WINDOW_IS_POPUP(data->window) &&
           !SDL_GetHintBoolean(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, false);
}

static void WIN_CheckWParamMouseButton(Uint64 timestamp, bool bwParamMousePressed, Uint32 mouseFlags, bool bSwapButtons, SDL_WindowData *data, Uint8 button, SDL_MouseID mouseID)
{
    if (bSwapButtons) {
        if (button == SDL_BUTTON_LEFT) {
            button = SDL_BUTTON_RIGHT;
        } else if (button == SDL_BUTTON_RIGHT) {
            button = SDL_BUTTON_LEFT;
        }
    }

    if (data->focus_click_pending & SDL_BUTTON_MASK(button)) {
        // Ignore the button click for activation
        if (!bwParamMousePressed) {
            data->focus_click_pending &= ~SDL_BUTTON_MASK(button);
            WIN_UpdateClipCursor(data->window);
        }
        return;
    }

    if (bwParamMousePressed && !(mouseFlags & SDL_BUTTON_MASK(button))) {
        SDL_SendMouseButton(timestamp, data->window, mouseID, button, true);
    } else if (!bwParamMousePressed && (mouseFlags & SDL_BUTTON_MASK(button))) {
        SDL_SendMouseButton(timestamp, data->window, mouseID, button, false);
    }
}

/*
 * Some windows systems fail to send a WM_LBUTTONDOWN sometimes, but each mouse move contains the current button state also
 *  so this function reconciles our view of the world with the current buttons reported by windows
 */
static void WIN_CheckWParamMouseButtons(Uint64 timestamp, WPARAM wParam, SDL_WindowData *data, SDL_MouseID mouseID)
{
    if (wParam != data->mouse_button_flags) {
        SDL_MouseButtonFlags mouseFlags = SDL_GetMouseState(NULL, NULL);

        // WM_LBUTTONDOWN and friends handle button swapping for us. No need to check SM_SWAPBUTTON here.
        WIN_CheckWParamMouseButton(timestamp, (wParam & MK_LBUTTON), mouseFlags, false, data, SDL_BUTTON_LEFT, mouseID);
        WIN_CheckWParamMouseButton(timestamp, (wParam & MK_MBUTTON), mouseFlags, false, data, SDL_BUTTON_MIDDLE, mouseID);
        WIN_CheckWParamMouseButton(timestamp, (wParam & MK_RBUTTON), mouseFlags, false, data, SDL_BUTTON_RIGHT, mouseID);
        WIN_CheckWParamMouseButton(timestamp, (wParam & MK_XBUTTON1), mouseFlags, false, data, SDL_BUTTON_X1, mouseID);
        WIN_CheckWParamMouseButton(timestamp, (wParam & MK_XBUTTON2), mouseFlags, false, data, SDL_BUTTON_X2, mouseID);

        data->mouse_button_flags = wParam;
    }
}

static void WIN_CheckAsyncMouseRelease(Uint64 timestamp, SDL_WindowData *data)
{
    SDL_MouseID mouseID = SDL_GLOBAL_MOUSE_ID;
    Uint32 mouseFlags;
    SHORT keyState;
    bool swapButtons;

    /* mouse buttons may have changed state here, we need to resync them,
       but we will get a WM_MOUSEMOVE right away which will fix things up if in non raw mode also
    */
    mouseFlags = SDL_GetMouseState(NULL, NULL);
    swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;

    keyState = GetAsyncKeyState(VK_LBUTTON);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(timestamp, false, mouseFlags, swapButtons, data, SDL_BUTTON_LEFT, mouseID);
    }
    keyState = GetAsyncKeyState(VK_RBUTTON);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(timestamp, false, mouseFlags, swapButtons, data, SDL_BUTTON_RIGHT, mouseID);
    }
    keyState = GetAsyncKeyState(VK_MBUTTON);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(timestamp, false, mouseFlags, swapButtons, data, SDL_BUTTON_MIDDLE, mouseID);
    }
    keyState = GetAsyncKeyState(VK_XBUTTON1);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(timestamp, false, mouseFlags, swapButtons, data, SDL_BUTTON_X1, mouseID);
    }
    keyState = GetAsyncKeyState(VK_XBUTTON2);
    if (!(keyState & 0x8000)) {
        WIN_CheckWParamMouseButton(timestamp, false, mouseFlags, swapButtons, data, SDL_BUTTON_X2, mouseID);
    }
    data->mouse_button_flags = (WPARAM)-1;
}

static void WIN_UpdateFocus(SDL_Window *window, bool expect_focus, DWORD pos)
{
    SDL_WindowData *data = window->internal;
    HWND hwnd = data->hwnd;
    bool had_focus = (SDL_GetKeyboardFocus() == window);
    bool has_focus = (GetForegroundWindow() == hwnd);

    if (had_focus == has_focus || has_focus != expect_focus) {
        return;
    }

    if (has_focus) {
        POINT cursorPos;

        if (WIN_ShouldIgnoreFocusClick(data) && !(window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
            bool swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;
            if (GetAsyncKeyState(VK_LBUTTON)) {
                data->focus_click_pending |= !swapButtons ? SDL_BUTTON_LMASK : SDL_BUTTON_RMASK;
            }
            if (GetAsyncKeyState(VK_RBUTTON)) {
                data->focus_click_pending |= !swapButtons ? SDL_BUTTON_RMASK : SDL_BUTTON_LMASK;
            }
            if (GetAsyncKeyState(VK_MBUTTON)) {
                data->focus_click_pending |= SDL_BUTTON_MMASK;
            }
            if (GetAsyncKeyState(VK_XBUTTON1)) {
                data->focus_click_pending |= SDL_BUTTON_X1MASK;
            }
            if (GetAsyncKeyState(VK_XBUTTON2)) {
                data->focus_click_pending |= SDL_BUTTON_X2MASK;
            }
        }

        SDL_SetKeyboardFocus(window->keyboard_focus ? window->keyboard_focus : window);

        // In relative mode we are guaranteed to have mouse focus if we have keyboard focus
        if (!SDL_GetMouse()->relative_mode) {
            cursorPos.x = (LONG)GET_X_LPARAM(pos);
            cursorPos.y = (LONG)GET_Y_LPARAM(pos);
            ScreenToClient(hwnd, &cursorPos);
            SDL_SendMouseMotion(WIN_GetEventTimestamp(), window, SDL_GLOBAL_MOUSE_ID, false, (float)cursorPos.x, (float)cursorPos.y);
        }

        WIN_CheckAsyncMouseRelease(WIN_GetEventTimestamp(), data);
        WIN_UpdateClipCursor(window);

        /*
         * FIXME: Update keyboard state
         */
        WIN_CheckClipboardUpdate(data->videodata);

        SDL_ToggleModState(SDL_KMOD_CAPS, (GetKeyState(VK_CAPITAL) & 0x0001) ? true : false);
        SDL_ToggleModState(SDL_KMOD_NUM, (GetKeyState(VK_NUMLOCK) & 0x0001) ? true : false);
        SDL_ToggleModState(SDL_KMOD_SCROLL, (GetKeyState(VK_SCROLL) & 0x0001) ? true : false);

        WIN_UpdateWindowICCProfile(data->window, true);
    } else {
        data->in_window_deactivation = true;

        SDL_SetKeyboardFocus(NULL);
        // In relative mode we are guaranteed to not have mouse focus if we don't have keyboard focus
        if (SDL_GetMouse()->relative_mode) {
            SDL_SetMouseFocus(NULL);
        }
        WIN_ResetDeadKeys();

        WIN_UnclipCursorForWindow(window);

        data->in_window_deactivation = false;
    }
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

static bool ShouldGenerateWindowCloseOnAltF4(void)
{
    return SDL_GetHintBoolean(SDL_HINT_WINDOWS_CLOSE_ON_ALT_F4, true);
}

static bool ShouldClearWindowOnEraseBackground(SDL_WindowData *data)
{
    switch (data->hint_erase_background_mode) {
    case SDL_ERASEBACKGROUNDMODE_NEVER:
        return false;
    case SDL_ERASEBACKGROUNDMODE_INITIAL:
        return !data->videodata->cleared;
    case SDL_ERASEBACKGROUNDMODE_ALWAYS:
        return true;
    default:
        // Unexpected value, fallback to default behaviour
        return !data->videodata->cleared;
    }
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
// We want to generate mouse events from mouse and pen, and touch events from touchscreens
#define MI_WP_SIGNATURE      0xFF515700
#define MI_WP_SIGNATURE_MASK 0xFFFFFF00
#define IsTouchEvent(dw)     ((dw)&MI_WP_SIGNATURE_MASK) == MI_WP_SIGNATURE

typedef enum
{
    SDL_MOUSE_EVENT_SOURCE_UNKNOWN,
    SDL_MOUSE_EVENT_SOURCE_MOUSE,
    SDL_MOUSE_EVENT_SOURCE_TOUCH,
    SDL_MOUSE_EVENT_SOURCE_PEN,
} SDL_MOUSE_EVENT_SOURCE;

static SDL_MOUSE_EVENT_SOURCE GetMouseMessageSource(ULONG extrainfo)
{
    // Mouse data (ignoring synthetic mouse events generated for touchscreens)
    /* Versions below Vista will set the low 7 bits to the Mouse ID and don't use bit 7:
       Check bits 8-31 for the signature (which will indicate a Tablet PC Pen or Touch Device).
       Only check bit 7 when Vista and up(Cleared=Pen, Set=Touch(which we need to filter out)),
       when the signature is set. The Mouse ID will be zero for an actual mouse. */
    if (IsTouchEvent(extrainfo)) {
        if (extrainfo & 0x80) {
            return SDL_MOUSE_EVENT_SOURCE_TOUCH;
        } else {
            return SDL_MOUSE_EVENT_SOURCE_PEN;
        }
    }
    /* Sometimes WM_INPUT events won't have the correct touch signature,
      so we have to rely purely on the touch bit being set. */
    if (SDL_TouchDevicesAvailable() && extrainfo & 0x80) {
        return SDL_MOUSE_EVENT_SOURCE_TOUCH;
    }
    return SDL_MOUSE_EVENT_SOURCE_MOUSE;
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

static SDL_WindowData *WIN_GetWindowDataFromHWND(HWND hwnd)
{
    SDL_VideoDevice *_this = SDL_GetVideoDevice();
    SDL_Window *window;

    if (_this) {
        for (window = _this->windows; window; window = window->next) {
            SDL_WindowData *data = window->internal;
            if (data && data->hwnd == hwnd) {
                return data;
            }
        }
    }
    return NULL;
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
LRESULT CALLBACK
WIN_KeyboardHookProc(int nCode, WPARAM wParam, LPARAM lParam)
{
    KBDLLHOOKSTRUCT *hookData = (KBDLLHOOKSTRUCT *)lParam;
    SDL_VideoData *data = SDL_GetVideoDevice()->internal;
    SDL_Scancode scanCode;

    if (nCode < 0 || nCode != HC_ACTION) {
        return CallNextHookEx(NULL, nCode, wParam, lParam);
    }
    if (hookData->scanCode == 0x21d) {
        // Skip fake LCtrl when RAlt is pressed
        return 1;
    }

    switch (hookData->vkCode) {
    case VK_LWIN:
        scanCode = SDL_SCANCODE_LGUI;
        break;
    case VK_RWIN:
        scanCode = SDL_SCANCODE_RGUI;
        break;
    case VK_LMENU:
        scanCode = SDL_SCANCODE_LALT;
        break;
    case VK_RMENU:
        scanCode = SDL_SCANCODE_RALT;
        break;
    case VK_LCONTROL:
        scanCode = SDL_SCANCODE_LCTRL;
        break;
    case VK_RCONTROL:
        scanCode = SDL_SCANCODE_RCTRL;
        break;

    // These are required to intercept Alt+Tab and Alt+Esc on Windows 7
    case VK_TAB:
        scanCode = SDL_SCANCODE_TAB;
        break;
    case VK_ESCAPE:
        scanCode = SDL_SCANCODE_ESCAPE;
        break;

    default:
        return CallNextHookEx(NULL, nCode, wParam, lParam);
    }

    if (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN) {
        if (!data->raw_keyboard_enabled) {
            SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, hookData->scanCode, scanCode, true);
        }
    } else {
        if (!data->raw_keyboard_enabled) {
            SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, hookData->scanCode, scanCode, false);
        }

        /* If the key was down prior to our hook being installed, allow the
           key up message to pass normally the first time. This ensures other
           windows have a consistent view of the key state, and avoids keys
           being stuck down in those windows if they are down when the grab
           happens and raised while grabbed. */
        if (hookData->vkCode <= 0xFF && data->pre_hook_key_state[hookData->vkCode]) {
            data->pre_hook_key_state[hookData->vkCode] = 0;
            return CallNextHookEx(NULL, nCode, wParam, lParam);
        }
    }

    return 1;
}

static bool WIN_SwapButtons(HANDLE hDevice)
{
    if (hDevice == NULL) {
        // Touchpad, already has buttons swapped
        return false;
    }
    return GetSystemMetrics(SM_SWAPBUTTON) != 0;
}

static void WIN_HandleRawMouseInput(Uint64 timestamp, SDL_VideoData *data, HANDLE hDevice, RAWMOUSE *rawmouse)
{
    static struct {
        USHORT usButtonFlags;
        Uint8 button;
        bool down;
    } raw_buttons[] = {
        { RI_MOUSE_LEFT_BUTTON_DOWN, SDL_BUTTON_LEFT, true },
        { RI_MOUSE_LEFT_BUTTON_UP, SDL_BUTTON_LEFT, false },
        { RI_MOUSE_RIGHT_BUTTON_DOWN, SDL_BUTTON_RIGHT, true },
        { RI_MOUSE_RIGHT_BUTTON_UP, SDL_BUTTON_RIGHT, false },
        { RI_MOUSE_MIDDLE_BUTTON_DOWN, SDL_BUTTON_MIDDLE, true },
        { RI_MOUSE_MIDDLE_BUTTON_UP, SDL_BUTTON_MIDDLE, false },
        { RI_MOUSE_BUTTON_4_DOWN, SDL_BUTTON_X1, true },
        { RI_MOUSE_BUTTON_4_UP, SDL_BUTTON_X1, false },
        { RI_MOUSE_BUTTON_5_DOWN, SDL_BUTTON_X2, true },
        { RI_MOUSE_BUTTON_5_UP, SDL_BUTTON_X2, false }
    };

    int dx = (int)rawmouse->lLastX;
    int dy = (int)rawmouse->lLastY;
    bool haveMotion = (dx || dy);
    bool haveButton = (rawmouse->usButtonFlags != 0);
    bool isAbsolute = ((rawmouse->usFlags & MOUSE_MOVE_ABSOLUTE) != 0);
    SDL_MouseID mouseID = (SDL_MouseID)(uintptr_t)hDevice;

    // Check whether relative mode should also receive events from the rawinput stream
    if (!data->raw_mouse_enabled) {
        return;
    }

    // Relative mouse motion is delivered to the window with keyboard focus
    SDL_Window *window = SDL_GetKeyboardFocus();
    if (!window) {
        return;
    }

    if (GetMouseMessageSource(rawmouse->ulExtraInformation) != SDL_MOUSE_EVENT_SOURCE_MOUSE) {
        return;
    }

    SDL_WindowData *windowdata = window->internal;

    if (haveMotion && !windowdata->in_modal_loop) {
        if (!isAbsolute) {
            SDL_SendMouseMotion(timestamp, window, mouseID, true, (float)dx, (float)dy);
        } else {
            /* This is absolute motion, either using a tablet or mouse over RDP

                Notes on how RDP appears to work, as of Windows 10 2004:
                - SetCursorPos() calls are cached, with multiple calls coalesced into a single call that's sent to the RDP client. If the last call to SetCursorPos() has the same value as the last one that was sent to the client, it appears to be ignored and not sent. This means that we need to jitter the SetCursorPos() position slightly in order for the recentering to work correctly.
                - User mouse motion is coalesced with SetCursorPos(), so the WM_INPUT positions we see will not necessarily match the position we requested with SetCursorPos().
                - SetCursorPos() outside of the bounds of the focus window appears not to do anything.
                - SetCursorPos() while the cursor is NULL doesn't do anything

                We handle this by creating a safe area within the application window, and when the mouse leaves that safe area, we warp back to the opposite side. Any single motion > 50% of the safe area is assumed to be a warp and ignored.
            */
            bool remote_desktop = (GetSystemMetrics(SM_REMOTESESSION) == TRUE);
            bool virtual_desktop = ((rawmouse->usFlags & MOUSE_VIRTUAL_DESKTOP) != 0);
            bool raw_coordinates = ((rawmouse->usFlags & 0x40) != 0);
            int w = GetSystemMetrics(virtual_desktop ? SM_CXVIRTUALSCREEN : SM_CXSCREEN);
            int h = GetSystemMetrics(virtual_desktop ? SM_CYVIRTUALSCREEN : SM_CYSCREEN);
            int x = raw_coordinates ? dx : (int)(((float)dx / 65535.0f) * w);
            int y = raw_coordinates ? dy : (int)(((float)dy / 65535.0f) * h);
            int relX, relY;

            /* Calculate relative motion */
            if (data->last_raw_mouse_position.x == 0 && data->last_raw_mouse_position.y == 0) {
                data->last_raw_mouse_position.x = x;
                data->last_raw_mouse_position.y = y;
            }
            relX = x - data->last_raw_mouse_position.x;
            relY = y - data->last_raw_mouse_position.y;

            if (remote_desktop) {
                if (!windowdata->in_title_click && !windowdata->focus_click_pending) {
                    static int wobble;
                    float floatX = (float)x / w;
                    float floatY = (float)y / h;

                    /* See if the mouse is at the edge of the screen, or in the RDP title bar area */
                    if (floatX <= 0.01f || floatX >= 0.99f || floatY <= 0.01f || floatY >= 0.99f || y < 32) {
                        /* Wobble the cursor position so it's not ignored if the last warp didn't have any effect */
                        RECT rect = windowdata->cursor_clipped_rect;
                        int warpX = rect.left + ((rect.right - rect.left) / 2) + wobble;
                        int warpY = rect.top + ((rect.bottom - rect.top) / 2);

                        WIN_SetCursorPos(warpX, warpY);

                        ++wobble;
                        if (wobble > 1) {
                            wobble = -1;
                        }
                    } else {
                        /* Send relative motion if we didn't warp last frame (had good position data)
                           We also sometimes get large deltas due to coalesced mouse motion and warping,
                           so ignore those.
                         */
                        const int MAX_RELATIVE_MOTION = (h / 6);
                        if (SDL_abs(relX) < MAX_RELATIVE_MOTION &&
                            SDL_abs(relY) < MAX_RELATIVE_MOTION) {
                            SDL_SendMouseMotion(timestamp, window, mouseID, true, (float)relX, (float)relY);
                        }
                    }
                }
            } else {
                const int MAXIMUM_TABLET_RELATIVE_MOTION = 32;
                if (SDL_abs(relX) > MAXIMUM_TABLET_RELATIVE_MOTION ||
                    SDL_abs(relY) > MAXIMUM_TABLET_RELATIVE_MOTION) {
                    /* Ignore this motion, probably a pen lift and drop */
                } else {
                    SDL_SendMouseMotion(timestamp, window, mouseID, true, (float)relX, (float)relY);
                }
            }

            data->last_raw_mouse_position.x = x;
            data->last_raw_mouse_position.y = y;
        }
    }

    if (haveButton) {
        for (int i = 0; i < SDL_arraysize(raw_buttons); ++i) {
            if (rawmouse->usButtonFlags & raw_buttons[i].usButtonFlags) {
                Uint8 button = raw_buttons[i].button;
                bool down = raw_buttons[i].down;

                if (button == SDL_BUTTON_LEFT) {
                    if (WIN_SwapButtons(hDevice)) {
                        button = SDL_BUTTON_RIGHT;
                    }
                } else if (button == SDL_BUTTON_RIGHT) {
                    if (WIN_SwapButtons(hDevice)) {
                        button = SDL_BUTTON_LEFT;
                    }
                }

                if (windowdata->focus_click_pending & SDL_BUTTON_MASK(button)) {
                    // Ignore the button click for activation
                    if (!down) {
                        windowdata->focus_click_pending &= ~SDL_BUTTON_MASK(button);
                        WIN_UpdateClipCursor(window);
                    }
                    continue;
                }

                SDL_SendMouseButton(timestamp, window, mouseID, button, down);
            }
        }

        if (rawmouse->usButtonFlags & RI_MOUSE_WHEEL) {
            SHORT amount = (SHORT)rawmouse->usButtonData;
            float fAmount = (float)amount / WHEEL_DELTA;
            SDL_SendMouseWheel(WIN_GetEventTimestamp(), window, mouseID, 0.0f, fAmount, SDL_MOUSEWHEEL_NORMAL);
        } else if (rawmouse->usButtonFlags & RI_MOUSE_HWHEEL) {
            SHORT amount = (SHORT)rawmouse->usButtonData;
            float fAmount = (float)amount / WHEEL_DELTA;
            SDL_SendMouseWheel(WIN_GetEventTimestamp(), window, mouseID, fAmount, 0.0f, SDL_MOUSEWHEEL_NORMAL);
        }
    }
}

static void WIN_HandleRawKeyboardInput(Uint64 timestamp, SDL_VideoData *data, HANDLE hDevice, RAWKEYBOARD *rawkeyboard)
{
    SDL_KeyboardID keyboardID = (SDL_KeyboardID)(uintptr_t)hDevice;

    if (!data->raw_keyboard_enabled) {
        return;
    }

    if (rawkeyboard->Flags & RI_KEY_E1) {
        // First key in a Ctrl+{key} sequence
        data->pending_E1_key_sequence = true;
        return;
    }

    if ((rawkeyboard->Flags & RI_KEY_E0) && rawkeyboard->MakeCode == 0x2A) {
        // 0xE02A make code prefix, ignored
        return;
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (!rawkeyboard->MakeCode) {
        rawkeyboard->MakeCode = LOWORD(MapVirtualKey(rawkeyboard->VKey, WIN_IsWindowsXP() ? MAPVK_VK_TO_VSC : MAPVK_VK_TO_VSC_EX));
    }
#endif
    if (!rawkeyboard->MakeCode) {
        return;
    }

    bool down = !(rawkeyboard->Flags & RI_KEY_BREAK);
    SDL_Scancode code;
    USHORT rawcode = rawkeyboard->MakeCode;
    if (data->pending_E1_key_sequence) {
        rawcode |= 0xE100;
        if (rawkeyboard->MakeCode == 0x45) {
            // Ctrl+NumLock == Pause
            code = SDL_SCANCODE_PAUSE;
        } else {
            // Ctrl+ScrollLock == Break (no SDL scancode?)
            code = SDL_SCANCODE_UNKNOWN;
        }
        data->pending_E1_key_sequence = false;
    } else {
        // The code is in the lower 7 bits, the high bit is set for the E0 prefix
        Uint8 index = (Uint8)rawkeyboard->MakeCode;
        if (rawkeyboard->Flags & RI_KEY_E0) {
            rawcode |= 0xE000;
            index |= 0x80;
        }
        code = windows_scancode_table[index];
    }

    if (down) {
        SDL_Window *focus = SDL_GetKeyboardFocus();
        if (!focus || focus->text_input_active) {
            return;
        }
    }

    SDL_SendKeyboardKey(timestamp, keyboardID, rawcode, code, down);
}

void WIN_PollRawInput(SDL_VideoDevice *_this, Uint64 poll_start)
{
    SDL_VideoData *data = _this->internal;
    UINT size, i, count, total = 0;
    RAWINPUT *input;
    Uint64 poll_finish;

    if (data->rawinput_offset == 0) {
        BOOL isWow64;

        data->rawinput_offset = sizeof(RAWINPUTHEADER);
        if (IsWow64Process(GetCurrentProcess(), &isWow64) && isWow64) {
            // We're going to get 64-bit data, so use the 64-bit RAWINPUTHEADER size
            data->rawinput_offset += 8;
        }
    }

    // Get all available events
    input = (RAWINPUT *)data->rawinput;
    for (;;) {
        size = data->rawinput_size - (UINT)((BYTE *)input - data->rawinput);
        count = GetRawInputBuffer(input, &size, sizeof(RAWINPUTHEADER));
        poll_finish = SDL_GetTicksNS();
        if (count == 0 || count == (UINT)-1) {
            if (!data->rawinput || (count == (UINT)-1 && GetLastError() == ERROR_INSUFFICIENT_BUFFER)) {
                const UINT RAWINPUT_BUFFER_SIZE_INCREMENT = 96;   // 2 64-bit raw mouse packets
                BYTE *rawinput = (BYTE *)SDL_realloc(data->rawinput, data->rawinput_size + RAWINPUT_BUFFER_SIZE_INCREMENT);
                if (!rawinput) {
                    break;
                }
                input = (RAWINPUT *)(rawinput + ((BYTE *)input - data->rawinput));
                data->rawinput = rawinput;
                data->rawinput_size += RAWINPUT_BUFFER_SIZE_INCREMENT;
            } else {
                break;
            }
        } else {
            total += count;

            // Advance input to the end of the buffer
            while (count--) {
                input = NEXTRAWINPUTBLOCK(input);
            }
        }
    }

    if (total > 0) {
        Uint64 delta = poll_finish - poll_start;
        UINT mouse_total = 0;
        for (i = 0, input = (RAWINPUT *)data->rawinput; i < total; ++i, input = NEXTRAWINPUTBLOCK(input)) {
            if (input->header.dwType == RIM_TYPEMOUSE) {
                mouse_total += 1;
            }
        }
        int mouse_index = 0;
        for (i = 0, input = (RAWINPUT *)data->rawinput; i < total; ++i, input = NEXTRAWINPUTBLOCK(input)) {
            if (input->header.dwType == RIM_TYPEMOUSE) {
                mouse_index += 1; // increment first so that it starts at one
                RAWMOUSE *rawmouse = (RAWMOUSE *)((BYTE *)input + data->rawinput_offset);
                Uint64 time = poll_finish - (delta * (mouse_total - mouse_index)) / mouse_total;
                WIN_HandleRawMouseInput(time, data, input->header.hDevice, rawmouse);
            } else if (input->header.dwType == RIM_TYPEKEYBOARD) {
                RAWKEYBOARD *rawkeyboard = (RAWKEYBOARD *)((BYTE *)input + data->rawinput_offset);
                WIN_HandleRawKeyboardInput(poll_finish, data, input->header.hDevice, rawkeyboard);
            }
        }
    }
    data->last_rawinput_poll = poll_finish;
}

#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

static void AddDeviceID(Uint32 deviceID, Uint32 **list, int *count)
{
    int new_count = (*count + 1);
    Uint32 *new_list = (Uint32 *)SDL_realloc(*list, new_count * sizeof(*new_list));
    if (!new_list) {
        // Oh well, we'll drop this one
        return;
    }
    new_list[new_count - 1] = deviceID;

    *count = new_count;
    *list = new_list;
}

static bool HasDeviceID(Uint32 deviceID, const Uint32 *list, int count)
{
    for (int i = 0; i < count; ++i) {
        if (deviceID == list[i]) {
            return true;
        }
    }
    return false;
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static char *GetDeviceName(HANDLE hDevice, HDEVINFO devinfo, const char *instance, const char *default_name, bool hid_loaded)
{
    char *vendor_name = NULL;
    char *product_name = NULL;
    char *name = NULL;

    // These are 126 for USB, but can be longer for Bluetooth devices
    WCHAR vend[256], prod[256];
    vend[0] = 0;
    prod[0] = 0;


    HIDD_ATTRIBUTES attr;
    attr.VendorID = 0;
    attr.ProductID = 0;
    attr.Size = sizeof(attr);

    if (hid_loaded) {
        char devName[MAX_PATH + 1];
        UINT cap = sizeof(devName) - 1;
        UINT len = GetRawInputDeviceInfoA(hDevice, RIDI_DEVICENAME, devName, &cap);
        if (len != (UINT)-1) {
            devName[len] = '\0';

            // important: for devices with exclusive access mode as per
            // https://learn.microsoft.com/en-us/windows-hardware/drivers/hid/top-level-collections-opened-by-windows-for-system-use
            // they can only be opened with a desired access of none instead of generic read.
            HANDLE hFile = CreateFileA(devName, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
            if (hFile != INVALID_HANDLE_VALUE) {
                SDL_HidD_GetAttributes(hFile, &attr);
                SDL_HidD_GetManufacturerString(hFile, vend, sizeof(vend));
                SDL_HidD_GetProductString(hFile, prod, sizeof(prod));
                CloseHandle(hFile);
            }
        }
    }

    if (vend[0]) {
        vendor_name = WIN_StringToUTF8W(vend);
    }

    if (prod[0]) {
        product_name = WIN_StringToUTF8W(prod);
    } else {
        SP_DEVINFO_DATA data;
        SDL_zero(data);
        data.cbSize = sizeof(data);
        for (DWORD i = 0;; ++i) {
            if (!SetupDiEnumDeviceInfo(devinfo, i, &data)) {
                if (GetLastError() == ERROR_NO_MORE_ITEMS) {
                    break;
                } else {
                    continue;
                }
            }

            char DeviceInstanceId[64];
            if (!SetupDiGetDeviceInstanceIdA(devinfo, &data, DeviceInstanceId, sizeof(DeviceInstanceId), NULL))
                continue;

            if (SDL_strcasecmp(instance, DeviceInstanceId) == 0) {
                DWORD size = 0;
                if (SetupDiGetDeviceRegistryPropertyW(devinfo, &data, SPDRP_DEVICEDESC, NULL, (PBYTE)prod, sizeof(prod), &size)) {
                    // Make sure the device description is null terminated
                    size /= sizeof(*prod);
                    if (size >= SDL_arraysize(prod)) {
                        // Truncated description...
                        size = (SDL_arraysize(prod) - 1);
                    }
                    prod[size] = 0;

                    if (attr.VendorID || attr.ProductID) {
                        SDL_asprintf(&product_name, "%S (0x%.4x/0x%.4x)", prod, attr.VendorID, attr.ProductID);
                    } else {
                        product_name = WIN_StringToUTF8W(prod);
                    }
                }
                break;
            }
        }
    }

    if (!product_name && (attr.VendorID || attr.ProductID)) {
        SDL_asprintf(&product_name, "%s (0x%.4x/0x%.4x)", default_name, attr.VendorID, attr.ProductID);
    }
    name = SDL_CreateDeviceName(attr.VendorID, attr.ProductID, vendor_name, product_name, default_name);
    SDL_free(vendor_name);
    SDL_free(product_name);

    return name;
}

void WIN_CheckKeyboardAndMouseHotplug(SDL_VideoDevice *_this, bool initial_check)
{
    PRAWINPUTDEVICELIST raw_devices = NULL;
    UINT raw_device_count = 0;
    int old_keyboard_count = 0;
    SDL_KeyboardID *old_keyboards = NULL;
    int new_keyboard_count = 0;
    SDL_KeyboardID *new_keyboards = NULL;
    int old_mouse_count = 0;
    SDL_MouseID *old_mice = NULL;
    int new_mouse_count = 0;
    SDL_MouseID *new_mice = NULL;
    bool send_event = !initial_check;

    // Check to see if anything has changed
    static Uint64 s_last_device_change;
    Uint64 last_device_change = WIN_GetLastDeviceNotification();
    if (!initial_check && last_device_change == s_last_device_change) {
        return;
    }
    s_last_device_change = last_device_change;

    if ((GetRawInputDeviceList(NULL, &raw_device_count, sizeof(RAWINPUTDEVICELIST)) == -1) || (!raw_device_count)) {
        return; // oh well.
    }

    raw_devices = (PRAWINPUTDEVICELIST)SDL_malloc(sizeof(RAWINPUTDEVICELIST) * raw_device_count);
    if (!raw_devices) {
        return; // oh well.
    }

    raw_device_count = GetRawInputDeviceList(raw_devices, &raw_device_count, sizeof(RAWINPUTDEVICELIST));
    if (raw_device_count == (UINT)-1) {
        SDL_free(raw_devices);
        raw_devices = NULL;
        return; // oh well.
    }

    HDEVINFO devinfo = SetupDiGetClassDevsA(NULL, NULL, NULL, (DIGCF_ALLCLASSES | DIGCF_PRESENT));

    old_keyboards = SDL_GetKeyboards(&old_keyboard_count);
    old_mice = SDL_GetMice(&old_mouse_count);

    bool hid_loaded = WIN_LoadHIDDLL();
    for (UINT i = 0; i < raw_device_count; i++) {
        RID_DEVICE_INFO rdi;
        char devName[MAX_PATH] = { 0 };
        UINT rdiSize = sizeof(rdi);
        UINT nameSize = SDL_arraysize(devName);
        int vendor = 0, product = 0;
        DWORD dwType = raw_devices[i].dwType;
        char *instance, *ptr, *name;

        if (dwType != RIM_TYPEKEYBOARD && dwType != RIM_TYPEMOUSE) {
            continue;
        }

        rdi.cbSize = sizeof(rdi);
        if (GetRawInputDeviceInfoA(raw_devices[i].hDevice, RIDI_DEVICEINFO, &rdi, &rdiSize) == ((UINT)-1) ||
            GetRawInputDeviceInfoA(raw_devices[i].hDevice, RIDI_DEVICENAME, devName, &nameSize) == ((UINT)-1)) {
            continue;
        }

        // Extract the device instance
        instance = devName;
        while (*instance == '\\' || *instance == '?') {
            ++instance;
        }
        for (ptr = instance; *ptr; ++ptr) {
            if (*ptr == '#') {
                *ptr = '\\';
            }
            if (*ptr == '{') {
                if (ptr > instance && ptr[-1] == '\\') {
                    --ptr;
                }
                break;
            }
        }
        *ptr = '\0';

        SDL_sscanf(instance, "HID\\VID_%X&PID_%X&", &vendor, &product);

        switch (dwType) {
        case RIM_TYPEKEYBOARD:
            if (SDL_IsKeyboard((Uint16)vendor, (Uint16)product, rdi.keyboard.dwNumberOfKeysTotal)) {
                SDL_KeyboardID keyboardID = (Uint32)(uintptr_t)raw_devices[i].hDevice;
                AddDeviceID(keyboardID, &new_keyboards, &new_keyboard_count);
                if (!HasDeviceID(keyboardID, old_keyboards, old_keyboard_count)) {
                    name = GetDeviceName(raw_devices[i].hDevice, devinfo, instance, "Keyboard", hid_loaded);
                    SDL_AddKeyboard(keyboardID, name, send_event);
                    SDL_free(name);
                }
            }
            break;
        case RIM_TYPEMOUSE:
            if (SDL_IsMouse((Uint16)vendor, (Uint16)product)) {
                SDL_MouseID mouseID = (Uint32)(uintptr_t)raw_devices[i].hDevice;
                AddDeviceID(mouseID, &new_mice, &new_mouse_count);
                if (!HasDeviceID(mouseID, old_mice, old_mouse_count)) {
                    name = GetDeviceName(raw_devices[i].hDevice, devinfo, instance, "Mouse", hid_loaded);
                    SDL_AddMouse(mouseID, name, send_event);
                    SDL_free(name);
                }
            }
            break;
        default:
            break;
        }
    }
    if (hid_loaded) {
        WIN_UnloadHIDDLL();
    }

    for (int i = old_keyboard_count; i--;) {
        if (!HasDeviceID(old_keyboards[i], new_keyboards, new_keyboard_count)) {
            SDL_RemoveKeyboard(old_keyboards[i], send_event);
        }
    }

    for (int i = old_mouse_count; i--;) {
        if (!HasDeviceID(old_mice[i], new_mice, new_mouse_count)) {
            SDL_RemoveMouse(old_mice[i], send_event);
        }
    }

    SDL_free(old_keyboards);
    SDL_free(new_keyboards);
    SDL_free(old_mice);
    SDL_free(new_mice);

    SetupDiDestroyDeviceInfoList(devinfo);

    SDL_free(raw_devices);
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

// Return true if spurious LCtrl is pressed
// LCtrl is sent when RAltGR is pressed
static bool SkipAltGrLeftControl(WPARAM wParam, LPARAM lParam)
{
    if (wParam != VK_CONTROL) {
        return false;
    }

    // Is this an extended key (i.e. right key)?
    if (lParam & 0x01000000) {
        return false;
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    // Here is a trick: "Alt Gr" sends LCTRL, then RALT. We only
    // want the RALT message, so we try to see if the next message
    // is a RALT message. In that case, this is a false LCTRL!
    MSG next_msg;
    DWORD msg_time = GetMessageTime();
    if (PeekMessage(&next_msg, NULL, 0, 0, PM_NOREMOVE)) {
        if (next_msg.message == WM_KEYDOWN ||
            next_msg.message == WM_SYSKEYDOWN) {
            if (next_msg.wParam == VK_MENU && (next_msg.lParam & 0x01000000) && next_msg.time == msg_time) {
                // Next message is a RALT down message, which means that this is NOT a proper LCTRL message!
                return true;
            }
        }
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    return false;
}

static bool DispatchModalLoopMessageHook(HWND *hwnd, UINT *msg, WPARAM *wParam, LPARAM *lParam)
{
    MSG dummy;

    SDL_zero(dummy);
    dummy.hwnd = *hwnd;
    dummy.message = *msg;
    dummy.wParam = *wParam;
    dummy.lParam = *lParam;
    if (g_WindowsMessageHook(g_WindowsMessageHookData, &dummy)) {
        // Can't modify the hwnd, but everything else is fair game
        *msg = dummy.message;
        *wParam = dummy.wParam;
        *lParam = dummy.lParam;
        return true;
    }
    return false;
}

LRESULT CALLBACK WIN_WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    SDL_WindowData *data;
    LRESULT returnCode = -1;

    // Get the window data for the window
    data = WIN_GetWindowDataFromHWND(hwnd);
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (!data) {
        // Fallback
        data = (SDL_WindowData *)GetProp(hwnd, TEXT("SDL_WindowData"));
    }
#endif
    if (!data) {
        return CallWindowProc(DefWindowProc, hwnd, msg, wParam, lParam);
    }

#ifdef WMMSG_DEBUG
    {
        char message[1024];
        if (msg > MAX_WMMSG) {
            SDL_snprintf(message, sizeof(message), "Received windows message: %p UNKNOWN (%d) -- 0x%x, 0x%x\r\n", hwnd, msg, wParam, lParam);
        } else {
            SDL_snprintf(message, sizeof(message), "Received windows message: %p %s -- 0x%x, 0x%x\r\n", hwnd, wmtab[msg], wParam, lParam);
        }
        OutputDebugStringA(message);
    }
#endif // WMMSG_DEBUG


    if (g_WindowsMessageHook && data->in_modal_loop) {
        // Synthesize a message for window hooks so they can modify the message if desired
        if (!DispatchModalLoopMessageHook(&hwnd, &msg, &wParam, &lParam)) {
            return 0;
        }
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (WIN_HandleIMEMessage(hwnd, msg, wParam, &lParam, data->videodata)) {
        return 0;
    }
#endif

    switch (msg) {

    case WM_SHOWWINDOW:
    {
        if (wParam) {
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_SHOWN, 0, 0);
        } else {
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
        }
    } break;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    case WM_NCACTIVATE:
    {
        // Don't immediately clip the cursor in case we're clicking minimize/maximize buttons
        data->postpone_clipcursor = true;
        data->clipcursor_queued = true;

        /* Update the focus here, since it's possible to get WM_ACTIVATE and WM_SETFOCUS without
           actually being the foreground window, but this appears to get called in all cases where
           the global foreground window changes to and from this window. */
        WIN_UpdateFocus(data->window, !!wParam, GetMessagePos());
    } break;

    case WM_ACTIVATE:
    {
        // Update the focus in case we changed focus to a child window and then away from the application
        WIN_UpdateFocus(data->window, !!LOWORD(wParam), GetMessagePos());
    } break;

    case WM_MOUSEACTIVATE:
    {
        if (SDL_WINDOW_IS_POPUP(data->window)) {
            return MA_NOACTIVATE;
        }

        // Check parents to see if they are in relative mouse mode and focused
        SDL_Window *parent = data->window->parent;
        while (parent) {
            if ((parent->flags & SDL_WINDOW_INPUT_FOCUS) &&
                (parent->flags & SDL_WINDOW_MOUSE_RELATIVE_MODE)) {
                return MA_NOACTIVATE;
            }
            parent = parent->parent;
        }
    } break;

    case WM_SETFOCUS:
    {
        // Update the focus in case it's changing between top-level windows in the same application
        WIN_UpdateFocus(data->window, true, GetMessagePos());
    } break;

    case WM_KILLFOCUS:
    case WM_ENTERIDLE:
    {
        // Update the focus in case it's changing between top-level windows in the same application
        WIN_UpdateFocus(data->window, false, GetMessagePos());
    } break;

    case WM_POINTERENTER:
    {
        if (!data->videodata->GetPointerType) {
            break;  // Not on Windows8 or later? We shouldn't get this event, but just in case...
        }

        const UINT32 pointerid = GET_POINTERID_WPARAM(wParam);
        void *hpointer = (void *) (size_t) pointerid;
        POINTER_INPUT_TYPE pointer_type = PT_POINTER;
        if (!data->videodata->GetPointerType(pointerid, &pointer_type)) {
            break;  // oh well.
        } else if (pointer_type != PT_PEN) {
            break;  // we only care about pens here.
        } else if (SDL_FindPenByHandle(hpointer)) {
            break;  // we already have this one, don't readd it.
        }

        // one can use GetPointerPenInfo() to get the current state of the pen, and check POINTER_PEN_INFO::penMask,
        //  but the docs aren't clear if these masks are _always_ set for pens with specific features, or if they
        //  could be unset at this moment because Windows is still deciding what capabilities the pen has, and/or
        //  doesn't yet have valid data for them. As such, just say everything that the interface supports is
        //  available...we don't expose this information through the public API at the moment anyhow.
        SDL_PenInfo info;
        SDL_zero(info);
        info.capabilities = SDL_PEN_CAPABILITY_PRESSURE | SDL_PEN_CAPABILITY_XTILT | SDL_PEN_CAPABILITY_YTILT | SDL_PEN_CAPABILITY_DISTANCE | SDL_PEN_CAPABILITY_ROTATION | SDL_PEN_CAPABILITY_ERASER;
        info.max_tilt = 90.0f;
        info.num_buttons = 1;
        info.subtype = SDL_PEN_TYPE_PENCIL;
        SDL_AddPenDevice(0, NULL, &info, hpointer);
        returnCode = 0;
    } break;

    case WM_POINTERCAPTURECHANGED:
    case WM_POINTERLEAVE:
    {
        const UINT32 pointerid = GET_POINTERID_WPARAM(wParam);
        void *hpointer = (void *) (size_t) pointerid;
        const SDL_PenID pen = SDL_FindPenByHandle(hpointer);
        if (pen == 0) {
            break;  // not a pen, or not a pen we already knew about.
        }

        // if this just left the _window_, we don't care. If this is no longer visible to the tablet, time to remove it!
        if ((msg == WM_POINTERCAPTURECHANGED) || !IS_POINTER_INCONTACT_WPARAM(wParam)) {
            SDL_RemovePenDevice(WIN_GetEventTimestamp(), pen);
        }
        returnCode = 0;
    } break;

    case WM_POINTERUPDATE: {
        POINTER_INPUT_TYPE pointer_type = PT_POINTER;
        if (!data->videodata->GetPointerType || !data->videodata->GetPointerType(GET_POINTERID_WPARAM(wParam), &pointer_type)) {
            break;  // oh well.
        }

        if (pointer_type == PT_MOUSE) {
            data->last_pointer_update = lParam;
            returnCode = 0;
            break;
        }
    }
    SDL_FALLTHROUGH;

    case WM_POINTERDOWN:
    case WM_POINTERUP: {
        POINTER_PEN_INFO pen_info;
        const UINT32 pointerid = GET_POINTERID_WPARAM(wParam);
        void *hpointer = (void *) (size_t) pointerid;
        const SDL_PenID pen = SDL_FindPenByHandle(hpointer);
        if (pen == 0) {
            break;  // not a pen, or not a pen we already knew about.
        } else if (!data->videodata->GetPointerPenInfo || !data->videodata->GetPointerPenInfo(pointerid, &pen_info)) {
            break;  // oh well.
        }

        const Uint64 timestamp = WIN_GetEventTimestamp();
        SDL_Window *window = data->window;

        const bool istouching = IS_POINTER_INCONTACT_WPARAM(wParam) && IS_POINTER_FIRSTBUTTON_WPARAM(wParam);

        // if lifting off, do it first, so any motion changes don't cause app issues.
        if (!istouching) {
            SDL_SendPenTouch(timestamp, pen, window, (pen_info.penFlags & PEN_FLAG_INVERTED) != 0, false);
        }

        POINT position;
        position.x = (LONG) GET_X_LPARAM(lParam);
        position.y = (LONG) GET_Y_LPARAM(lParam);
        ScreenToClient(data->hwnd, &position);

        SDL_SendPenMotion(timestamp, pen, window, (float) position.x, (float) position.y);
        SDL_SendPenButton(timestamp, pen, window, 1, (pen_info.penFlags & PEN_FLAG_BARREL) != 0);
        SDL_SendPenButton(timestamp, pen, window, 2, (pen_info.penFlags & PEN_FLAG_ERASER) != 0);

        if (pen_info.penMask & PEN_MASK_PRESSURE) {
            SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_PRESSURE, ((float) pen_info.pressure) / 1024.0f);  // pen_info.pressure is in the range 0..1024.
        }

        if (pen_info.penMask & PEN_MASK_ROTATION) {
            SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_ROTATION, ((float) pen_info.rotation));  // it's already in the range of 0 to 359.
        }

        if (pen_info.penMask & PEN_MASK_TILT_X) {
            SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_XTILT, ((float) pen_info.tiltX));  // it's already in the range of -90 to 90..
        }

        if (pen_info.penMask & PEN_MASK_TILT_Y) {
            SDL_SendPenAxis(timestamp, pen, window, SDL_PEN_AXIS_YTILT, ((float) pen_info.tiltY));  // it's already in the range of -90 to 90..
        }

        // if setting down, do it last, so the pen is positioned correctly from the first contact.
        if (istouching) {
            SDL_SendPenTouch(timestamp, pen, window, (pen_info.penFlags & PEN_FLAG_INVERTED) != 0, true);
        }

        returnCode = 0;
    } break;

    case WM_MOUSEMOVE:
    {
        SDL_Window *window = data->window;

        if (window->flags & SDL_WINDOW_INPUT_FOCUS) {
            bool wish_clip_cursor = (
                window->flags & (SDL_WINDOW_MOUSE_RELATIVE_MODE | SDL_WINDOW_MOUSE_GRABBED) ||
                (window->mouse_rect.w > 0 && window->mouse_rect.h > 0)
            );
            if (wish_clip_cursor) { // queue clipcursor refresh on pump finish
                data->clipcursor_queued = true;
            }
        }

        if (!data->mouse_tracked) {
            TRACKMOUSEEVENT trackMouseEvent;

            trackMouseEvent.cbSize = sizeof(TRACKMOUSEEVENT);
            trackMouseEvent.dwFlags = TME_LEAVE;
            trackMouseEvent.hwndTrack = data->hwnd;

            if (TrackMouseEvent(&trackMouseEvent)) {
                data->mouse_tracked = true;
            }

            WIN_CheckAsyncMouseRelease(WIN_GetEventTimestamp(), data);
        }

        if (!data->videodata->raw_mouse_enabled) {
            // Only generate mouse events for real mouse
            if (GetMouseMessageSource((ULONG)GetMessageExtraInfo()) == SDL_MOUSE_EVENT_SOURCE_MOUSE &&
                lParam != data->last_pointer_update) {
                SDL_SendMouseMotion(WIN_GetEventTimestamp(), window, SDL_GLOBAL_MOUSE_ID, false, (float)GET_X_LPARAM(lParam), (float)GET_Y_LPARAM(lParam));
            }
        }
        
        return 0;
        
    } break;

    case WM_LBUTTONUP:
    case WM_RBUTTONUP:
    case WM_MBUTTONUP:
    case WM_XBUTTONUP:
    case WM_LBUTTONDOWN:
    case WM_LBUTTONDBLCLK:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONDBLCLK:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONDBLCLK:
    case WM_XBUTTONDOWN:
    case WM_XBUTTONDBLCLK:
    {
        /* SDL_Mouse *mouse = SDL_GetMouse(); */
        if (!data->videodata->raw_mouse_enabled) {
            if (GetMouseMessageSource((ULONG)GetMessageExtraInfo()) == SDL_MOUSE_EVENT_SOURCE_MOUSE &&
                lParam != data->last_pointer_update) {
                WIN_CheckWParamMouseButtons(WIN_GetEventTimestamp(), wParam, data, SDL_GLOBAL_MOUSE_ID);
            }
        }
    } break;

#if 0   // We handle raw input all at once instead of using a syscall for each mouse event
    case WM_INPUT:
    {
        HRAWINPUT hRawInput = (HRAWINPUT)lParam;
        RAWINPUT inp;
        UINT size = sizeof(inp);

        // Relative mouse motion is delivered to the window with keyboard focus
        if (data->window != SDL_GetKeyboardFocus()) {
            break;
        }

        GetRawInputData(hRawInput, RID_INPUT, &inp, &size, sizeof(RAWINPUTHEADER));
        if (inp.header.dwType == RIM_TYPEMOUSE) {
            WIN_HandleRawMouseInput(WIN_GetEventTimestamp(), data, inp.header.hDevice, &inp.data.mouse);
        } else if (inp.header.dwType == RIM_TYPEKEYBOARD) {
            WIN_HandleRawKeyboardInput(WIN_GetEventTimestamp(), data, inp.header.hDevice, &inp.data.keyboard);
        }
    } break;
#endif

    case WM_MOUSEWHEEL:
    case WM_MOUSEHWHEEL:
    {
        if (!data->videodata->raw_mouse_enabled) {
            short amount = GET_WHEEL_DELTA_WPARAM(wParam);
            float fAmount = (float)amount / WHEEL_DELTA;
            if (msg == WM_MOUSEWHEEL) {
                SDL_SendMouseWheel(WIN_GetEventTimestamp(), data->window, SDL_GLOBAL_MOUSE_ID, 0.0f, fAmount, SDL_MOUSEWHEEL_NORMAL);
            } else {
                SDL_SendMouseWheel(WIN_GetEventTimestamp(), data->window, SDL_GLOBAL_MOUSE_ID, fAmount, 0.0f, SDL_MOUSEWHEEL_NORMAL);
            }
        }
    } break;

    case WM_MOUSELEAVE:
        if (!(data->window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
            if (SDL_GetMouseFocus() == data->window && !SDL_GetMouse()->relative_mode && !IsIconic(hwnd)) {
                SDL_Mouse *mouse;
                DWORD pos = GetMessagePos();
                POINT cursorPos;
                cursorPos.x = GET_X_LPARAM(pos);
                cursorPos.y = GET_Y_LPARAM(pos);
                ScreenToClient(hwnd, &cursorPos);
                mouse = SDL_GetMouse();
                if (!mouse->was_touch_mouse_events) { // we're not a touch handler causing a mouse leave?
                    SDL_SendMouseMotion(WIN_GetEventTimestamp(), data->window, SDL_GLOBAL_MOUSE_ID, false, (float)cursorPos.x, (float)cursorPos.y);
                } else {                                       // touch handling?
                    mouse->was_touch_mouse_events = false; // not anymore
                    if (mouse->touch_mouse_events) {           // convert touch to mouse events
                        SDL_SendMouseMotion(WIN_GetEventTimestamp(), data->window, SDL_TOUCH_MOUSEID, false, (float)cursorPos.x, (float)cursorPos.y);
                    } else { // normal handling
                        SDL_SendMouseMotion(WIN_GetEventTimestamp(), data->window, SDL_GLOBAL_MOUSE_ID, false, (float)cursorPos.x, (float)cursorPos.y);
                    }
                }
            }

            if (!SDL_GetMouse()->relative_mode) {
                // When WM_MOUSELEAVE is fired we can be assured that the cursor has left the window
                SDL_SetMouseFocus(NULL);
            }
        }

        // Once we get WM_MOUSELEAVE we're guaranteed that the window is no longer tracked
        data->mouse_tracked = false;

        returnCode = 0;
        break;
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    case WM_KEYDOWN:
    case WM_SYSKEYDOWN:
    {
        if (SkipAltGrLeftControl(wParam, lParam)) {
            returnCode = 0;
            break;
        }

        bool virtual_key = false;
        Uint16 rawcode = 0;
        SDL_Scancode code = WindowsScanCodeToSDLScanCode(lParam, wParam, &rawcode, &virtual_key);

        // Detect relevant keyboard shortcuts
        if (code == SDL_SCANCODE_F4 && (SDL_GetModState() & SDL_KMOD_ALT)) {
            // ALT+F4: Close window
            if (ShouldGenerateWindowCloseOnAltF4()) {
                SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_CLOSE_REQUESTED, 0, 0);
            }
        }

        if (virtual_key || !data->videodata->raw_keyboard_enabled || data->window->text_input_active) {
            SDL_SendKeyboardKey(WIN_GetEventTimestamp(), SDL_GLOBAL_KEYBOARD_ID, rawcode, code, true);
        }
    }

        returnCode = 0;
        break;

    case WM_SYSKEYUP:
    case WM_KEYUP:
    {
        if (SkipAltGrLeftControl(wParam, lParam)) {
            returnCode = 0;
            break;
        }

        bool virtual_key = false;
        Uint16 rawcode = 0;
        SDL_Scancode code = WindowsScanCodeToSDLScanCode(lParam, wParam, &rawcode, &virtual_key);
        const bool *keyboardState = SDL_GetKeyboardState(NULL);

        if (virtual_key || !data->videodata->raw_keyboard_enabled || data->window->text_input_active) {
            if (code == SDL_SCANCODE_PRINTSCREEN && !keyboardState[code]) {
                SDL_SendKeyboardKey(WIN_GetEventTimestamp(), SDL_GLOBAL_KEYBOARD_ID, rawcode, code, true);
            }
            SDL_SendKeyboardKey(WIN_GetEventTimestamp(), SDL_GLOBAL_KEYBOARD_ID, rawcode, code, false);
        }
    }
        returnCode = 0;
        break;

    case WM_UNICHAR:
        if (wParam == UNICODE_NOCHAR) {
            returnCode = 1;
        } else {
            if (SDL_TextInputActive(data->window)) {
                char text[5];
                char *end = SDL_UCS4ToUTF8((Uint32)wParam, text);
                *end = '\0';
                SDL_SendKeyboardText(text);
            }
            returnCode = 0;
        }
        break;

    case WM_CHAR:
        if (SDL_TextInputActive(data->window)) {
            /* Characters outside Unicode Basic Multilingual Plane (BMP)
             * are coded as so called "surrogate pair" in two separate UTF-16 character events.
             * Cache high surrogate until next character event. */
            if (IS_HIGH_SURROGATE(wParam)) {
                data->high_surrogate = (WCHAR)wParam;
            } else {
                WCHAR utf16[3];

                utf16[0] = data->high_surrogate ? data->high_surrogate : (WCHAR)wParam;
                utf16[1] = data->high_surrogate ? (WCHAR)wParam : L'\0';
                utf16[2] = L'\0';

                char utf8[5];
                int result = WIN_WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, utf16, -1, utf8, sizeof(utf8), NULL, NULL);
                if (result > 0) {
                    SDL_SendKeyboardText(utf8);
                }
                data->high_surrogate = L'\0';
            }
        } else {
            data->high_surrogate = L'\0';
        }

        returnCode = 0;
        break;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
#ifdef WM_INPUTLANGCHANGE
    case WM_INPUTLANGCHANGE:
    {
        WIN_UpdateKeymap(true);
    }
        returnCode = 1;
        break;
#endif // WM_INPUTLANGCHANGE

    case WM_NCLBUTTONDOWN:
    {
        data->in_title_click = true;

        // Fix for 500ms hang after user clicks on the title bar, but before moving mouse
        // Reference: https://gamedev.net/forums/topic/672094-keeping-things-moving-during-win32-moveresize-events/5254386/
        if (SendMessage(hwnd, WM_NCHITTEST, wParam, lParam) == HTCAPTION) {
            POINT cursorPos;
            GetCursorPos(&cursorPos); // want the most current pos so as to not cause position change
            ScreenToClient(hwnd, &cursorPos);
            PostMessage(hwnd, WM_MOUSEMOVE, 0, cursorPos.x | (((Uint32)((Sint16)cursorPos.y)) << 16));
        }
    } break;

    case WM_CAPTURECHANGED:
    {
        data->in_title_click = false;

        // The mouse may have been released during a modal loop
        WIN_CheckAsyncMouseRelease(WIN_GetEventTimestamp(), data);
    } break;

#ifdef WM_GETMINMAXINFO
    case WM_GETMINMAXINFO:
    {
        MINMAXINFO *info;
        RECT size;
        int x, y;
        int w, h;
        int min_w, min_h;
        int max_w, max_h;
        BOOL constrain_max_size;

        // If this is an expected size change, allow it
        if (data->expected_resize) {
            break;
        }

        // Get the current position of our window
        GetWindowRect(hwnd, &size);
        x = size.left;
        y = size.top;

        // Calculate current size of our window
        SDL_GetWindowSize(data->window, &w, &h);
        SDL_GetWindowMinimumSize(data->window, &min_w, &min_h);
        SDL_GetWindowMaximumSize(data->window, &max_w, &max_h);

        /* Store in min_w and min_h difference between current size and minimal
           size so we don't need to call AdjustWindowRectEx twice */
        min_w -= w;
        min_h -= h;
        if (max_w && max_h) {
            max_w -= w;
            max_h -= h;
            constrain_max_size = TRUE;
        } else {
            constrain_max_size = FALSE;
        }

        if (!(SDL_GetWindowFlags(data->window) & SDL_WINDOW_BORDERLESS) && !SDL_WINDOW_IS_POPUP(data->window)) {
            size.top = 0;
            size.left = 0;
            size.bottom = h;
            size.right = w;
            WIN_AdjustWindowRectForHWND(hwnd, &size, 0);
            w = size.right - size.left;
            h = size.bottom - size.top;
#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_GETMINMAXINFO: max window size: %dx%d using dpi: %u", w, h, dpi);
#endif
        }

        // Fix our size to the current size
        info = (MINMAXINFO *)lParam;
        if (SDL_GetWindowFlags(data->window) & SDL_WINDOW_RESIZABLE) {
            if (SDL_GetWindowFlags(data->window) & SDL_WINDOW_BORDERLESS) {
                int screenW = GetSystemMetrics(SM_CXSCREEN);
                int screenH = GetSystemMetrics(SM_CYSCREEN);
                info->ptMaxSize.x = SDL_max(w, screenW);
                info->ptMaxSize.y = SDL_max(h, screenH);
                info->ptMaxPosition.x = SDL_min(0, ((screenW - w) / 2));
                info->ptMaxPosition.y = SDL_min(0, ((screenH - h) / 2));
            }
            info->ptMinTrackSize.x = (LONG)w + min_w;
            info->ptMinTrackSize.y = (LONG)h + min_h;
            if (constrain_max_size) {
                info->ptMaxTrackSize.x = (LONG)w + max_w;
                info->ptMaxTrackSize.y = (LONG)h + max_h;
            }
        } else {
            info->ptMaxSize.x = w;
            info->ptMaxSize.y = h;
            info->ptMaxPosition.x = x;
            info->ptMaxPosition.y = y;
            info->ptMinTrackSize.x = w;
            info->ptMinTrackSize.y = h;
            info->ptMaxTrackSize.x = w;
            info->ptMaxTrackSize.y = h;
        }
    }
        returnCode = 0;
        break;
#endif // WM_GETMINMAXINFO

    case WM_WINDOWPOSCHANGING:

        if (data->expected_resize) {
            returnCode = 0;
        }
        break;

    case WM_WINDOWPOSCHANGED:
    {
        SDL_Window *win;
        const SDL_DisplayID original_displayID = data->last_displayID;
        const WINDOWPOS *windowpos = (WINDOWPOS *)lParam;
        bool iconic;
        bool zoomed;
        RECT rect;
        int x, y;
        int w, h;

        if (windowpos->flags & SWP_SHOWWINDOW) {
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_SHOWN, 0, 0);
        }

        // These must be set after sending SDL_EVENT_WINDOW_SHOWN as that may apply pending
        // window operations that change the window state.
        iconic = IsIconic(hwnd);
        zoomed = IsZoomed(hwnd);

        if (iconic) {
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MINIMIZED, 0, 0);
        } else if (zoomed) {
            if (data->window->flags & SDL_WINDOW_MINIMIZED) {
                // If going from minimized to maximized, send the restored event first.
                SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
            }
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MAXIMIZED, 0, 0);
            data->force_ws_maximizebox = true;
        } else if (data->window->flags & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED)) {
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESTORED, 0, 0);

            /* If resizable was forced on for the maximized window, clear the style flags now,
             * but not if the window is fullscreen, as this needs to be preserved in that case.
             */
            if (!(data->window->flags & SDL_WINDOW_FULLSCREEN)) {
                data->force_ws_maximizebox = false;
                WIN_SetWindowResizable(SDL_GetVideoDevice(), data->window, !!(data->window->flags & SDL_WINDOW_RESIZABLE));
            }
        }

        if (windowpos->flags & SWP_HIDEWINDOW) {
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
        }

        // When the window is minimized it's resized to the dock icon size, ignore this
        if (iconic) {
            break;
        }

        if (data->initializing) {
            break;
        }

        if (!data->disable_move_size_events) {
            if (GetClientRect(hwnd, &rect) && WIN_WindowRectValid(&rect)) {
                ClientToScreen(hwnd, (LPPOINT) &rect);
                ClientToScreen(hwnd, (LPPOINT) &rect + 1);

                x = rect.left;
                y = rect.top;

                SDL_GlobalToRelativeForWindow(data->window, x, y, &x, &y);
                SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MOVED, x, y);
            }

            // Moving the window from one display to another can change the size of the window (in the handling of SDL_EVENT_WINDOW_MOVED), so we need to re-query the bounds
            if (GetClientRect(hwnd, &rect) && WIN_WindowRectValid(&rect)) {
                w = rect.right;
                h = rect.bottom;

                SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESIZED, w, h);
            }
        }

        WIN_UpdateClipCursor(data->window);

        // Update the window display position
        data->last_displayID = SDL_GetDisplayForWindow(data->window);

        if (data->last_displayID != original_displayID) {
            // Display changed, check ICC profile
            WIN_UpdateWindowICCProfile(data->window, true);
        }

        // Update the position of any child windows
        for (win = data->window->first_child; win; win = win->next_sibling) {
            // Don't update hidden child popup windows, their relative position doesn't change
            if (SDL_WINDOW_IS_POPUP(win) && !(win->flags & SDL_WINDOW_HIDDEN)) {
                WIN_SetWindowPositionInternal(win, SWP_NOCOPYBITS | SWP_NOACTIVATE, SDL_WINDOWRECT_CURRENT);
            }
        }

        // Forces a WM_PAINT event
        InvalidateRect(hwnd, NULL, FALSE);

    } break;

    case WM_ENTERSIZEMOVE:
    case WM_ENTERMENULOOP:
    {
        if (g_WindowsMessageHook) {
            if (!DispatchModalLoopMessageHook(&hwnd, &msg, &wParam, &lParam)) {
                return 0;
            }
        }

        ++data->in_modal_loop;
        if (data->in_modal_loop == 1) {
            data->initial_size_rect.left = data->window->x;
            data->initial_size_rect.right = data->window->x + data->window->w;
            data->initial_size_rect.top = data->window->y;
            data->initial_size_rect.bottom = data->window->y + data->window->h;

            SetTimer(hwnd, (UINT_PTR)SDL_IterateMainCallbacks, USER_TIMER_MINIMUM, NULL);

            // Reset the keyboard, as we won't get any key up events during the modal loop
            SDL_ResetKeyboard();
        }
    } break;

    case WM_TIMER:
    {
        if (wParam == (UINT_PTR)SDL_IterateMainCallbacks) {
            SDL_OnWindowLiveResizeUpdate(data->window);

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
#if 0 // This locks up the Windows compositor when called by Steam; disabling until we understand why
            // Make sure graphics operations are complete for smooth refresh
            if (data->videodata->DwmFlush) {
                data->videodata->DwmFlush();
            }
#endif
#endif
            return 0;
        }
    } break;

    case WM_EXITSIZEMOVE:
    case WM_EXITMENULOOP:
    {
        --data->in_modal_loop;
        if (data->in_modal_loop == 0) {
            KillTimer(hwnd, (UINT_PTR)SDL_IterateMainCallbacks);
        }
    } break;

    case WM_SIZING:
        {
            WPARAM edge = wParam;
            RECT* dragRect = (RECT*)lParam;
            RECT clientDragRect = *dragRect;
            bool lock_aspect_ratio = (data->window->max_aspect == data->window->min_aspect) ? true : false;
            RECT rc;
            LONG w, h;
            float new_aspect;

            // if aspect ratio constraints are not enabled then skip this message
            if (data->window->min_aspect <= 0 && data->window->max_aspect <= 0) {
                break;
            }

            // unadjust the dragRect from the window rect to the client rect
            SetRectEmpty(&rc);
            if (!AdjustWindowRectEx(&rc, GetWindowStyle(hwnd), GetMenu(hwnd) != NULL, GetWindowExStyle(hwnd))) {
                break;
            }

            clientDragRect.left -= rc.left;
            clientDragRect.top -= rc.top;
            clientDragRect.right -= rc.right;
            clientDragRect.bottom -= rc.bottom;

            w = clientDragRect.right - clientDragRect.left;
            h = clientDragRect.bottom - clientDragRect.top;
            new_aspect = w / (float)h;

            // handle the special case in which the min ar and max ar are the same so the window can size symmetrically
            if (lock_aspect_ratio) {
                switch (edge) {
                case WMSZ_LEFT:
                case WMSZ_RIGHT:
                    h = (int)SDL_roundf(w / data->window->max_aspect);
                    break;
                default:
                    // resizing via corners or top or bottom
                    w = (int)SDL_roundf(h * data->window->max_aspect);
                    break;
                }
            } else {
                switch (edge) {
                case WMSZ_LEFT:
                case WMSZ_RIGHT:
                    if (data->window->max_aspect > 0.0f && new_aspect > data->window->max_aspect) {
                        w = (int)SDL_roundf(h * data->window->max_aspect);
                    } else if (data->window->min_aspect > 0.0f && new_aspect < data->window->min_aspect) {
                        w = (int)SDL_roundf(h * data->window->min_aspect);
                    }
                    break;
                case WMSZ_TOP:
                case WMSZ_BOTTOM:
                    if (data->window->min_aspect > 0.0f && new_aspect < data->window->min_aspect) {
                        h = (int)SDL_roundf(w / data->window->min_aspect);
                    } else if (data->window->max_aspect > 0.0f && new_aspect > data->window->max_aspect) {
                        h = (int)SDL_roundf(w / data->window->max_aspect);
                    }
                    break;

                default:
                    // resizing via corners
                    if (data->window->max_aspect > 0.0f && new_aspect > data->window->max_aspect) {
                        w = (int)SDL_roundf(h * data->window->max_aspect);
                    } else if (data->window->min_aspect > 0.0f && new_aspect < data->window->min_aspect) {
                        h = (int)SDL_roundf(w / data->window->min_aspect);
                    }
                    break;
                }
            }

            switch (edge) {
            case WMSZ_LEFT:
                clientDragRect.left = clientDragRect.right - w;
                if (lock_aspect_ratio) {
                    clientDragRect.top = (data->initial_size_rect.bottom + data->initial_size_rect.top - h) / 2;
                }
                clientDragRect.bottom = h + clientDragRect.top;
                break;
            case WMSZ_BOTTOMLEFT:
                clientDragRect.left = clientDragRect.right - w;
                clientDragRect.bottom = h + clientDragRect.top;
                break;
            case WMSZ_RIGHT:
                clientDragRect.right = w + clientDragRect.left;
                if (lock_aspect_ratio) {
                    clientDragRect.top = (data->initial_size_rect.bottom + data->initial_size_rect.top - h) / 2;
                }
                clientDragRect.bottom = h + clientDragRect.top;
                break;
            case WMSZ_TOPRIGHT:
                clientDragRect.right = w + clientDragRect.left;
                clientDragRect.top = clientDragRect.bottom - h;
                break;
            case WMSZ_TOP:
                if (lock_aspect_ratio) {
                    clientDragRect.left = (data->initial_size_rect.right + data->initial_size_rect.left - w) / 2;
                }
                clientDragRect.right = w + clientDragRect.left;
                clientDragRect.top = clientDragRect.bottom - h;
                break;
            case WMSZ_TOPLEFT:
                clientDragRect.left = clientDragRect.right - w;
                clientDragRect.top = clientDragRect.bottom - h;
                break;
            case WMSZ_BOTTOM:
                if (lock_aspect_ratio) {
                    clientDragRect.left = (data->initial_size_rect.right + data->initial_size_rect.left - w) / 2;
                }
                clientDragRect.right = w + clientDragRect.left;
                clientDragRect.bottom = h + clientDragRect.top;
                break;
            case WMSZ_BOTTOMRIGHT:
                clientDragRect.right = w + clientDragRect.left;
                clientDragRect.bottom = h + clientDragRect.top;
                break;
            }

            // convert the client rect to a window rect
            if (!AdjustWindowRectEx(&clientDragRect, GetWindowStyle(hwnd), GetMenu(hwnd) != NULL, GetWindowExStyle(hwnd))) {
                break;
            }

            *dragRect = clientDragRect;
        }
        break;

    case WM_SETCURSOR:
    {
        Uint16 hittest;

        hittest = LOWORD(lParam);
        if (hittest == HTCLIENT) {
            SetCursor(SDL_cursor);
            returnCode = TRUE;
        } else if (!g_WindowFrameUsableWhileCursorHidden && !SDL_cursor) {
            SetCursor(NULL);
            returnCode = TRUE;
        }
    } break;

        // We were occluded, refresh our display
    case WM_PAINT:
    {
        RECT rect;
        if (GetUpdateRect(hwnd, &rect, FALSE)) {
            const LONG style = GetWindowLong(hwnd, GWL_EXSTYLE);

            /* Composited windows will continue to receive WM_PAINT messages for update
               regions until the window is actually painted through Begin/EndPaint */
            if (style & WS_EX_COMPOSITED) {
                PAINTSTRUCT ps;
                BeginPaint(hwnd, &ps);
                EndPaint(hwnd, &ps);
            }

            ValidateRect(hwnd, NULL);
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_EXPOSED, 0, 0);
        }
    }
        returnCode = 0;
        break;

        // We'll do our own drawing, prevent flicker
    case WM_ERASEBKGND:
        if (ShouldClearWindowOnEraseBackground(data)) {
            RECT client_rect;
            HBRUSH brush;
            data->videodata->cleared = true;
            GetClientRect(hwnd, &client_rect);
            brush = CreateSolidBrush(0);
            FillRect(GetDC(hwnd), &client_rect, brush);
            DeleteObject(brush);
        }
        return 1;

    case WM_SYSCOMMAND:
    {
        if (!g_WindowsEnableMenuMnemonics) {
            if ((wParam & 0xFFF0) == SC_KEYMENU) {
                return 0;
            }
        }

#if defined(SC_SCREENSAVE) || defined(SC_MONITORPOWER)
        // Don't start the screensaver or blank the monitor in fullscreen apps
        if ((wParam & 0xFFF0) == SC_SCREENSAVE ||
            (wParam & 0xFFF0) == SC_MONITORPOWER) {
            if (SDL_GetVideoDevice()->suspend_screensaver) {
                return 0;
            }
        }
#endif // System has screensaver support
    } break;

    case WM_CLOSE:
    {
        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_CLOSE_REQUESTED, 0, 0);
    }
        returnCode = 0;
        break;

    case WM_TOUCH:
        if (data->videodata->GetTouchInputInfo && data->videodata->CloseTouchInputHandle) {
            UINT i, num_inputs = LOWORD(wParam);
            bool isstack;
            PTOUCHINPUT inputs = SDL_small_alloc(TOUCHINPUT, num_inputs, &isstack);
            if (inputs && data->videodata->GetTouchInputInfo((HTOUCHINPUT)lParam, num_inputs, inputs, sizeof(TOUCHINPUT))) {
                RECT rect;
                float x, y;

                if (!GetClientRect(hwnd, &rect) || !WIN_WindowRectValid(&rect)) {
                    if (inputs) {
                        SDL_small_free(inputs, isstack);
                    }
                    break;
                }
                ClientToScreen(hwnd, (LPPOINT)&rect);
                ClientToScreen(hwnd, (LPPOINT)&rect + 1);
                rect.top *= 100;
                rect.left *= 100;
                rect.bottom *= 100;
                rect.right *= 100;

                for (i = 0; i < num_inputs; ++i) {
                    PTOUCHINPUT input = &inputs[i];
                    const int w = (rect.right - rect.left);
                    const int h = (rect.bottom - rect.top);

                    const SDL_TouchID touchId = (SDL_TouchID)((uintptr_t)input->hSource);
                    const SDL_FingerID fingerId = (input->dwID + 1);

                    /* TODO: Can we use GetRawInputDeviceInfo and HID info to
                       determine if this is a direct or indirect touch device?
                     */
                    if (SDL_AddTouch(touchId, SDL_TOUCH_DEVICE_DIRECT, (input->dwFlags & TOUCHEVENTF_PEN) == TOUCHEVENTF_PEN ? "pen" : "touch") < 0) {
                        continue;
                    }

                    // Get the normalized coordinates for the window
                    if (w <= 1) {
                        x = 0.5f;
                    } else {
                        x = (float)(input->x - rect.left) / (w - 1);
                    }
                    if (h <= 1) {
                        y = 0.5f;
                    } else {
                        y = (float)(input->y - rect.top) / (h - 1);
                    }

                    // FIXME: Should we use the input->dwTime field for the tick source of the timestamp?
                    if (input->dwFlags & TOUCHEVENTF_DOWN) {
                        SDL_SendTouch(WIN_GetEventTimestamp(), touchId, fingerId, data->window, SDL_EVENT_FINGER_DOWN, x, y, 1.0f);
                    }
                    if (input->dwFlags & TOUCHEVENTF_MOVE) {
                        SDL_SendTouchMotion(WIN_GetEventTimestamp(), touchId, fingerId, data->window, x, y, 1.0f);
                    }
                    if (input->dwFlags & TOUCHEVENTF_UP) {
                        SDL_SendTouch(WIN_GetEventTimestamp(), touchId, fingerId, data->window, SDL_EVENT_FINGER_UP, x, y, 1.0f);
                    }
                }
            }
            SDL_small_free(inputs, isstack);

            data->videodata->CloseTouchInputHandle((HTOUCHINPUT)lParam);
            return 0;
        }
        break;

#ifdef HAVE_TPCSHRD_H

    case WM_TABLET_QUERYSYSTEMGESTURESTATUS:
        /* See https://msdn.microsoft.com/en-us/library/windows/desktop/bb969148(v=vs.85).aspx .
         * If we're handling our own touches, we don't want any gestures.
         * Not all of these settings are documented.
         * The use of the undocumented ones was suggested by https://github.com/bjarkeck/GCGJ/blob/master/Monogame/Windows/WinFormsGameForm.cs . */
        return TABLET_DISABLE_PRESSANDHOLD | TABLET_DISABLE_PENTAPFEEDBACK | TABLET_DISABLE_PENBARRELFEEDBACK | TABLET_DISABLE_TOUCHUIFORCEON | TABLET_DISABLE_TOUCHUIFORCEOFF | TABLET_DISABLE_TOUCHSWITCH | TABLET_DISABLE_FLICKS | TABLET_DISABLE_SMOOTHSCROLLING | TABLET_DISABLE_FLICKFALLBACKKEYS; // disables press and hold (right-click) gesture
                                                                                                                                                                                                                                                                                                         // disables UI feedback on pen up (waves)
                                                                                                                                                                                                                                                                                                         // disables UI feedback on pen button down (circle)
                                                                                                                                                                                                                                                                                                         // disables pen flicks (back, forward, drag down, drag up)

#endif // HAVE_TPCSHRD_H

    case WM_DROPFILES:
    {
        UINT i;
        HDROP drop = (HDROP)wParam;
        UINT count = DragQueryFile(drop, 0xFFFFFFFF, NULL, 0);
        for (i = 0; i < count; ++i) {
            UINT size = DragQueryFile(drop, i, NULL, 0) + 1;
            LPTSTR buffer = (LPTSTR)SDL_malloc(sizeof(TCHAR) * size);
            if (buffer) {
                if (DragQueryFile(drop, i, buffer, size)) {
                    char *file = WIN_StringToUTF8(buffer);
                    SDL_SendDropFile(data->window, NULL, file);
                    SDL_free(file);
                }
                SDL_free(buffer);
            }
        }
        SDL_SendDropComplete(data->window);
        DragFinish(drop);
        return 0;
    } break;

    case WM_DISPLAYCHANGE:
    {
        // Reacquire displays if any were added or removed
        WIN_RefreshDisplays(SDL_GetVideoDevice());
    } break;

    case WM_NCCALCSIZE:
    {
        SDL_WindowFlags window_flags = SDL_GetWindowFlags(data->window);
        if (wParam == TRUE && (window_flags & SDL_WINDOW_BORDERLESS) && !(window_flags & SDL_WINDOW_FULLSCREEN)) {
            // When borderless, need to tell windows that the size of the non-client area is 0
            NCCALCSIZE_PARAMS *params = (NCCALCSIZE_PARAMS *)lParam;
            WINDOWPLACEMENT placement;
            if (GetWindowPlacement(hwnd, &placement) && placement.showCmd == SW_MAXIMIZE) {
                // Maximized borderless windows should use the monitor work area.
                HMONITOR hMonitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONULL);
                if (!hMonitor) {
                    // The returned monitor can be null when restoring from minimized, so use the last coordinates.
                    const POINT pt = { data->window->windowed.x, data->window->windowed.y };
                    hMonitor = MonitorFromPoint(pt, MONITOR_DEFAULTTONEAREST);
                }
                if (hMonitor) {
                    MONITORINFO info;
                    SDL_zero(info);
                    info.cbSize = sizeof(info);
                    if (GetMonitorInfo(hMonitor, &info)) {
                        params->rgrc[0] = info.rcWork;
                    }
                }
            } else if (!(window_flags & SDL_WINDOW_RESIZABLE) && !data->force_ws_maximizebox) {
                int w, h;
                if (data->window->last_size_pending) {
                    w = data->window->pending.w;
                    h = data->window->pending.h;
                } else {
                    w = data->window->floating.w;
                    h = data->window->floating.h;
                }
                params->rgrc[0].right = params->rgrc[0].left + w;
                params->rgrc[0].bottom = params->rgrc[0].top + h;
            }
            return 0;
        }
    } break;

    case WM_NCHITTEST:
    {
        SDL_Window *window = data->window;

        if (window->flags & SDL_WINDOW_TOOLTIP) {
            return HTTRANSPARENT;
        }

        if (window->hit_test) {
            POINT winpoint;
            winpoint.x = GET_X_LPARAM(lParam);
            winpoint.y = GET_Y_LPARAM(lParam);
            if (ScreenToClient(hwnd, &winpoint)) {
                SDL_Point point;
                SDL_HitTestResult rc;
                point.x = winpoint.x;
                point.y = winpoint.y;
                rc = window->hit_test(window, &point, window->hit_test_data);
                switch (rc) {
#define POST_HIT_TEST(ret)                                                 \
    {                                                                      \
        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_HIT_TEST, 0, 0); \
        return ret;                                                        \
    }
                case SDL_HITTEST_DRAGGABLE:
                {
                    /* If the mouse button state is something other than none or left button down,
                     * return HTCLIENT, or Windows will eat the button press.
                     */
                    SDL_MouseButtonFlags buttonState = SDL_GetGlobalMouseState(NULL, NULL);
                    if (buttonState && !(buttonState & SDL_BUTTON_LMASK)) {
                        // Set focus in case it was lost while previously moving over a draggable area.
                        SDL_SetMouseFocus(window);
                        return HTCLIENT;
                    }

                    POST_HIT_TEST(HTCAPTION);
                }
                case SDL_HITTEST_RESIZE_TOPLEFT:
                    POST_HIT_TEST(HTTOPLEFT);
                case SDL_HITTEST_RESIZE_TOP:
                    POST_HIT_TEST(HTTOP);
                case SDL_HITTEST_RESIZE_TOPRIGHT:
                    POST_HIT_TEST(HTTOPRIGHT);
                case SDL_HITTEST_RESIZE_RIGHT:
                    POST_HIT_TEST(HTRIGHT);
                case SDL_HITTEST_RESIZE_BOTTOMRIGHT:
                    POST_HIT_TEST(HTBOTTOMRIGHT);
                case SDL_HITTEST_RESIZE_BOTTOM:
                    POST_HIT_TEST(HTBOTTOM);
                case SDL_HITTEST_RESIZE_BOTTOMLEFT:
                    POST_HIT_TEST(HTBOTTOMLEFT);
                case SDL_HITTEST_RESIZE_LEFT:
                    POST_HIT_TEST(HTLEFT);
#undef POST_HIT_TEST
                case SDL_HITTEST_NORMAL:
                    return HTCLIENT;
                }
            }
            // If we didn't return, this will call DefWindowProc below.
        }
    } break;

    case WM_GETDPISCALEDSIZE:
        // Windows 10 Creators Update+
        /* Documented as only being sent to windows that are per-monitor V2 DPI aware.

           Experimentation shows it's only sent during interactive dragging, not in response to
           SetWindowPos. */
        if (data->videodata->GetDpiForWindow && data->videodata->AdjustWindowRectExForDpi) {
            /* Windows expects applications to scale their window rects linearly
               when dragging between monitors with different DPI's.
               e.g. a 100x100 window dragged to a 200% scaled monitor
               becomes 200x200.

               For SDL, we instead want the client size to scale linearly.
               This is not the same as the window rect scaling linearly,
               because Windows doesn't scale the non-client area (titlebar etc.)
               linearly. So, we need to handle this message to request custom
               scaling. */

            const int nextDPI = (int)wParam;
            const int prevDPI = (int)data->videodata->GetDpiForWindow(hwnd);
            SIZE *sizeInOut = (SIZE *)lParam;

            int frame_w, frame_h;
            int query_client_w_win, query_client_h_win;

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_GETDPISCALEDSIZE: current DPI: %d potential DPI: %d input size: (%dx%d)",
                    prevDPI, nextDPI, sizeInOut->cx, sizeInOut->cy);
#endif

            // Subtract the window frame size that would have been used at prevDPI
            {
                RECT rect = { 0 };

                if (!(data->window->flags & SDL_WINDOW_BORDERLESS) && !SDL_WINDOW_IS_POPUP(data->window)) {
                    WIN_AdjustWindowRectForHWND(hwnd, &rect, prevDPI);
                }

                frame_w = -rect.left + rect.right;
                frame_h = -rect.top + rect.bottom;

                query_client_w_win = sizeInOut->cx - frame_w;
                query_client_h_win = sizeInOut->cy - frame_h;
            }

            // Add the window frame size that would be used at nextDPI
            {
                RECT rect = { 0 };
                rect.right = query_client_w_win;
                rect.bottom = query_client_h_win;

                if (!(data->window->flags & SDL_WINDOW_BORDERLESS) && !SDL_WINDOW_IS_POPUP(data->window)) {
                    WIN_AdjustWindowRectForHWND(hwnd, &rect, nextDPI);
                }

                // This is supposed to control the suggested rect param of WM_DPICHANGED
                sizeInOut->cx = rect.right - rect.left;
                sizeInOut->cy = rect.bottom - rect.top;
            }

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_GETDPISCALEDSIZE: output size: (%dx%d)", sizeInOut->cx, sizeInOut->cy);
#endif
            return TRUE;
        }
        break;

    case WM_DPICHANGED:
        // Windows 8.1+
        {
            const int newDPI = HIWORD(wParam);
            RECT *const suggestedRect = (RECT *)lParam;
            int w, h;

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_DPICHANGED: to %d\tsuggested rect: (%d, %d), (%dx%d)", newDPI,
                    suggestedRect->left, suggestedRect->top, suggestedRect->right - suggestedRect->left, suggestedRect->bottom - suggestedRect->top);
#endif

            if (data->expected_resize) {
                /* This DPI change is coming from an explicit SetWindowPos call within SDL.
                   Assume all call sites are calculating the DPI-aware frame correctly, so
                   we don't need to do any further adjustment. */
#ifdef HIGHDPI_DEBUG
                SDL_Log("WM_DPICHANGED: Doing nothing, assuming window is already sized correctly");
#endif
                return 0;
            }

            // Interactive user-initiated resizing/movement
            {
                /* Calculate the new frame w/h such that
                   the client area size is maintained. */
                RECT rect = { 0 };
                rect.right = data->window->w;
                rect.bottom = data->window->h;

                if (!(data->window->flags & SDL_WINDOW_BORDERLESS)) {
                    WIN_AdjustWindowRectForHWND(hwnd, &rect, newDPI);
                }

                w = rect.right - rect.left;
                h = rect.bottom - rect.top;
            }

#ifdef HIGHDPI_DEBUG
            SDL_Log("WM_DPICHANGED: current SDL window size: (%dx%d)\tcalling SetWindowPos: (%d, %d), (%dx%d)",
                    data->window->w, data->window->h,
                    suggestedRect->left, suggestedRect->top, w, h);
#endif

            data->expected_resize = true;
            SetWindowPos(hwnd,
                         NULL,
                         suggestedRect->left,
                         suggestedRect->top,
                         w,
                         h,
                         SWP_NOZORDER | SWP_NOACTIVATE);
            data->expected_resize = false;
            return 0;
        }
        break;

    case WM_SETTINGCHANGE:
        if (wParam == 0 && lParam != 0 && SDL_wcscmp((wchar_t *)lParam, L"ImmersiveColorSet") == 0) {
            SDL_SetSystemTheme(WIN_GetSystemTheme());
            WIN_UpdateDarkModeForHWND(hwnd);
        }
        if (wParam == SPI_SETMOUSE || wParam == SPI_SETMOUSESPEED) {
            WIN_UpdateMouseSystemScale();
        }
        break;

#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    }

#ifdef HAVE_SHOBJIDL_CORE_H
    if (msg == data->videodata->WM_TASKBAR_BUTTON_CREATED) {
        data->taskbar_button_created = true;
        WIN_ApplyWindowProgress(SDL_GetVideoDevice(), data->window);
    }
#endif

    // If there's a window proc, assume it's going to handle messages
    if (data->wndproc) {
        return CallWindowProc(data->wndproc, hwnd, msg, wParam, lParam);
    } else if (returnCode >= 0) {
        return returnCode;
    } else {
        return CallWindowProc(DefWindowProc, hwnd, msg, wParam, lParam);
    }
}

int WIN_WaitEventTimeout(SDL_VideoDevice *_this, Sint64 timeoutNS)
{
    if (g_WindowsEnableMessageLoop) {
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        DWORD timeout, ret;
        timeout = timeoutNS < 0 ? INFINITE : (DWORD)SDL_NS_TO_MS(timeoutNS);
        ret = MsgWaitForMultipleObjects(0, NULL, FALSE, timeout, QS_ALLINPUT);
        if (ret == WAIT_OBJECT_0) {
            return 1;
        } else {
            return 0;
        }
#else
        // MsgWaitForMultipleObjects is desktop-only.
        MSG msg;
        BOOL message_result;
        UINT_PTR timer_id = 0;
        if (timeoutNS > 0) {
            timer_id = SetTimer(NULL, 0, (UINT)SDL_NS_TO_MS(timeoutNS), NULL);
            message_result = GetMessage(&msg, 0, 0, 0);
            KillTimer(NULL, timer_id);
        } else if (timeoutNS == 0) {
            message_result = PeekMessage(&msg, NULL, 0, 0, PM_REMOVE);
        } else {
            message_result = GetMessage(&msg, 0, 0, 0);
        }
        if (message_result) {
            if (msg.message == WM_TIMER && !msg.hwnd && msg.wParam == timer_id) {
                return 0;
            }
            if (g_WindowsMessageHook) {
                if (!g_WindowsMessageHook(g_WindowsMessageHookData, &msg)) {
                    return 1;
                }
            }
            // Always translate the message in case it's a non-SDL window (e.g. with Qt integration)
            TranslateMessage(&msg);
            DispatchMessage(&msg);
            return 1;
        } else {
            return 0;
        }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    } else {
        // Fail the wait so the caller falls back to polling
        return -1;
    }
}

void WIN_SendWakeupEvent(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    PostMessage(data->hwnd, data->videodata->_SDL_WAKEUP, 0, 0);
}

void WIN_PumpEvents(SDL_VideoDevice *_this)
{
    MSG msg;
#ifdef _MSC_VER // We explicitly want to use GetTickCount(), not GetTickCount64()
#pragma warning(push)
#pragma warning(disable : 28159)
#endif
    DWORD end_ticks = GetTickCount() + 1;
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    int new_messages = 0;

    if (_this->internal->gameinput_context) {
        WIN_UpdateGameInput(_this);
    }

    if (g_WindowsEnableMessageLoop) {
        SDL_processing_messages = true;

        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (g_WindowsMessageHook) {
                if (!g_WindowsMessageHook(g_WindowsMessageHookData, &msg)) {
                    continue;
                }
            }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
            // Don't dispatch any mouse motion queued prior to or including the last mouse warp
            if (msg.message == WM_MOUSEMOVE && SDL_last_warp_time) {
                if (!SDL_TICKS_PASSED(msg.time, (SDL_last_warp_time + 1))) {
                    continue;
                }

                // This mouse message happened after the warp
                SDL_last_warp_time = 0;
            }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

            WIN_SetMessageTick(msg.time);

            // Always translate the message in case it's a non-SDL window (e.g. with Qt integration)
            TranslateMessage(&msg);
            DispatchMessage(&msg);

            // Make sure we don't busy loop here forever if there are lots of events coming in
            if (SDL_TICKS_PASSED(msg.time, end_ticks)) {
                /* We might get a few new messages generated by the Steam overlay or other application hooks
                   In this case those messages will be processed before any pending input, so we want to continue after those messages.
                   (thanks to Peter Deayton for his investigation here)
                 */
                const int MAX_NEW_MESSAGES = 3;
                ++new_messages;
                if (new_messages > MAX_NEW_MESSAGES) {
                    break;
                }
            }
        }

        SDL_processing_messages = false;
    }

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    /* Windows loses a shift KEYUP event when you have both pressed at once and let go of one.
       You won't get a KEYUP until both are released, and that keyup will only be for the second
       key you released. Take heroic measures and check the keystate as of the last handled event,
       and if we think a key is pressed when Windows doesn't, unstick it in SDL's state. */
    const bool *keystate = SDL_GetKeyboardState(NULL);
    if (keystate[SDL_SCANCODE_LSHIFT] && !(GetKeyState(VK_LSHIFT) & 0x8000)) {
        SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_LSHIFT, false);
    }
    if (keystate[SDL_SCANCODE_RSHIFT] && !(GetKeyState(VK_RSHIFT) & 0x8000)) {
        SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_RSHIFT, false);
    }

    /* The Windows key state gets lost when using Windows+Space or Windows+G shortcuts and
       not grabbing the keyboard. Note: If we *are* grabbing the keyboard, GetKeyState()
       will return inaccurate results for VK_LWIN and VK_RWIN but we don't need it anyway. */
    SDL_Window *focusWindow = SDL_GetKeyboardFocus();
    if (!focusWindow || !(focusWindow->flags & SDL_WINDOW_KEYBOARD_GRABBED)) {
        if (keystate[SDL_SCANCODE_LGUI] && !(GetKeyState(VK_LWIN) & 0x8000)) {
            SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_LGUI, false);
        }
        if (keystate[SDL_SCANCODE_RGUI] && !(GetKeyState(VK_RWIN) & 0x8000)) {
            SDL_SendKeyboardKey(0, SDL_GLOBAL_KEYBOARD_ID, 0, SDL_SCANCODE_RGUI, false);
        }
    }

    // fire queued clipcursor refreshes
    if (_this) {
        SDL_Window *window = _this->windows;
        while (window) {
            bool refresh_clipcursor = false;
            SDL_WindowData *data = window->internal;
            if (data) {
                refresh_clipcursor = data->clipcursor_queued;
                data->clipcursor_queued = false;    // Must be cleared unconditionally.
                data->postpone_clipcursor = false;  // Must be cleared unconditionally.
                                                    // Must happen before UpdateClipCursor.
                                                    // Although its occurrence currently
                                                    // always coincides with the queuing of
                                                    // clipcursor, it is logically distinct
                                                    // and this coincidence might no longer
                                                    // be true in the future.
                                                    // Ergo this placement concordantly
                                                    // conveys its unconditionality 
                                                    // vis-a-vis the queuing of clipcursor.
            }
            if (refresh_clipcursor) {
                WIN_UpdateClipCursor(window);
            }
            window = window->next;
        }
    }

    // Synchronize internal mouse capture state to the most current cursor state
    // since for whatever reason we are not depending exclusively on SetCapture/
    // ReleaseCapture to pipe in out-of-window mouse events.
    // Formerly WIN_UpdateMouseCapture().
    // TODO: can this go before clipcursor?
    focusWindow = SDL_GetKeyboardFocus();
    if (focusWindow && (focusWindow->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
        SDL_WindowData *data = focusWindow->internal;

        if (!data->mouse_tracked) {
            POINT cursorPos;

            if (GetCursorPos(&cursorPos) && ScreenToClient(data->hwnd, &cursorPos)) {
                bool swapButtons = GetSystemMetrics(SM_SWAPBUTTON) != 0;
                SDL_MouseID mouseID = SDL_GLOBAL_MOUSE_ID;

                SDL_SendMouseMotion(WIN_GetEventTimestamp(), data->window, mouseID, false, (float)cursorPos.x, (float)cursorPos.y);
                SDL_SendMouseButton(WIN_GetEventTimestamp(), data->window, mouseID,
                                    !swapButtons ? SDL_BUTTON_LEFT : SDL_BUTTON_RIGHT,
                                    (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0);
                SDL_SendMouseButton(WIN_GetEventTimestamp(), data->window, mouseID,
                                    !swapButtons ? SDL_BUTTON_RIGHT : SDL_BUTTON_LEFT,
                                    (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0);
                SDL_SendMouseButton(WIN_GetEventTimestamp(), data->window, mouseID,
                                    SDL_BUTTON_MIDDLE,
                                    (GetAsyncKeyState(VK_MBUTTON) & 0x8000) != 0);
                SDL_SendMouseButton(WIN_GetEventTimestamp(), data->window, mouseID,
                                    SDL_BUTTON_X1,
                                    (GetAsyncKeyState(VK_XBUTTON1) & 0x8000) != 0);
                SDL_SendMouseButton(WIN_GetEventTimestamp(), data->window, mouseID,
                                    SDL_BUTTON_X2,
                                    (GetAsyncKeyState(VK_XBUTTON2) & 0x8000) != 0);
            }
        }
    }

    if (!_this->internal->gameinput_context) {
        WIN_CheckKeyboardAndMouseHotplug(_this, false);
    }

    WIN_UpdateIMECandidates(_this);

#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

#ifdef SDL_PLATFORM_GDK
    GDK_DispatchTaskQueue();
#endif
}

static int app_registered = 0;
LPTSTR SDL_Appname = NULL;
Uint32 SDL_Appstyle = 0;
HINSTANCE SDL_Instance = NULL;

static void WIN_CleanRegisterApp(WNDCLASSEX wcex)
{
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    if (wcex.hIcon) {
        DestroyIcon(wcex.hIcon);
    }
    if (wcex.hIconSm) {
        DestroyIcon(wcex.hIconSm);
    }
#endif
    SDL_free(SDL_Appname);
    SDL_Appname = NULL;
}

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
static BOOL CALLBACK WIN_ResourceNameCallback(HMODULE hModule, LPCTSTR lpType, LPTSTR lpName, LONG_PTR lParam)
{
    WNDCLASSEX *wcex = (WNDCLASSEX *)lParam;

    (void)lpType; // We already know that the resource type is RT_GROUP_ICON.

    /* We leave hIconSm as NULL as it will allow Windows to automatically
       choose the appropriate small icon size to suit the current DPI. */
    wcex->hIcon = LoadIcon(hModule, lpName);

    // Do not bother enumerating any more.
    return FALSE;
}
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

// Register the class for this application
bool SDL_RegisterApp(const char *name, Uint32 style, void *hInst)
{
    WNDCLASSEX wcex;
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    const char *hint;
#endif

    // Only do this once...
    if (app_registered) {
        ++app_registered;
        return true;
    }
    SDL_assert(!SDL_Appname);
    if (!name) {
        name = "SDL_app";
#if defined(CS_BYTEALIGNCLIENT) || defined(CS_OWNDC)
        style = (CS_BYTEALIGNCLIENT | CS_OWNDC);
#endif
    }
    SDL_Appname = WIN_UTF8ToString(name);
    SDL_Appstyle = style;
    SDL_Instance = hInst ? (HINSTANCE)hInst : GetModuleHandle(NULL);

    // Register the application class
    wcex.cbSize = sizeof(WNDCLASSEX);
    wcex.hCursor = NULL;
    wcex.hIcon = NULL;
    wcex.hIconSm = NULL;
    wcex.lpszMenuName = NULL;
    wcex.lpszClassName = SDL_Appname;
    wcex.style = SDL_Appstyle;
    wcex.hbrBackground = NULL;
    wcex.lpfnWndProc = WIN_WindowProc;
    wcex.hInstance = SDL_Instance;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;

#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
    hint = SDL_GetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON);
    if (hint && *hint) {
        wcex.hIcon = LoadIcon(SDL_Instance, MAKEINTRESOURCE(SDL_atoi(hint)));

        hint = SDL_GetHint(SDL_HINT_WINDOWS_INTRESOURCE_ICON_SMALL);
        if (hint && *hint) {
            wcex.hIconSm = LoadIcon(SDL_Instance, MAKEINTRESOURCE(SDL_atoi(hint)));
        }
    } else {
        // Use the first icon as a default icon, like in the Explorer.
        EnumResourceNames(SDL_Instance, RT_GROUP_ICON, WIN_ResourceNameCallback, (LONG_PTR)&wcex);
    }
#endif // !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)

    if (!RegisterClassEx(&wcex)) {
        WIN_CleanRegisterApp(wcex);
        return SDL_SetError("Couldn't register application class");
    }

    app_registered = 1;
    return true;
}

// Unregisters the windowclass registered in SDL_RegisterApp above.
void SDL_UnregisterApp(void)
{
    WNDCLASSEX wcex;

    // SDL_RegisterApp might not have been called before
    if (!app_registered) {
        return;
    }
    --app_registered;
    if (app_registered == 0) {
        // Ensure the icons are initialized.
        wcex.hIcon = NULL;
        wcex.hIconSm = NULL;
        // Check for any registered window classes.
#if !defined(SDL_PLATFORM_XBOXONE) && !defined(SDL_PLATFORM_XBOXSERIES)
        if (GetClassInfoEx(SDL_Instance, SDL_Appname, &wcex)) {
            UnregisterClass(SDL_Appname, SDL_Instance);
        }
#endif
        WIN_CleanRegisterApp(wcex);
    }
}

#endif // SDL_VIDEO_DRIVER_WINDOWS
