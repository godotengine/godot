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

#ifdef SDL_VIDEO_DRIVER_X11

#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>
#include <limits.h> // For INT_MAX

#include "SDL_x11video.h"
#include "SDL_x11pen.h"
#include "SDL_x11touch.h"
#include "SDL_x11xinput2.h"
#include "SDL_x11xfixes.h"
#include "SDL_x11settings.h"
#include "../SDL_clipboard_c.h"
#include "SDL_x11xsync.h"
#include "../../core/unix/SDL_poll.h"
#include "../../events/SDL_events_c.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_touch_c.h"
#include "../../core/linux/SDL_system_theme.h"
#include "../SDL_sysvideo.h"

#include <stdio.h>

#if 0
#define DEBUG_XEVENTS
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_TOPLEFT
#define _NET_WM_MOVERESIZE_SIZE_TOPLEFT 0
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_TOP
#define _NET_WM_MOVERESIZE_SIZE_TOP 1
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_TOPRIGHT
#define _NET_WM_MOVERESIZE_SIZE_TOPRIGHT 2
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_RIGHT
#define _NET_WM_MOVERESIZE_SIZE_RIGHT 3
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_BOTTOMRIGHT
#define _NET_WM_MOVERESIZE_SIZE_BOTTOMRIGHT 4
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_BOTTOM
#define _NET_WM_MOVERESIZE_SIZE_BOTTOM 5
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_BOTTOMLEFT
#define _NET_WM_MOVERESIZE_SIZE_BOTTOMLEFT 6
#endif

#ifndef _NET_WM_MOVERESIZE_SIZE_LEFT
#define _NET_WM_MOVERESIZE_SIZE_LEFT 7
#endif

#ifndef _NET_WM_MOVERESIZE_MOVE
#define _NET_WM_MOVERESIZE_MOVE 8
#endif

typedef struct
{
    unsigned char *data;
    int format, count;
    Atom type;
} SDL_x11Prop;

/* Reads property
   Must call X11_XFree on results
 */
static void X11_ReadProperty(SDL_x11Prop *p, Display *disp, Window w, Atom prop)
{
    unsigned char *ret = NULL;
    Atom type;
    int fmt;
    unsigned long count;
    unsigned long bytes_left;
    int bytes_fetch = 0;

    do {
        if (ret) {
            X11_XFree(ret);
        }
        X11_XGetWindowProperty(disp, w, prop, 0, bytes_fetch, False, AnyPropertyType, &type, &fmt, &count, &bytes_left, &ret);
        bytes_fetch += bytes_left;
    } while (bytes_left != 0);

    p->data = ret;
    p->format = fmt;
    p->count = count;
    p->type = type;
}

/* Find text-uri-list in a list of targets and return it's atom
   if available, else return None */
static Atom X11_PickTarget(Display *disp, Atom list[], int list_count)
{
    Atom request = None;
    char *name;
    int i;
    for (i = 0; i < list_count && request == None; i++) {
        name = X11_XGetAtomName(disp, list[i]);
        // Preferred MIME targets
        if ((SDL_strcmp("text/uri-list", name) == 0) ||
            (SDL_strcmp("text/plain;charset=utf-8", name) == 0) ||
            (SDL_strcmp("UTF8_STRING", name) == 0)) {
            request = list[i];
        }
        // Fallback MIME targets
        if ((SDL_strcmp("text/plain", name) == 0) ||
            (SDL_strcmp("TEXT", name) == 0)) {
            if (request == None) {
                request = list[i];
            }
        }
        X11_XFree(name);
    }
    return request;
}

/* Wrapper for X11_PickTarget for a maximum of three targets, a special
   case in the Xdnd protocol */
static Atom X11_PickTargetFromAtoms(Display *disp, Atom a0, Atom a1, Atom a2)
{
    int count = 0;
    Atom atom[3];
    if (a0 != None) {
        atom[count++] = a0;
    }
    if (a1 != None) {
        atom[count++] = a1;
    }
    if (a2 != None) {
        atom[count++] = a2;
    }
    return X11_PickTarget(disp, atom, count);
}

struct KeyRepeatCheckData
{
    XEvent *event;
    bool found;
};

static Bool X11_KeyRepeatCheckIfEvent(Display *display, XEvent *chkev,
                                      XPointer arg)
{
    struct KeyRepeatCheckData *d = (struct KeyRepeatCheckData *)arg;
    if (chkev->type == KeyPress && chkev->xkey.keycode == d->event->xkey.keycode && chkev->xkey.time - d->event->xkey.time < 2) {
        d->found = true;
    }
    return False;
}

/* Check to see if this is a repeated key.
   (idea shamelessly lifted from GII -- thanks guys! :)
 */
static bool X11_KeyRepeat(Display *display, XEvent *event)
{
    XEvent dummyev;
    struct KeyRepeatCheckData d;
    d.event = event;
    d.found = false;
    if (X11_XPending(display)) {
        X11_XCheckIfEvent(display, &dummyev, X11_KeyRepeatCheckIfEvent, (XPointer)&d);
    }
    return d.found;
}

static bool X11_IsWheelEvent(Display *display, int button, int *xticks, int *yticks)
{
    /* according to the xlib docs, no specific mouse wheel events exist.
       However, the defacto standard is that the vertical wheel is X buttons
       4 (up) and 5 (down) and a horizontal wheel is 6 (left) and 7 (right). */

    // Xlib defines "Button1" through 5, so we just use literals here.
    switch (button) {
    case 4:
        *yticks = 1;
        return true;
    case 5:
        *yticks = -1;
        return true;
    case 6:
        *xticks = 1;
        return true;
    case 7:
        *xticks = -1;
        return true;
    default:
        break;
    }
    return false;
}

// An X11 event hook
static SDL_X11EventHook g_X11EventHook = NULL;
static void *g_X11EventHookData = NULL;

void SDL_SetX11EventHook(SDL_X11EventHook callback, void *userdata)
{
    g_X11EventHook = callback;
    g_X11EventHookData = userdata;
}

#ifdef SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS
static void X11_HandleGenericEvent(SDL_VideoDevice *_this, XEvent *xev)
{
    SDL_VideoData *videodata = _this->internal;

    // event is a union, so cookie == &event, but this is type safe.
    XGenericEventCookie *cookie = &xev->xcookie;
    if (X11_XGetEventData(videodata->display, cookie)) {
        if (!g_X11EventHook || g_X11EventHook(g_X11EventHookData, xev)) {
            X11_HandleXinput2Event(_this, cookie);
        }
        X11_XFreeEventData(videodata->display, cookie);
    }
}
#endif // SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS

static void X11_UpdateSystemKeyModifiers(SDL_VideoData *viddata)
{
    Window junk_window;
    int x, y;

    X11_XQueryPointer(viddata->display, DefaultRootWindow(viddata->display), &junk_window, &junk_window, &x, &y, &x, &y, &viddata->xkb.xkb_modifiers);
}

static void X11_ReconcileModifiers(SDL_VideoData *viddata)
{
    const Uint32 xk_modifiers = viddata->xkb.xkb_modifiers;

    /* If a modifier was activated by a keypress, it will be tied to the
     * specific left/right key that initiated it. Otherwise, the ambiguous
     * left/right combo is used.
     */
    if (xk_modifiers & ShiftMask) {
        if (!(viddata->xkb.sdl_modifiers & SDL_KMOD_SHIFT)) {
            viddata->xkb.sdl_modifiers |= SDL_KMOD_SHIFT;
        }
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_SHIFT;
    }

    if (xk_modifiers & ControlMask) {
        if (!(viddata->xkb.sdl_modifiers & SDL_KMOD_CTRL)) {
            viddata->xkb.sdl_modifiers |= SDL_KMOD_CTRL;
        }
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_CTRL;
    }

    // Mod1 is used for the Alt keys
    if (xk_modifiers & Mod1Mask) {
        if (!(viddata->xkb.sdl_modifiers & SDL_KMOD_ALT)) {
            viddata->xkb.sdl_modifiers |= SDL_KMOD_ALT;
        }
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_ALT;
    }

    // Mod4 is used for the Super (aka GUI/Logo) keys.
    if (xk_modifiers & Mod4Mask) {
        if (!(viddata->xkb.sdl_modifiers & SDL_KMOD_GUI)) {
            viddata->xkb.sdl_modifiers |= SDL_KMOD_GUI;
        }
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_GUI;
    }

    // Mod3 is typically Level 5 shift.
    if (xk_modifiers & Mod3Mask) {
        viddata->xkb.sdl_modifiers |= SDL_KMOD_LEVEL5;
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_LEVEL5;
    }

    // Mod5 is typically Level 3 shift (aka AltGr).
    if (xk_modifiers & Mod5Mask) {
        viddata->xkb.sdl_modifiers |= SDL_KMOD_MODE;
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_MODE;
    }

    if (xk_modifiers & LockMask) {
        viddata->xkb.sdl_modifiers |= SDL_KMOD_CAPS;
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_CAPS;
    }

    if (xk_modifiers & viddata->xkb.numlock_mask) {
        viddata->xkb.sdl_modifiers |= SDL_KMOD_NUM;
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_NUM;
    }

    if (xk_modifiers & viddata->xkb.scrolllock_mask) {
        viddata->xkb.sdl_modifiers |= SDL_KMOD_SCROLL;
    } else {
        viddata->xkb.sdl_modifiers &= ~SDL_KMOD_SCROLL;
    }

    SDL_SetModState(viddata->xkb.sdl_modifiers);
}

static void X11_HandleModifierKeys(SDL_VideoData *viddata, SDL_Scancode scancode, bool pressed, bool allow_reconciliation)
{
    const SDL_Keycode keycode = SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false);
    SDL_Keymod mod = SDL_KMOD_NONE;
    bool reconcile = false;

    /* SDL clients expect modifier state to be activated at the same time as the
     * source keypress, so we set pressed modifier state with the usual modifier
     * keys here, as the explicit modifier event won't arrive until after the
     * keypress event. If this is wrong, it will be corrected when the explicit
     * modifier state is checked.
     */
    switch (keycode) {
    case SDLK_LSHIFT:
        mod = SDL_KMOD_LSHIFT;
        break;
    case SDLK_RSHIFT:
        mod = SDL_KMOD_RSHIFT;
        break;
    case SDLK_LCTRL:
        mod = SDL_KMOD_LCTRL;
        break;
    case SDLK_RCTRL:
        mod = SDL_KMOD_RCTRL;
        break;
    case SDLK_LALT:
        mod = SDL_KMOD_LALT;
        break;
    case SDLK_RALT:
        mod = SDL_KMOD_RALT;
        break;
    case SDLK_LGUI:
        mod = SDL_KMOD_LGUI;
        break;
    case SDLK_RGUI:
        mod = SDL_KMOD_RGUI;
        break;
    case SDLK_MODE:
        mod = SDL_KMOD_MODE;
        break;
    case SDLK_LEVEL5_SHIFT:
        mod = SDL_KMOD_LEVEL5;
        break;
    case SDLK_CAPSLOCK:
    case SDLK_NUMLOCKCLEAR:
    case SDLK_SCROLLLOCK:
    {
        /* For locking modifier keys, query the lock state directly, or we may have to wait until the next
         * key press event to know if a lock was actually activated from the key event.
         */
        unsigned int cur_mask = viddata->xkb.xkb_modifiers;
        X11_UpdateSystemKeyModifiers(viddata);

        if (viddata->xkb.xkb_modifiers & LockMask) {
            cur_mask |= LockMask;
        } else {
            cur_mask &= ~LockMask;
        }
        if (viddata->xkb.xkb_modifiers & viddata->xkb.numlock_mask) {
            cur_mask |= viddata->xkb.numlock_mask;
        } else {
            cur_mask &= ~viddata->xkb.numlock_mask;
        }
        if (viddata->xkb.xkb_modifiers & viddata->xkb.scrolllock_mask) {
            cur_mask |= viddata->xkb.scrolllock_mask;
        } else {
            cur_mask &= ~viddata->xkb.scrolllock_mask;
        }

        viddata->xkb.xkb_modifiers = cur_mask;
    } SDL_FALLTHROUGH;
    default:
        reconcile = true;
        break;
    }

    if (pressed) {
        viddata->xkb.sdl_modifiers |= mod;
    } else {
        viddata->xkb.sdl_modifiers &= ~mod;
    }

    if (allow_reconciliation) {
        if (reconcile) {
            X11_ReconcileModifiers(viddata);
        } else {
            SDL_SetModState(viddata->xkb.sdl_modifiers);
        }
    }
}

void X11_ReconcileKeyboardState(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;
    char keys[32];
    int keycode;
    const bool *keyboardState;

    X11_XQueryKeymap(display, keys);

    keyboardState = SDL_GetKeyboardState(0);
    for (keycode = 0; keycode < SDL_arraysize(videodata->key_layout); ++keycode) {
        SDL_Scancode scancode = videodata->key_layout[keycode];
        bool x11KeyPressed = (keys[keycode / 8] & (1 << (keycode % 8))) != 0;
        bool sdlKeyPressed = keyboardState[scancode];

        if (x11KeyPressed && !sdlKeyPressed) {
            // Only update modifier state for keys that are pressed in another application
            switch (SDL_GetKeyFromScancode(scancode, SDL_KMOD_NONE, false)) {
            case SDLK_LCTRL:
            case SDLK_RCTRL:
            case SDLK_LSHIFT:
            case SDLK_RSHIFT:
            case SDLK_LALT:
            case SDLK_RALT:
            case SDLK_LGUI:
            case SDLK_RGUI:
            case SDLK_MODE:
            case SDLK_LEVEL5_SHIFT:
                X11_HandleModifierKeys(videodata, scancode, true, false);
                SDL_SendKeyboardKeyIgnoreModifiers(0, SDL_GLOBAL_KEYBOARD_ID, keycode, scancode, true);
                break;
            default:
                break;
            }
        } else if (!x11KeyPressed && sdlKeyPressed) {
            X11_HandleModifierKeys(videodata, scancode, false, false);
            SDL_SendKeyboardKeyIgnoreModifiers(0, SDL_GLOBAL_KEYBOARD_ID, keycode, scancode, false);
        }
    }

    X11_UpdateSystemKeyModifiers(videodata);
    X11_ReconcileModifiers(videodata);
}

static void X11_DispatchFocusIn(SDL_VideoDevice *_this, SDL_WindowData *data)
{
#ifdef DEBUG_XEVENTS
    SDL_Log("window 0x%lx: Dispatching FocusIn", data->xwindow);
#endif
    SDL_SetKeyboardFocus(data->window);
    X11_ReconcileKeyboardState(_this);
#ifdef X_HAVE_UTF8_STRING
    if (data->ic) {
        X11_XSetICFocus(data->ic);
    }
#endif
    if (data->flashing_window) {
        X11_FlashWindow(_this, data->window, SDL_FLASH_CANCEL);
    }
}

static void X11_DispatchFocusOut(SDL_VideoDevice *_this, SDL_WindowData *data)
{
#ifdef DEBUG_XEVENTS
    SDL_Log("window 0x%lx: Dispatching FocusOut", data->xwindow);
#endif
    /* If another window has already processed a focus in, then don't try to
     * remove focus here.  Doing so will incorrectly remove focus from that
     * window, and the focus lost event for this window will have already
     * been dispatched anyway.
     */
    if (data->tracking_mouse_outside_window && data->window == SDL_GetMouseFocus()) {
        // If tracking the pointer and keyboard focus is lost, raise all buttons and relinquish mouse focus.
        SDL_SendMouseButton(0, data->window, SDL_GLOBAL_MOUSE_ID, SDL_BUTTON_LEFT, false);
        SDL_SendMouseButton(0, data->window, SDL_GLOBAL_MOUSE_ID, SDL_BUTTON_MIDDLE, false);
        SDL_SendMouseButton(0, data->window, SDL_GLOBAL_MOUSE_ID, SDL_BUTTON_RIGHT, false);
        SDL_SendMouseButton(0, data->window, SDL_GLOBAL_MOUSE_ID, SDL_BUTTON_X1, false);
        SDL_SendMouseButton(0, data->window, SDL_GLOBAL_MOUSE_ID, SDL_BUTTON_X2, false);
        SDL_SetMouseFocus(NULL);
    }
    if (data->window == SDL_GetKeyboardFocus()) {
        SDL_SetKeyboardFocus(NULL);
    }
#ifdef X_HAVE_UTF8_STRING
    if (data->ic) {
        X11_XUnsetICFocus(data->ic);
    }
#endif
}

static void X11_DispatchMapNotify(SDL_WindowData *data)
{
    SDL_Window *window = data->window;

    SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_SHOWN, 0, 0);
    data->was_shown = true;

    // This may be sent when restoring a minimized window.
    if (window->flags & SDL_WINDOW_MINIMIZED) {
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_EXPOSED, 0, 0);
    }

    if (window->flags & SDL_WINDOW_INPUT_FOCUS) {
        SDL_UpdateWindowGrab(window);
    }
}

static void X11_DispatchUnmapNotify(SDL_WindowData *data)
{
    SDL_Window *window = data->window;

    // This may be sent when minimizing a window.
    if (!window->is_hiding) {
        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MINIMIZED, 0, 0);
        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_OCCLUDED, 0, 0);
    } else {
        SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_HIDDEN, 0, 0);
    }
}

static void DispatchWindowMove(SDL_VideoDevice *_this, const SDL_WindowData *data, const SDL_Point *point)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_Window *window = data->window;
    Display *display = videodata->display;
    XEvent evt;

    // !!! FIXME: we need to regrab this if necessary when the drag is done.
    X11_XUngrabPointer(display, 0L);
    X11_XFlush(display);

    evt.xclient.type = ClientMessage;
    evt.xclient.window = data->xwindow;
    evt.xclient.message_type = videodata->atoms._NET_WM_MOVERESIZE;
    evt.xclient.format = 32;
    evt.xclient.data.l[0] = (size_t)window->x + point->x;
    evt.xclient.data.l[1] = (size_t)window->y + point->y;
    evt.xclient.data.l[2] = _NET_WM_MOVERESIZE_MOVE;
    evt.xclient.data.l[3] = Button1;
    evt.xclient.data.l[4] = 0;
    X11_XSendEvent(display, DefaultRootWindow(display), False, SubstructureRedirectMask | SubstructureNotifyMask, &evt);

    X11_XSync(display, 0);
}

static void ScheduleWindowMove(SDL_VideoDevice *_this, SDL_WindowData *data, const SDL_Point *point)
{
    data->pending_move = true;
    data->pending_move_point = *point;
}

static void InitiateWindowResize(SDL_VideoDevice *_this, const SDL_WindowData *data, const SDL_Point *point, int direction)
{
    SDL_VideoData *videodata = _this->internal;
    SDL_Window *window = data->window;
    Display *display = videodata->display;
    XEvent evt;

    if (direction < _NET_WM_MOVERESIZE_SIZE_TOPLEFT || direction > _NET_WM_MOVERESIZE_SIZE_LEFT) {
        return;
    }

    // !!! FIXME: we need to regrab this if necessary when the drag is done.
    X11_XUngrabPointer(display, 0L);
    X11_XFlush(display);

    evt.xclient.type = ClientMessage;
    evt.xclient.window = data->xwindow;
    evt.xclient.message_type = videodata->atoms._NET_WM_MOVERESIZE;
    evt.xclient.format = 32;
    evt.xclient.data.l[0] = (size_t)window->x + point->x;
    evt.xclient.data.l[1] = (size_t)window->y + point->y;
    evt.xclient.data.l[2] = direction;
    evt.xclient.data.l[3] = Button1;
    evt.xclient.data.l[4] = 0;
    X11_XSendEvent(display, DefaultRootWindow(display), False, SubstructureRedirectMask | SubstructureNotifyMask, &evt);

    X11_XSync(display, 0);
}

bool X11_ProcessHitTest(SDL_VideoDevice *_this, SDL_WindowData *data, const float x, const float y, bool force_new_result)
{
    SDL_Window *window = data->window;
    if (!window->hit_test) return false;
    const SDL_Point point = { (int)x, (int)y };
    SDL_HitTestResult rc = window->hit_test(window, &point, window->hit_test_data);
    if (!force_new_result && rc == data->hit_test_result) {
        return true;
    }
    X11_SetHitTestCursor(rc);
    data->hit_test_result = rc;
    return true;
}

bool X11_TriggerHitTestAction(SDL_VideoDevice *_this, SDL_WindowData *data, const float x, const float y)
{
    SDL_Window *window = data->window;

    if (window->hit_test) {
        const SDL_Point point = { (int)x, (int)y };
        static const int directions[] = {
            _NET_WM_MOVERESIZE_SIZE_TOPLEFT, _NET_WM_MOVERESIZE_SIZE_TOP,
            _NET_WM_MOVERESIZE_SIZE_TOPRIGHT, _NET_WM_MOVERESIZE_SIZE_RIGHT,
            _NET_WM_MOVERESIZE_SIZE_BOTTOMRIGHT, _NET_WM_MOVERESIZE_SIZE_BOTTOM,
            _NET_WM_MOVERESIZE_SIZE_BOTTOMLEFT, _NET_WM_MOVERESIZE_SIZE_LEFT
        };

        switch (data->hit_test_result) {
        case SDL_HITTEST_DRAGGABLE:
            /* Some window managers get in a bad state when a move event starts while input is transitioning
               to the SDL window. This can happen when clicking on a drag region of an unfocused window
               where the same mouse down event will trigger a drag event and a window activate. */
            if (data->window->flags & SDL_WINDOW_INPUT_FOCUS) {
                DispatchWindowMove(_this, data, &point);
            } else {
                ScheduleWindowMove(_this, data, &point);
            }
            return true;

        case SDL_HITTEST_RESIZE_TOPLEFT:
        case SDL_HITTEST_RESIZE_TOP:
        case SDL_HITTEST_RESIZE_TOPRIGHT:
        case SDL_HITTEST_RESIZE_RIGHT:
        case SDL_HITTEST_RESIZE_BOTTOMRIGHT:
        case SDL_HITTEST_RESIZE_BOTTOM:
        case SDL_HITTEST_RESIZE_BOTTOMLEFT:
        case SDL_HITTEST_RESIZE_LEFT:
            InitiateWindowResize(_this, data, &point, directions[data->hit_test_result - SDL_HITTEST_RESIZE_TOPLEFT]);
            return true;

        default:
            return false;
        }
    }

    return false;
}

static void X11_UpdateUserTime(SDL_WindowData *data, const unsigned long latest)
{
    if (latest && (latest != data->user_time)) {
        SDL_VideoData *videodata = data->videodata;
        Display *display = videodata->display;
        X11_XChangeProperty(display, data->xwindow, videodata->atoms._NET_WM_USER_TIME,
                            XA_CARDINAL, 32, PropModeReplace,
                            (const unsigned char *)&latest, 1);
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: updating _NET_WM_USER_TIME to %lu", data->xwindow, latest);
#endif
        data->user_time = latest;
    }
}

static void X11_HandleClipboardEvent(SDL_VideoDevice *_this, const XEvent *xevent)
{
    int i;
    SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;

    SDL_assert(videodata->clipboard_window != None);
    SDL_assert(xevent->xany.window == videodata->clipboard_window);

    switch (xevent->type) {
        // Copy the selection from our own CUTBUFFER to the requested property
    case SelectionRequest:
    {
        const XSelectionRequestEvent *req = &xevent->xselectionrequest;
        XEvent sevent;
        int mime_formats;
        unsigned char *seln_data;
        size_t seln_length = 0;
        Atom XA_TARGETS = videodata->atoms.TARGETS;
        SDLX11_ClipboardData *clipboard;

#ifdef DEBUG_XEVENTS
        char *atom_name;
        atom_name = X11_XGetAtomName(display, req->target);
        SDL_Log("window CLIPBOARD: SelectionRequest (requestor = 0x%lx, target = 0x%lx, mime_type = %s)",
               req->requestor, req->target, atom_name);
        if (atom_name) {
            X11_XFree(atom_name);
        }
#endif

        if (req->selection == XA_PRIMARY) {
            clipboard = &videodata->primary_selection;
        } else {
            clipboard = &videodata->clipboard;
        }

        SDL_zero(sevent);
        sevent.xany.type = SelectionNotify;
        sevent.xselection.selection = req->selection;
        sevent.xselection.target = None;
        sevent.xselection.property = None; // tell them no by default
        sevent.xselection.requestor = req->requestor;
        sevent.xselection.time = req->time;

        /* !!! FIXME: We were probably storing this on the root window
           because an SDL window might go away...? but we don't have to do
           this now (or ever, really). */

        if (req->target == XA_TARGETS) {
            Atom *supportedFormats;
            supportedFormats = SDL_malloc((clipboard->mime_count + 1) * sizeof(Atom));
            supportedFormats[0] = XA_TARGETS;
            mime_formats = 1;
            for (i = 0; i < clipboard->mime_count; ++i) {
                supportedFormats[mime_formats++] = X11_XInternAtom(display, clipboard->mime_types[i], False);
            }
            X11_XChangeProperty(display, req->requestor, req->property,
                                XA_ATOM, 32, PropModeReplace,
                                (unsigned char *)supportedFormats,
                                mime_formats);
            sevent.xselection.property = req->property;
            sevent.xselection.target = XA_TARGETS;
            SDL_free(supportedFormats);
        } else {
            if (clipboard->callback) {
                for (i = 0; i < clipboard->mime_count; ++i) {
                    const char *mime_type = clipboard->mime_types[i];
                    if (X11_XInternAtom(display, mime_type, False) != req->target) {
                        continue;
                    }

                    // FIXME: We don't support the X11 INCR protocol for large clipboards. Do we want that? - Yes, yes we do.
                    // This is a safe cast, XChangeProperty() doesn't take a const value, but it doesn't modify the data
                    seln_data = (unsigned char *)clipboard->callback(clipboard->userdata, mime_type, &seln_length);
                    if (seln_data) {
                        X11_XChangeProperty(display, req->requestor, req->property,
                                            req->target, 8, PropModeReplace,
                                            seln_data, seln_length);
                        sevent.xselection.property = req->property;
                        sevent.xselection.target = req->target;
                    }
                    break;
                }
            }
        }
        X11_XSendEvent(display, req->requestor, False, 0, &sevent);
        X11_XSync(display, False);
    } break;

    case SelectionNotify:
    {
        const XSelectionEvent *xsel = &xevent->xselection;
#ifdef DEBUG_XEVENTS
        const char *propName = xsel->property ? X11_XGetAtomName(display, xsel->property) : "None";
        const char *targetName = xsel->target ? X11_XGetAtomName(display, xsel->target) : "None";

        SDL_Log("window CLIPBOARD: SelectionNotify (requestor = 0x%lx, target = %s, property = %s)",
               xsel->requestor, targetName, propName);
#endif
        if (xsel->target == videodata->atoms.TARGETS && xsel->property == videodata->atoms.SDL_FORMATS) {
            /* the new mime formats are the SDL_FORMATS property as an array of Atoms */
            Atom atom = None;
            Atom *patom;
            unsigned char* data = NULL;
            int format_property = 0;
            unsigned long length = 0;
            unsigned long bytes_left = 0;
            int j;

            X11_XGetWindowProperty(display, GetWindow(_this), videodata->atoms.SDL_FORMATS, 0, 200,
                                            0, XA_ATOM, &atom, &format_property, &length, &bytes_left, &data);

            int allocationsize = (length + 1) * sizeof(char*);
            for (j = 0, patom = (Atom*)data; j < length; j++, patom++) {
                char *atomStr = X11_XGetAtomName(display, *patom);
                allocationsize += SDL_strlen(atomStr) + 1;
                X11_XFree(atomStr);
            }

            char **new_mime_types = SDL_AllocateTemporaryMemory(allocationsize);
            if (new_mime_types) {
                char *strPtr = (char *)(new_mime_types + length + 1);

                for (j = 0, patom = (Atom*)data; j < length; j++, patom++) {
                    char *atomStr = X11_XGetAtomName(display, *patom);
                    new_mime_types[j] = strPtr;
                    strPtr = stpcpy(strPtr, atomStr) + 1;
                    X11_XFree(atomStr);
                }
                new_mime_types[length] = NULL;

                SDL_SendClipboardUpdate(false, new_mime_types, length);
            }

            if (data) {
                X11_XFree(data);
            }
        }

        videodata->selection_waiting = false;
    } break;

    case SelectionClear:
    {
        Atom XA_CLIPBOARD = videodata->atoms.CLIPBOARD;
        SDLX11_ClipboardData *clipboard = NULL;

#ifdef DEBUG_XEVENTS
        SDL_Log("window CLIPBOARD: SelectionClear (requestor = 0x%lx, target = 0x%lx)",
               xevent->xselection.requestor, xevent->xselection.target);
#endif

        if (xevent->xselectionclear.selection == XA_PRIMARY) {
            clipboard = &videodata->primary_selection;
        } else if (XA_CLIPBOARD != None && xevent->xselectionclear.selection == XA_CLIPBOARD) {
            clipboard = &videodata->clipboard;
        }
        if (clipboard && clipboard->callback) {
            if (clipboard->sequence) {
                SDL_CancelClipboardData(clipboard->sequence);
            } else {
                SDL_free(clipboard->userdata);
            }
            SDL_zerop(clipboard);
        }
    } break;

    case PropertyNotify:
    {
        char *name_of_atom = X11_XGetAtomName(display, xevent->xproperty.atom);

        if (SDL_strncmp(name_of_atom, "SDL_SELECTION", sizeof("SDL_SELECTION") - 1) == 0 && xevent->xproperty.state == PropertyNewValue) {
            videodata->selection_incr_waiting = false;
        }

        if (name_of_atom) {
            X11_XFree(name_of_atom);
        }
    } break;
    }
}

static void X11_HandleSettingsEvent(SDL_VideoDevice *_this, const XEvent *xevent)
{
    SDL_VideoData *videodata = _this->internal;

    SDL_assert(videodata->xsettings_window != None);
    SDL_assert(xevent->xany.window == videodata->xsettings_window);

    X11_HandleXsettings(_this, xevent);
}

static Bool isMapNotify(Display *display, XEvent *ev, XPointer arg)
{
    XUnmapEvent *unmap;

    unmap = (XUnmapEvent *)arg;

    return ev->type == MapNotify &&
           ev->xmap.window == unmap->window &&
           ev->xmap.serial == unmap->serial;
}

static Bool isReparentNotify(Display *display, XEvent *ev, XPointer arg)
{
    XUnmapEvent *unmap;

    unmap = (XUnmapEvent *)arg;

    return ev->type == ReparentNotify &&
           ev->xreparent.window == unmap->window &&
           ev->xreparent.serial == unmap->serial;
}

static bool IsHighLatin1(const char *string, int length)
{
    while (length-- > 0) {
        Uint8 ch = (Uint8)*string;
        if (ch >= 0x80) {
            return true;
        }
        ++string;
    }
    return false;
}

static int XLookupStringAsUTF8(XKeyEvent *event_struct, char *buffer_return, int bytes_buffer, KeySym *keysym_return, XComposeStatus *status_in_out)
{
    int result = X11_XLookupString(event_struct, buffer_return, bytes_buffer, keysym_return, status_in_out);
    if (IsHighLatin1(buffer_return, result)) {
        char *utf8_text = SDL_iconv_string("UTF-8", "ISO-8859-1", buffer_return, result + 1);
        if (utf8_text) {
            SDL_strlcpy(buffer_return, utf8_text, bytes_buffer);
            SDL_free(utf8_text);
            return SDL_strlen(buffer_return);
        } else {
            return 0;
        }
    }
    return result;
}

SDL_WindowData *X11_FindWindow(SDL_VideoDevice *_this, Window window)
{
    const SDL_VideoData *videodata = _this->internal;
    int i;

    if (videodata && videodata->windowlist) {
        for (i = 0; i < videodata->numwindows; ++i) {
            if ((videodata->windowlist[i] != NULL) &&
                (videodata->windowlist[i]->xwindow == window)) {
                return videodata->windowlist[i];
            }
        }
    }
    return NULL;
}

Uint64 X11_GetEventTimestamp(unsigned long time)
{
    // FIXME: Get the event time in the SDL tick time base
    return SDL_GetTicksNS();
}

void X11_HandleKeyEvent(SDL_VideoDevice *_this, SDL_WindowData *windowdata, SDL_KeyboardID keyboardID, XEvent *xevent)
{
    SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;
    KeyCode keycode = xevent->xkey.keycode;
    KeySym keysym = NoSymbol;
    int text_length = 0;
    char text[64];
    Status status = 0;
    bool handled_by_ime = false;
    bool pressed = (xevent->type == KeyPress);
    SDL_Scancode scancode = videodata->key_layout[keycode];
    Uint64 timestamp = X11_GetEventTimestamp(xevent->xkey.time);

#ifdef DEBUG_XEVENTS
    SDL_Log("window 0x%lx %s (X11 keycode = 0x%X)", xevent->xany.window, (xevent->type == KeyPress ? "KeyPress" : "KeyRelease"), xevent->xkey.keycode);
#endif
#ifdef DEBUG_SCANCODES
    if (scancode == SDL_SCANCODE_UNKNOWN && keycode) {
        int min_keycode, max_keycode;
        X11_XDisplayKeycodes(display, &min_keycode, &max_keycode);
        keysym = X11_KeyCodeToSym(_this, keycode, xevent->xkey.state >> 13);
        SDL_Log("The key you just pressed is not recognized by SDL. To help get this fixed, please report this to the SDL forums/mailing list <https://discourse.libsdl.org/> X11 KeyCode %d (%d), X11 KeySym 0x%lX (%s).",
                keycode, keycode - min_keycode, keysym,
                X11_XKeysymToString(keysym));
    }
#endif // DEBUG SCANCODES

    text[0] = '\0';
    videodata->xkb.xkb_modifiers = xevent->xkey.state;

    if (SDL_TextInputActive(windowdata->window)) {
        // filter events catches XIM events and sends them to the correct handler
        if (X11_XFilterEvent(xevent, None)) {
#ifdef DEBUG_XEVENTS
            SDL_Log("Filtered event type = %d display = %p window = 0x%lx",
                   xevent->type, xevent->xany.display, xevent->xany.window);
#endif
            handled_by_ime = true;
        }

        if (!handled_by_ime) {
#ifdef X_HAVE_UTF8_STRING
            if (windowdata->ic && xevent->type == KeyPress) {
                text_length = X11_Xutf8LookupString(windowdata->ic, &xevent->xkey, text, sizeof(text) - 1,
                                      &keysym, &status);
            } else {
                text_length = XLookupStringAsUTF8(&xevent->xkey, text, sizeof(text) - 1, &keysym, NULL);
            }
#else
            text_length = XLookupStringAsUTF8(&xevent->xkey, text, sizeof(text) - 1, &keysym, NULL);
#endif
        }
    }

    if (pressed) {
        X11_HandleModifierKeys(videodata, scancode, true, true);
        SDL_SendKeyboardKeyIgnoreModifiers(timestamp, keyboardID, keycode, scancode, true);

        // Synthesize a text event if the IME didn't consume a printable character
        if (*text && !(SDL_GetModState() & (SDL_KMOD_CTRL | SDL_KMOD_ALT))) {
            text[text_length] = '\0';
            X11_ClearComposition(windowdata);
            SDL_SendKeyboardText(text);
        }

        X11_UpdateUserTime(windowdata, xevent->xkey.time);
    } else {
        if (X11_KeyRepeat(display, xevent)) {
            // We're about to get a repeated key down, ignore the key up
            return;
        }

        X11_HandleModifierKeys(videodata, scancode, false, true);
        SDL_SendKeyboardKeyIgnoreModifiers(timestamp, keyboardID, keycode, scancode, false);
    }
}

void X11_HandleButtonPress(SDL_VideoDevice *_this, SDL_WindowData *windowdata, SDL_MouseID mouseID, int button, float x, float y, unsigned long time)
{
    SDL_Window *window = windowdata->window;
    const SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;
    int xticks = 0, yticks = 0;
    Uint64 timestamp = X11_GetEventTimestamp(time);

#ifdef DEBUG_XEVENTS
    SDL_Log("window 0x%lx: ButtonPress (X11 button = %d)", windowdata->xwindow, button);
#endif

    SDL_Mouse *mouse = SDL_GetMouse();
    if (!mouse->relative_mode && (x != mouse->x || y != mouse->y)) {
        X11_ProcessHitTest(_this, windowdata, x, y, false);
        SDL_SendMouseMotion(timestamp, window, mouseID, false, x, y);
    }

    if (X11_IsWheelEvent(display, button, &xticks, &yticks)) {
        SDL_SendMouseWheel(timestamp, window, mouseID, (float)-xticks, (float)yticks, SDL_MOUSEWHEEL_NORMAL);
    } else {
        bool ignore_click = false;
        if (button > 7) {
            /* X button values 4-7 are used for scrolling, so X1 is 8, X2 is 9, ...
               => subtract (8-SDL_BUTTON_X1) to get value SDL expects */
            button -= (8 - SDL_BUTTON_X1);
        }
        if (button == Button1) {
            if (X11_TriggerHitTestAction(_this, windowdata, x, y)) {
                SDL_SendWindowEvent(window, SDL_EVENT_WINDOW_HIT_TEST, 0, 0);
                return; // don't pass this event on to app.
            }
        }
        if (windowdata->last_focus_event_time) {
            const int X11_FOCUS_CLICK_TIMEOUT = 10;
            if (SDL_GetTicks() < (windowdata->last_focus_event_time + X11_FOCUS_CLICK_TIMEOUT)) {
                ignore_click = !SDL_GetHintBoolean(SDL_HINT_MOUSE_FOCUS_CLICKTHROUGH, false);
            }
            windowdata->last_focus_event_time = 0;
        }
        if (!ignore_click) {
            SDL_SendMouseButton(timestamp, window, mouseID, button, true);
        }
    }
    X11_UpdateUserTime(windowdata, time);
}

void X11_HandleButtonRelease(SDL_VideoDevice *_this, SDL_WindowData *windowdata, SDL_MouseID mouseID, int button, unsigned long time)
{
    SDL_Window *window = windowdata->window;
    const SDL_VideoData *videodata = _this->internal;
    Display *display = videodata->display;
    // The X server sends a Release event for each Press for wheels. Ignore them.
    int xticks = 0, yticks = 0;
    Uint64 timestamp = X11_GetEventTimestamp(time);

#ifdef DEBUG_XEVENTS
    SDL_Log("window 0x%lx: ButtonRelease (X11 button = %d)", windowdata->xwindow, button);
#endif
    if (!X11_IsWheelEvent(display, button, &xticks, &yticks)) {
        if (button > 7) {
            // see explanation at case ButtonPress
            button -= (8 - SDL_BUTTON_X1);
        }

        /* If the mouse is captured and all buttons are now released, clear the capture
         * flag so the focus will be cleared if the mouse is outside the window.
         */
        if ((window->flags & SDL_WINDOW_MOUSE_CAPTURE)  &&
            !(SDL_GetMouseState(NULL, NULL) & ~SDL_BUTTON_MASK(button))) {
            window->flags &= ~SDL_WINDOW_MOUSE_CAPTURE;
            windowdata->tracking_mouse_outside_window = false;
        }

        SDL_SendMouseButton(timestamp, window, mouseID, button, false);
    }
}

void X11_GetBorderValues(SDL_WindowData *data)
{
    SDL_VideoData *videodata = data->videodata;
    Display *display = videodata->display;

    Atom type;
    int format;
    unsigned long nitems, bytes_after;
    unsigned char *property;

    // Some compositors will send extents even when the border hint is turned off. Ignore them in this case.
    if (!(data->window->flags & SDL_WINDOW_BORDERLESS)) {
        if (X11_XGetWindowProperty(display, data->xwindow, videodata->atoms._NET_FRAME_EXTENTS, 0, 16, 0, XA_CARDINAL, &type, &format, &nitems, &bytes_after, &property) == Success) {
            if (type != None && nitems == 4) {
                data->border_left = (int)((long *)property)[0];
                data->border_right = (int)((long *)property)[1];
                data->border_top = (int)((long *)property)[2];
                data->border_bottom = (int)((long *)property)[3];
            }
            X11_XFree(property);

#ifdef DEBUG_XEVENTS
            SDL_Log("New _NET_FRAME_EXTENTS: left=%d right=%d, top=%d, bottom=%d", data->border_left, data->border_right, data->border_top, data->border_bottom);
#endif
        }
    } else {
        data->border_left = data->border_top = data->border_right = data->border_bottom = 0;
    }
}

void X11_EmitConfigureNotifyEvents(SDL_WindowData *data, XConfigureEvent *xevent)
{
    if (xevent->x != data->last_xconfigure.x ||
        xevent->y != data->last_xconfigure.y) {
        if (!data->size_move_event_flags) {
            SDL_Window *w;
            int x = xevent->x;
            int y = xevent->y;

            data->pending_operation &= ~X11_PENDING_OP_MOVE;
            SDL_GlobalToRelativeForWindow(data->window, x, y, &x, &y);
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MOVED, x, y);

            for (w = data->window->first_child; w; w = w->next_sibling) {
                // Don't update hidden child popup windows, their relative position doesn't change
                if (SDL_WINDOW_IS_POPUP(w) && !(w->flags & SDL_WINDOW_HIDDEN)) {
                    X11_UpdateWindowPosition(w, true);
                }
            }
        }
    }

    if (xevent->width != data->last_xconfigure.width ||
        xevent->height != data->last_xconfigure.height) {
        if (!data->size_move_event_flags) {
            data->pending_operation &= ~X11_PENDING_OP_RESIZE;
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESIZED,
                                xevent->width,
                                xevent->height);
        }
    }

    SDL_copyp(&data->last_xconfigure, xevent);
}

static void X11_DispatchEvent(SDL_VideoDevice *_this, XEvent *xevent)
{
    SDL_VideoData *videodata = _this->internal;
    Display *display;
    SDL_WindowData *data;
    XClientMessageEvent m;
    int i;

    SDL_assert(videodata != NULL);
    display = videodata->display;

    // filter events catches XIM events and sends them to the correct handler
    // Key press/release events are filtered in X11_HandleKeyEvent()
    if (xevent->type != KeyPress && xevent->type != KeyRelease) {
        if (X11_XFilterEvent(xevent, None)) {
#ifdef DEBUG_XEVENTS
            SDL_Log("Filtered event type = %d display = %p window = 0x%lx",
                   xevent->type, xevent->xany.display, xevent->xany.window);
#endif
            return;
        }
    }

#ifdef SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS
    if (xevent->type == GenericEvent) {
        X11_HandleGenericEvent(_this, xevent);
        return;
    }
#endif

    // Calling the event hook for generic events happens in X11_HandleGenericEvent(), where the event data is available
    if (g_X11EventHook) {
        if (!g_X11EventHook(g_X11EventHookData, xevent)) {
            return;
        }
    }

#ifdef SDL_VIDEO_DRIVER_X11_XRANDR
    if (videodata->xrandr_event_base && (xevent->type == (videodata->xrandr_event_base + RRNotify))) {
        X11_HandleXRandREvent(_this, xevent);
    }
#endif

#ifdef DEBUG_XEVENTS
    SDL_Log("X11 event type = %d display = %p window = 0x%lx",
           xevent->type, xevent->xany.display, xevent->xany.window);
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
    if (SDL_X11_HAVE_XFIXES &&
        xevent->type == X11_GetXFixesSelectionNotifyEvent()) {
        XFixesSelectionNotifyEvent *ev = (XFixesSelectionNotifyEvent *)xevent;

#ifdef DEBUG_XEVENTS
        SDL_Log("window CLIPBOARD: XFixesSelectionNotify (selection = %s)",
               X11_XGetAtomName(display, ev->selection));
#endif

        if (ev->subtype == XFixesSetSelectionOwnerNotify)
        {
            if (ev->selection != videodata->atoms.CLIPBOARD)
                return;

            if (X11_XGetSelectionOwner(display, ev->selection) == videodata->clipboard_window)
                return;

            /* when here we're notified that the clipboard had an external change, we request the
             * available mime types by asking for a conversion to the TARGETS format. We should get a
             * SelectionNotify event later, and when treating these results, we will push a ClipboardUpdated
             * event
             */

            X11_XConvertSelection(display, videodata->atoms.CLIPBOARD, videodata->atoms.TARGETS,
                    videodata->atoms.SDL_FORMATS, GetWindow(_this), CurrentTime);
        }

        return;
    }
#endif // SDL_VIDEO_DRIVER_X11_XFIXES

    if ((videodata->clipboard_window != None) &&
        (videodata->clipboard_window == xevent->xany.window)) {
        X11_HandleClipboardEvent(_this, xevent);
        return;
    }

    if ((videodata->xsettings_window != None) &&
        (videodata->xsettings_window == xevent->xany.window)) {
        X11_HandleSettingsEvent(_this, xevent);
        return;
    }

    data = X11_FindWindow(_this, xevent->xany.window);

    if (!data) {
        // The window for KeymapNotify, etc events is 0
        if (xevent->type == KeymapNotify) {
#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: KeymapNotify!", xevent->xany.window);
#endif
            if (SDL_GetKeyboardFocus() != NULL) {
#ifdef SDL_VIDEO_DRIVER_X11_HAS_XKBLOOKUPKEYSYM
                if (videodata->xkb.desc_ptr) {
                    XkbStateRec state;
                    if (X11_XkbGetState(videodata->display, XkbUseCoreKbd, &state) == Success) {
                        if (state.group != videodata->xkb.current_group) {
                            // Only rebuild the keymap if the layout has changed.
                            videodata->xkb.current_group = state.group;
                            X11_UpdateKeymap(_this, true);
                        }
                    }
                }
#endif
                X11_ReconcileKeyboardState(_this);
            }
        } else if (xevent->type == MappingNotify) {
            // Has the keyboard layout changed?
            const int request = xevent->xmapping.request;

#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: MappingNotify!", xevent->xany.window);
#endif
            if ((request == MappingKeyboard) || (request == MappingModifier)) {
                X11_XRefreshKeyboardMapping(&xevent->xmapping);
            }

            X11_UpdateKeymap(_this, true);
        } else if (xevent->type == PropertyNotify && videodata && videodata->windowlist) {
            char *name_of_atom = X11_XGetAtomName(display, xevent->xproperty.atom);

            if (SDL_strncmp(name_of_atom, "_ICC_PROFILE", sizeof("_ICC_PROFILE") - 1) == 0) {
                XWindowAttributes attrib;
                int screennum;
                for (i = 0; i < videodata->numwindows; ++i) {
                    if (videodata->windowlist[i] != NULL) {
                        data = videodata->windowlist[i];
                        X11_XGetWindowAttributes(display, data->xwindow, &attrib);
                        screennum = X11_XScreenNumberOfScreen(attrib.screen);
                        if (screennum == 0 && SDL_strcmp(name_of_atom, "_ICC_PROFILE") == 0) {
                            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_ICCPROF_CHANGED, 0, 0);
                        } else if (SDL_strncmp(name_of_atom, "_ICC_PROFILE_", sizeof("_ICC_PROFILE_") - 1) == 0 && SDL_strlen(name_of_atom) > sizeof("_ICC_PROFILE_") - 1) {
                            int iccscreennum = SDL_atoi(&name_of_atom[sizeof("_ICC_PROFILE_") - 1]);

                            if (screennum == iccscreennum) {
                                SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_ICCPROF_CHANGED, 0, 0);
                            }
                        }
                    }
                }
            }

            if (name_of_atom) {
                X11_XFree(name_of_atom);
            }
        }
        return;
    }

    switch (xevent->type) {

        // Gaining mouse coverage?
    case EnterNotify:
    {
        SDL_Mouse *mouse = SDL_GetMouse();
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: EnterNotify! (%d,%d,%d)", xevent->xany.window,
               xevent->xcrossing.x,
               xevent->xcrossing.y,
               xevent->xcrossing.mode);
        if (xevent->xcrossing.mode == NotifyGrab) {
            SDL_Log("Mode: NotifyGrab");
        }
        if (xevent->xcrossing.mode == NotifyUngrab) {
            SDL_Log("Mode: NotifyUngrab");
        }
#endif
        data->tracking_mouse_outside_window = false;

        SDL_SetMouseFocus(data->window);

        mouse->last_x = xevent->xcrossing.x;
        mouse->last_y = xevent->xcrossing.y;

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
        {
            // Only create the barriers if we have input focus
            SDL_WindowData *windowdata = data->window->internal;
            if ((data->pointer_barrier_active == true) && windowdata->window->flags & SDL_WINDOW_INPUT_FOCUS) {
                X11_ConfineCursorWithFlags(_this, windowdata->window, &windowdata->barrier_rect, X11_BARRIER_HANDLED_BY_EVENT);
            }
        }
#endif

        if (!mouse->relative_mode) {
            SDL_SendMouseMotion(0, data->window, SDL_GLOBAL_MOUSE_ID, false, (float)xevent->xcrossing.x, (float)xevent->xcrossing.y);
        }

        // We ungrab in LeaveNotify, so we may need to grab again here
        SDL_UpdateWindowGrab(data->window);

        X11_ProcessHitTest(_this, data, mouse->last_x, mouse->last_y, true);
    } break;
        // Losing mouse coverage?
    case LeaveNotify:
    {
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: LeaveNotify! (%d,%d,%d)", xevent->xany.window,
               xevent->xcrossing.x,
               xevent->xcrossing.y,
               xevent->xcrossing.mode);
        if (xevent->xcrossing.mode == NotifyGrab) {
            SDL_Log("Mode: NotifyGrab");
        }
        if (xevent->xcrossing.mode == NotifyUngrab) {
            SDL_Log("Mode: NotifyUngrab");
        }
#endif
        if (!SDL_GetMouse()->relative_mode) {
            SDL_SendMouseMotion(0, data->window, SDL_GLOBAL_MOUSE_ID, false, (float)xevent->xcrossing.x, (float)xevent->xcrossing.y);
        }

        if (xevent->xcrossing.mode != NotifyGrab &&
            xevent->xcrossing.mode != NotifyUngrab &&
            xevent->xcrossing.detail != NotifyInferior) {
            if (!(data->window->flags & SDL_WINDOW_MOUSE_CAPTURE)) {
                /* In order for interaction with the window decorations and menu to work properly
                   on Mutter, we need to ungrab the keyboard when the mouse leaves. */
                if (!(data->window->flags & SDL_WINDOW_FULLSCREEN)) {
                    X11_SetWindowKeyboardGrab(_this, data->window, false);
                }

                SDL_SetMouseFocus(NULL);
            } else {
                data->tracking_mouse_outside_window = true;
            }
        }
    } break;

        // Gaining input focus?
    case FocusIn:
    {
        if (xevent->xfocus.mode == NotifyGrab || xevent->xfocus.mode == NotifyUngrab) {
            // Someone is handling a global hotkey, ignore it
#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: FocusIn (NotifyGrab/NotifyUngrab, ignoring)", xevent->xany.window);
#endif
            break;
        }

        if (xevent->xfocus.detail == NotifyInferior || xevent->xfocus.detail == NotifyPointer) {
#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: FocusIn (NotifyInferior/NotifyPointer, ignoring)", xevent->xany.window);
#endif
            break;
        }
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: FocusIn!", xevent->xany.window);
#endif
        if (!videodata->last_mode_change_deadline) /* no recent mode changes */ {
            data->pending_focus = PENDING_FOCUS_NONE;
            data->pending_focus_time = 0;
            X11_DispatchFocusIn(_this, data);
        } else {
            data->pending_focus = PENDING_FOCUS_IN;
            data->pending_focus_time = SDL_GetTicks() + PENDING_FOCUS_TIME;
        }
        data->last_focus_event_time = SDL_GetTicks();
    } break;

        // Losing input focus?
    case FocusOut:
    {
        if (xevent->xfocus.mode == NotifyGrab || xevent->xfocus.mode == NotifyUngrab) {
            // Someone is handling a global hotkey, ignore it
#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: FocusOut (NotifyGrab/NotifyUngrab, ignoring)", xevent->xany.window);
#endif
            break;
        }
        if (xevent->xfocus.detail == NotifyInferior || xevent->xfocus.detail == NotifyPointer) {
            /* We still have focus if a child gets focus. We also don't
               care about the position of the pointer when the keyboard
               focus changed. */
#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: FocusOut (NotifyInferior/NotifyPointer, ignoring)", xevent->xany.window);
#endif
            break;
        }
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: FocusOut!", xevent->xany.window);
#endif
        if (!videodata->last_mode_change_deadline) /* no recent mode changes */ {
            data->pending_focus = PENDING_FOCUS_NONE;
            data->pending_focus_time = 0;
            X11_DispatchFocusOut(_this, data);
        } else {
            data->pending_focus = PENDING_FOCUS_OUT;
            data->pending_focus_time = SDL_GetTicks() + PENDING_FOCUS_TIME;
        }

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
        // Disable confinement if it is activated.
        if (data->pointer_barrier_active == true) {
            X11_ConfineCursorWithFlags(_this, data->window, NULL, X11_BARRIER_HANDLED_BY_EVENT);
        }
#endif // SDL_VIDEO_DRIVER_X11_XFIXES
    } break;


        // Have we been iconified?
    case UnmapNotify:
    {
        XEvent ev;

#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: UnmapNotify!", xevent->xany.window);
#endif

        if (X11_XCheckIfEvent(display, &ev, &isReparentNotify, (XPointer)&xevent->xunmap)) {
            X11_XCheckIfEvent(display, &ev, &isMapNotify, (XPointer)&xevent->xunmap);
        } else {
            X11_DispatchUnmapNotify(data);
        }

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
        // Disable confinement if the window gets hidden.
        if (data->pointer_barrier_active == true) {
            X11_ConfineCursorWithFlags(_this, data->window, NULL, X11_BARRIER_HANDLED_BY_EVENT);
        }
#endif // SDL_VIDEO_DRIVER_X11_XFIXES
    } break;

        // Have we been restored?
    case MapNotify:
    {
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: MapNotify!", xevent->xany.window);
#endif
        X11_DispatchMapNotify(data);

#ifdef SDL_VIDEO_DRIVER_X11_XFIXES
        // Enable confinement if it was activated.
        if (data->pointer_barrier_active == true) {
            X11_ConfineCursorWithFlags(_this, data->window, &data->barrier_rect, X11_BARRIER_HANDLED_BY_EVENT);
        }
#endif // SDL_VIDEO_DRIVER_X11_XFIXES
    } break;

        // Have we been resized or moved?
    case ConfigureNotify:
    {
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: ConfigureNotify! (position: %d,%d, size: %dx%d)", xevent->xany.window,
               xevent->xconfigure.x, xevent->xconfigure.y,
               xevent->xconfigure.width, xevent->xconfigure.height);
#endif
        // Real configure notify events are relative to the parent, synthetic events are absolute.
        if (!xevent->xconfigure.send_event) {
            unsigned int NumChildren;
            Window ChildReturn, Root, Parent;
            Window *Children;
            // Translate these coordinates back to relative to root
            X11_XQueryTree(data->videodata->display, xevent->xconfigure.window, &Root, &Parent, &Children, &NumChildren);
            X11_XTranslateCoordinates(xevent->xconfigure.display,
                                      Parent, DefaultRootWindow(xevent->xconfigure.display),
                                      xevent->xconfigure.x, xevent->xconfigure.y,
                                      &xevent->xconfigure.x, &xevent->xconfigure.y,
                                      &ChildReturn);
        }

        /* Xfce sends ConfigureNotify before PropertyNotify when toggling fullscreen and maximized, which
         * is backwards from every other window manager, as well as what is expected by SDL and its clients.
         * Defer emitting the size/move events until the corresponding PropertyNotify arrives.
         */
        const Uint32 changed = X11_GetNetWMState(_this, data->window, xevent->xproperty.window) ^ data->window->flags;
        if (changed & (SDL_WINDOW_FULLSCREEN | SDL_WINDOW_MAXIMIZED)) {
            SDL_copyp(&data->pending_xconfigure, &xevent->xconfigure);
            data->emit_size_move_after_property_notify = true;
        }

        if (!data->emit_size_move_after_property_notify) {
            X11_EmitConfigureNotifyEvents(data, &xevent->xconfigure);
        }

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
        X11_HandleConfigure(data->window, &xevent->xconfigure);
#endif /* SDL_VIDEO_DRIVER_X11_XSYNC */
    } break;

        // Have we been requested to quit (or another client message?)
    case ClientMessage:
    {
        static int xdnd_version = 0;

        if (xevent->xclient.message_type == videodata->atoms.XdndEnter) {

            bool use_list = xevent->xclient.data.l[1] & 1;
            data->xdnd_source = xevent->xclient.data.l[0];
            xdnd_version = (xevent->xclient.data.l[1] >> 24);
#ifdef DEBUG_XEVENTS
            SDL_Log("XID of source window : 0x%lx", data->xdnd_source);
            SDL_Log("Protocol version to use : %d", xdnd_version);
            SDL_Log("More then 3 data types : %d", (int)use_list);
#endif

            if (use_list) {
                // fetch conversion targets
                SDL_x11Prop p;
                X11_ReadProperty(&p, display, data->xdnd_source, videodata->atoms.XdndTypeList);
                // pick one
                data->xdnd_req = X11_PickTarget(display, (Atom *)p.data, p.count);
                X11_XFree(p.data);
            } else {
                // pick from list of three
                data->xdnd_req = X11_PickTargetFromAtoms(display, xevent->xclient.data.l[2], xevent->xclient.data.l[3], xevent->xclient.data.l[4]);
            }
        } else if (xevent->xclient.message_type == videodata->atoms.XdndLeave) {
#ifdef DEBUG_XEVENTS
            SDL_Log("XID of source window : 0x%lx", xevent->xclient.data.l[0]);
#endif
            SDL_SendDropComplete(data->window);
        } else if (xevent->xclient.message_type == videodata->atoms.XdndPosition) {

#ifdef DEBUG_XEVENTS
            Atom act = videodata->atoms.XdndActionCopy;
            if (xdnd_version >= 2) {
                act = xevent->xclient.data.l[4];
            }
            SDL_Log("Action requested by user is : %s", X11_XGetAtomName(display, act));
#endif
            {
                // Drag and Drop position
                int root_x, root_y, window_x, window_y;
                Window ChildReturn;
                root_x = xevent->xclient.data.l[2] >> 16;
                root_y = xevent->xclient.data.l[2] & 0xffff;
                // Translate from root to current window position
                X11_XTranslateCoordinates(display, DefaultRootWindow(display), data->xwindow,
                                          root_x, root_y, &window_x, &window_y, &ChildReturn);

                SDL_SendDropPosition(data->window, (float)window_x, (float)window_y);
            }

            // reply with status
            SDL_memset(&m, 0, sizeof(XClientMessageEvent));
            m.type = ClientMessage;
            m.display = xevent->xclient.display;
            m.window = xevent->xclient.data.l[0];
            m.message_type = videodata->atoms.XdndStatus;
            m.format = 32;
            m.data.l[0] = data->xwindow;
            m.data.l[1] = (data->xdnd_req != None);
            m.data.l[2] = 0; // specify an empty rectangle
            m.data.l[3] = 0;
            m.data.l[4] = videodata->atoms.XdndActionCopy; // we only accept copying anyway

            X11_XSendEvent(display, xevent->xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
            X11_XFlush(display);
        } else if (xevent->xclient.message_type == videodata->atoms.XdndDrop) {
            if (data->xdnd_req == None) {
                // say again - not interested!
                SDL_memset(&m, 0, sizeof(XClientMessageEvent));
                m.type = ClientMessage;
                m.display = xevent->xclient.display;
                m.window = xevent->xclient.data.l[0];
                m.message_type = videodata->atoms.XdndFinished;
                m.format = 32;
                m.data.l[0] = data->xwindow;
                m.data.l[1] = 0;
                m.data.l[2] = None; // fail!
                X11_XSendEvent(display, xevent->xclient.data.l[0], False, NoEventMask, (XEvent *)&m);
            } else {
                // convert
                if (xdnd_version >= 1) {
                    X11_XConvertSelection(display, videodata->atoms.XdndSelection, data->xdnd_req, videodata->atoms.PRIMARY, data->xwindow, xevent->xclient.data.l[2]);
                } else {
                    X11_XConvertSelection(display, videodata->atoms.XdndSelection, data->xdnd_req, videodata->atoms.PRIMARY, data->xwindow, CurrentTime);
                }
            }
        } else if ((xevent->xclient.message_type == videodata->atoms.WM_PROTOCOLS) &&
                   (xevent->xclient.format == 32) &&
                   (xevent->xclient.data.l[0] == videodata->atoms._NET_WM_PING)) {
            Window root = DefaultRootWindow(display);

#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: _NET_WM_PING", xevent->xany.window);
#endif
            xevent->xclient.window = root;
            X11_XSendEvent(display, root, False, SubstructureRedirectMask | SubstructureNotifyMask, xevent);
            break;
        }

        else if ((xevent->xclient.message_type == videodata->atoms.WM_PROTOCOLS) &&
                 (xevent->xclient.format == 32) &&
                 (xevent->xclient.data.l[0] == videodata->atoms.WM_DELETE_WINDOW)) {

#ifdef DEBUG_XEVENTS
            SDL_Log("window 0x%lx: WM_DELETE_WINDOW", xevent->xany.window);
#endif
            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_CLOSE_REQUESTED, 0, 0);
            break;
        } else if ((xevent->xclient.message_type == videodata->atoms.WM_PROTOCOLS) &&
                   (xevent->xclient.format == 32) &&
                   (xevent->xclient.data.l[0] == videodata->atoms._NET_WM_SYNC_REQUEST)) {

#ifdef DEBUG_XEVENTS
            printf("window %p: _NET_WM_SYNC_REQUEST\n", data);
#endif
#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
            X11_HandleSyncRequest(data->window, &xevent->xclient);
#endif /* SDL_VIDEO_DRIVER_X11_XSYNC */
            break;
        }
    } break;

        // Do we need to refresh ourselves?
    case Expose:
    {
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: Expose (count = %d)", xevent->xany.window, xevent->xexpose.count);
#endif
        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_EXPOSED, 0, 0);
    } break;

    /* Use XInput2 instead of the xevents API if possible, for:
       - KeyPress
       - KeyRelease
       - MotionNotify
       - ButtonPress
       - ButtonRelease
       XInput2 has more precise information, e.g., to distinguish different input devices. */
    case KeyPress:
    case KeyRelease:
    {
        if (data->xinput2_keyboard_enabled) {
            // This input is being handled by XInput2
            break;
        }

        X11_HandleKeyEvent(_this, data, SDL_GLOBAL_KEYBOARD_ID, xevent);
    } break;

    case MotionNotify:
    {
        if (data->xinput2_mouse_enabled && !data->mouse_grabbed) {
            // This input is being handled by XInput2
            break;
        }

        SDL_Mouse *mouse = SDL_GetMouse();
        if (!mouse->relative_mode) {
#ifdef DEBUG_MOTION
            SDL_Log("window 0x%lx: X11 motion: %d,%d", xevent->xany.window, xevent->xmotion.x, xevent->xmotion.y);
#endif

            X11_ProcessHitTest(_this, data, (float)xevent->xmotion.x, (float)xevent->xmotion.y, false);
            SDL_SendMouseMotion(0, data->window, SDL_GLOBAL_MOUSE_ID, false, (float)xevent->xmotion.x, (float)xevent->xmotion.y);
        }
    } break;

    case ButtonPress:
    {
        if (data->xinput2_mouse_enabled) {
            // This input is being handled by XInput2
            break;
        }

        X11_HandleButtonPress(_this, data, SDL_GLOBAL_MOUSE_ID, xevent->xbutton.button,
                              xevent->xbutton.x, xevent->xbutton.y, xevent->xbutton.time);
    } break;

    case ButtonRelease:
    {
        if (data->xinput2_mouse_enabled) {
            // This input is being handled by XInput2
            break;
        }

        X11_HandleButtonRelease(_this, data, SDL_GLOBAL_MOUSE_ID, xevent->xbutton.button, xevent->xbutton.time);
    } break;

    case PropertyNotify:
    {
#ifdef DEBUG_XEVENTS
        unsigned char *propdata;
        int status, real_format;
        Atom real_type;
        unsigned long items_read, items_left;

        char *name = X11_XGetAtomName(display, xevent->xproperty.atom);
        if (name) {
            SDL_Log("window 0x%lx: PropertyNotify: %s %s time=%lu", xevent->xany.window, name, (xevent->xproperty.state == PropertyDelete) ? "deleted" : "changed", xevent->xproperty.time);
            X11_XFree(name);
        }

        status = X11_XGetWindowProperty(display, data->xwindow, xevent->xproperty.atom, 0L, 8192L, False, AnyPropertyType, &real_type, &real_format, &items_read, &items_left, &propdata);
        if (status == Success && items_read > 0) {
            if (real_type == XA_INTEGER) {
                int *values = (int *)propdata;

                SDL_Log("{");
                for (i = 0; i < items_read; i++) {
                    SDL_Log(" %d", values[i]);
                }
                SDL_Log(" }");
            } else if (real_type == XA_CARDINAL) {
                if (real_format == 32) {
                    Uint32 *values = (Uint32 *)propdata;

                    SDL_Log("{");
                    for (i = 0; i < items_read; i++) {
                        SDL_Log(" %d", values[i]);
                    }
                    SDL_Log(" }");
                } else if (real_format == 16) {
                    Uint16 *values = (Uint16 *)propdata;

                    SDL_Log("{");
                    for (i = 0; i < items_read; i++) {
                        SDL_Log(" %d", values[i]);
                    }
                    SDL_Log(" }");
                } else if (real_format == 8) {
                    Uint8 *values = (Uint8 *)propdata;

                    SDL_Log("{");
                    for (i = 0; i < items_read; i++) {
                        SDL_Log(" %d", values[i]);
                    }
                    SDL_Log(" }");
                }
            } else if (real_type == XA_STRING ||
                       real_type == videodata->atoms.UTF8_STRING) {
                SDL_Log("{ \"%s\" }", propdata);
            } else if (real_type == XA_ATOM) {
                Atom *atoms = (Atom *)propdata;

                SDL_Log("{");
                for (i = 0; i < items_read; i++) {
                    char *atomname = X11_XGetAtomName(display, atoms[i]);
                    if (atomname) {
                        SDL_Log(" %s", atomname);
                        X11_XFree(atomname);
                    }
                }
                SDL_Log(" }");
            } else {
                char *atomname = X11_XGetAtomName(display, real_type);
                SDL_Log("Unknown type: 0x%lx (%s)", real_type, atomname ? atomname : "UNKNOWN");
                if (atomname) {
                    X11_XFree(atomname);
                }
            }
        }
        if (status == Success) {
            X11_XFree(propdata);
        }
#endif // DEBUG_XEVENTS

        /* Take advantage of this moment to make sure user_time has a
            valid timestamp from the X server, so if we later try to
            raise/restore this window, _NET_ACTIVE_WINDOW can have a
            non-zero timestamp, even if there's never been a mouse or
            key press to this window so far. Note that we don't try to
            set _NET_WM_USER_TIME here, though. That's only for legit
            user interaction with the window. */
        if (!data->user_time) {
            data->user_time = xevent->xproperty.time;
        }

        if (xevent->xproperty.atom == data->videodata->atoms._NET_WM_STATE) {
            /* Get the new state from the window manager.
             * Compositing window managers can alter visibility of windows
             * without ever mapping / unmapping them, so we handle that here,
             * because they use the NETWM protocol to notify us of changes.
             */
            const SDL_WindowFlags flags = X11_GetNetWMState(_this, data->window, xevent->xproperty.window);
            const SDL_WindowFlags changed = flags ^ data->window->flags;

            if ((changed & SDL_WINDOW_HIDDEN) && !(flags & SDL_WINDOW_HIDDEN)) {
                X11_DispatchMapNotify(data);
            }

            if (!SDL_WINDOW_IS_POPUP(data->window)) {
                if (changed & SDL_WINDOW_FULLSCREEN) {
                    data->pending_operation &= ~X11_PENDING_OP_FULLSCREEN;

                    if (flags & SDL_WINDOW_FULLSCREEN) {
                        if (!(flags & SDL_WINDOW_MINIMIZED)) {
                            const bool commit = SDL_memcmp(&data->window->current_fullscreen_mode, &data->requested_fullscreen_mode, sizeof(SDL_DisplayMode)) != 0;

                            // Ensure the maximized flag is cleared before entering fullscreen.
                            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
                            SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_ENTER_FULLSCREEN, 0, 0);
                            if (commit) {
                                /* This was initiated by the compositor, or the mode was changed between the request and the window
                                 * becoming fullscreen. Switch to the application requested mode if necessary.
                                 */
                                SDL_copyp(&data->window->current_fullscreen_mode, &data->window->requested_fullscreen_mode);
                                SDL_UpdateFullscreenMode(data->window, SDL_FULLSCREEN_OP_UPDATE, true);
                            } else {
                                SDL_UpdateFullscreenMode(data->window, SDL_FULLSCREEN_OP_ENTER, false);
                            }
                        }
                    } else {
                        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_LEAVE_FULLSCREEN, 0, 0);
                        SDL_UpdateFullscreenMode(data->window, false, false);

                        SDL_zero(data->requested_fullscreen_mode);

                        // Need to restore or update any limits changed while the window was fullscreen.
                        X11_SetWindowMinMax(data->window, !!(flags & SDL_WINDOW_MAXIMIZED));

                        // Toggle the borders if they were forced on while creating a borderless fullscreen window.
                        if (data->fullscreen_borders_forced_on) {
                            data->toggle_borders = true;
                            data->fullscreen_borders_forced_on = false;
                        }
                    }

                    if ((flags & SDL_WINDOW_FULLSCREEN) &&
                        (data->border_top || data->border_left || data->border_bottom || data->border_right)) {
                        /* If the window is entering fullscreen and the borders are
                         * non-zero sized, turn off size events until the borders are
                         * shut off to avoid bogus window sizes and positions, and
                         * note that the old borders were non-zero for restoration.
                         */
                        data->size_move_event_flags |= X11_SIZE_MOVE_EVENTS_WAIT_FOR_BORDERS;
                        data->previous_borders_nonzero = true;
                    } else if (!(flags & SDL_WINDOW_FULLSCREEN) &&
                               data->previous_borders_nonzero &&
                               (!data->border_top && !data->border_left && !data->border_bottom && !data->border_right)) {
                        /* If the window is leaving fullscreen and the current borders
                         * are zero sized, but weren't when entering fullscreen, turn
                         * off size events until the borders come back to avoid bogus
                         * window sizes and positions.
                         */
                        data->size_move_event_flags |= X11_SIZE_MOVE_EVENTS_WAIT_FOR_BORDERS;
                        data->previous_borders_nonzero = false;
                    } else {
                        data->size_move_event_flags = 0;
                        data->previous_borders_nonzero = false;

                        if (!(data->window->flags & SDL_WINDOW_FULLSCREEN) && data->toggle_borders) {
                            data->toggle_borders = false;
                            X11_SetWindowBordered(_this, data->window, !(data->window->flags & SDL_WINDOW_BORDERLESS));
                        }
                    }
                }
                if ((changed & SDL_WINDOW_MAXIMIZED) && ((flags & SDL_WINDOW_MAXIMIZED) && !(flags & SDL_WINDOW_MINIMIZED))) {
                    data->pending_operation &= ~X11_PENDING_OP_MAXIMIZE;
                    if ((changed & SDL_WINDOW_MINIMIZED)) {
                        data->pending_operation &= ~X11_PENDING_OP_RESTORE;
                        // If coming out of minimized, send a restore event before sending maximized.
                        SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESTORED, 0, 0);
                    }
                    SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MAXIMIZED, 0, 0);
                }
                if ((changed & SDL_WINDOW_MINIMIZED) && (flags & SDL_WINDOW_MINIMIZED)) {
                    data->pending_operation &= ~X11_PENDING_OP_MINIMIZE;
                    SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_MINIMIZED, 0, 0);
                }
                if (!(flags & (SDL_WINDOW_MAXIMIZED | SDL_WINDOW_MINIMIZED))) {
                    data->pending_operation &= ~X11_PENDING_OP_RESTORE;
                    SDL_SendWindowEvent(data->window, SDL_EVENT_WINDOW_RESTORED, 0, 0);

                    // Apply any pending state if restored.
                    if (!(flags & SDL_WINDOW_FULLSCREEN)) {
                        if (data->pending_position) {
                            data->pending_position = false;
                            data->pending_operation |= X11_PENDING_OP_MOVE;
                            data->expected.x = data->window->pending.x - data->border_left;
                            data->expected.y = data->window->pending.y - data->border_top;
                            X11_XMoveWindow(display, data->xwindow, data->expected.x, data->expected.y);
                        }
                        if (data->pending_size) {
                            data->pending_size = false;
                            data->pending_operation |= X11_PENDING_OP_RESIZE;
                            data->expected.w = data->window->pending.w;
                            data->expected.h = data->window->pending.h;
                            X11_XResizeWindow(display, data->xwindow, data->window->pending.w, data->window->pending.h);
                        }
                    }
                }
                if (data->emit_size_move_after_property_notify) {
                    X11_EmitConfigureNotifyEvents(data, &data->pending_xconfigure);
                    data->emit_size_move_after_property_notify = false;
                }
                if ((flags & SDL_WINDOW_INPUT_FOCUS)) {
                    if (data->pending_move) {
                        DispatchWindowMove(_this, data, &data->pending_move_point);
                        data->pending_move = false;
                    }
                }
            }
            if (changed & SDL_WINDOW_OCCLUDED) {
                SDL_SendWindowEvent(data->window, (flags & SDL_WINDOW_OCCLUDED) ? SDL_EVENT_WINDOW_OCCLUDED : SDL_EVENT_WINDOW_EXPOSED, 0, 0);
            }
        } else if (xevent->xproperty.atom == videodata->atoms.XKLAVIER_STATE) {
            /* Hack for Ubuntu 12.04 (etc) that doesn't send MappingNotify
               events when the keyboard layout changes (for example,
               changing from English to French on the menubar's keyboard
               icon). Since it changes the XKLAVIER_STATE property, we
               notice and reinit our keymap here. This might not be the
               right approach, but it seems to work. */
            X11_UpdateKeymap(_this, true);
        } else if (xevent->xproperty.atom == videodata->atoms._NET_FRAME_EXTENTS) {
            /* Events are disabled when leaving fullscreen until the borders appear to avoid
             * incorrect size/position events.
             */
            if (data->size_move_event_flags) {
                data->size_move_event_flags &= ~X11_SIZE_MOVE_EVENTS_WAIT_FOR_BORDERS;
                X11_GetBorderValues(data);

            }
            if (!(data->window->flags & SDL_WINDOW_FULLSCREEN) && data->toggle_borders) {
                data->toggle_borders = false;
                X11_SetWindowBordered(_this, data->window, !(data->window->flags & SDL_WINDOW_BORDERLESS));
            }
        }
    } break;

    case SelectionNotify:
    {
        Atom target = xevent->xselection.target;
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: SelectionNotify (requestor = 0x%lx, target = 0x%lx)", xevent->xany.window,
               xevent->xselection.requestor, xevent->xselection.target);
#endif
        if (target == data->xdnd_req) {
            // read data
            SDL_x11Prop p;
            X11_ReadProperty(&p, display, data->xwindow, videodata->atoms.PRIMARY);

            if (p.format == 8) {
                char *saveptr = NULL;
                char *name = X11_XGetAtomName(display, target);
                if (name) {
                    char *token = SDL_strtok_r((char *)p.data, "\r\n", &saveptr);
                    while (token) {
                        if ((SDL_strcmp("text/plain;charset=utf-8", name) == 0) ||
                            (SDL_strcmp("UTF8_STRING", name) == 0) ||
                            (SDL_strcmp("text/plain", name) == 0) ||
                            (SDL_strcmp("TEXT", name) == 0)) {
                            SDL_SendDropText(data->window, token);
                        } else if (SDL_strcmp("text/uri-list", name) == 0) {
                            if (SDL_URIToLocal(token, token) >= 0) {
                                SDL_SendDropFile(data->window, NULL, token);
                            }
                        }
                        token = SDL_strtok_r(NULL, "\r\n", &saveptr);
                    }
                    X11_XFree(name);
                }
                SDL_SendDropComplete(data->window);
            }
            X11_XFree(p.data);

            // send reply
            SDL_memset(&m, 0, sizeof(XClientMessageEvent));
            m.type = ClientMessage;
            m.display = display;
            m.window = data->xdnd_source;
            m.message_type = videodata->atoms.XdndFinished;
            m.format = 32;
            m.data.l[0] = data->xwindow;
            m.data.l[1] = 1;
            m.data.l[2] = videodata->atoms.XdndActionCopy;
            X11_XSendEvent(display, data->xdnd_source, False, NoEventMask, (XEvent *)&m);

            X11_XSync(display, False);
        }
    } break;

    default:
    {
#ifdef DEBUG_XEVENTS
        SDL_Log("window 0x%lx: Unhandled event %d", xevent->xany.window, xevent->type);
#endif
    } break;
    }
}

static void X11_HandleFocusChanges(SDL_VideoDevice *_this)
{
    SDL_VideoData *videodata = _this->internal;
    int i;

    if (videodata && videodata->windowlist) {
        for (i = 0; i < videodata->numwindows; ++i) {
            SDL_WindowData *data = videodata->windowlist[i];
            if (data && data->pending_focus != PENDING_FOCUS_NONE) {
                Uint64 now = SDL_GetTicks();
                if (now >= data->pending_focus_time) {
                    if (data->pending_focus == PENDING_FOCUS_IN) {
                        X11_DispatchFocusIn(_this, data);
                    } else {
                        X11_DispatchFocusOut(_this, data);
                    }
                    data->pending_focus = PENDING_FOCUS_NONE;
                }
            }
        }
    }
}

static Bool isAnyEvent(Display *display, XEvent *ev, XPointer arg)
{
    return True;
}

static bool X11_PollEvent(Display *display, XEvent *event)
{
    if (!X11_XCheckIfEvent(display, event, isAnyEvent, NULL)) {
        return false;
    }

    return true;
}

void X11_SendWakeupEvent(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_VideoData *data = _this->internal;
    Display *req_display = data->request_display;
    Window xwindow = window->internal->xwindow;
    XClientMessageEvent event;

    SDL_memset(&event, 0, sizeof(XClientMessageEvent));
    event.type = ClientMessage;
    event.display = req_display;
    event.send_event = True;
    event.message_type = data->atoms._SDL_WAKEUP;
    event.format = 8;

    X11_XSendEvent(req_display, xwindow, False, NoEventMask, (XEvent *)&event);
    /* XSendEvent returns a status and it could be BadValue or BadWindow. If an
      error happens it is an SDL's internal error and there is nothing we can do here. */
    X11_XFlush(req_display);
}

int X11_WaitEventTimeout(SDL_VideoDevice *_this, Sint64 timeoutNS)
{
    SDL_VideoData *videodata = _this->internal;
    Display *display;
    XEvent xevent;
    display = videodata->display;

    SDL_zero(xevent);

    // Flush and poll to grab any events already read and queued
    X11_XFlush(display);
    if (X11_PollEvent(display, &xevent)) {
        // Fall through
    } else if (timeoutNS == 0) {
        return 0;
    } else {
        // Use SDL_IOR_NO_RETRY to ensure SIGINT will break us out of our wait
        int err = SDL_IOReady(ConnectionNumber(display), SDL_IOR_READ | SDL_IOR_NO_RETRY, timeoutNS);
        if (err > 0) {
            if (!X11_PollEvent(display, &xevent)) {
                /* Someone may have beat us to reading the fd. Return 1 here to
                 * trigger the normal spurious wakeup logic in the event core. */
                return 1;
            }
        } else if (err == 0) {
            // Timeout
            return 0;
        } else {
            // Error returned from poll()/select()

            if (errno == EINTR) {
                /* If the wait was interrupted by a signal, we may have generated a
                 * SDL_EVENT_QUIT event. Let the caller know to call SDL_PumpEvents(). */
                return 1;
            } else {
                return err;
            }
        }
    }

    X11_DispatchEvent(_this, &xevent);

#ifdef SDL_USE_LIBDBUS
    SDL_DBus_PumpEvents();
#endif
    return 1;
}

void X11_PumpEvents(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;
    XEvent xevent;
    int i;

    /* Check if a display had the mode changed and is waiting for a window to asynchronously become
     * fullscreen. If there is no fullscreen window past the elapsed timeout, revert the mode switch.
     */
    for (i = 0; i < _this->num_displays; ++i) {
        if (_this->displays[i]->internal->mode_switch_deadline_ns) {
            if (_this->displays[i]->fullscreen_window) {
                _this->displays[i]->internal->mode_switch_deadline_ns = 0;
            } else if (SDL_GetTicksNS() >= _this->displays[i]->internal->mode_switch_deadline_ns) {
                SDL_LogError(SDL_LOG_CATEGORY_VIDEO,
                             "Time out elapsed after mode switch on display %" SDL_PRIu32 " with no window becoming fullscreen; reverting", _this->displays[i]->id);
                SDL_SetDisplayModeForDisplay(_this->displays[i], NULL);
                _this->displays[i]->internal->mode_switch_deadline_ns = 0;
            }
        }
    }

    if (data->last_mode_change_deadline) {
        if (SDL_GetTicks() >= data->last_mode_change_deadline) {
            data->last_mode_change_deadline = 0; // assume we're done.
        }
    }

    // Update activity every 30 seconds to prevent screensaver
    if (_this->suspend_screensaver) {
        Uint64 now = SDL_GetTicks();
        if (!data->screensaver_activity || now >= (data->screensaver_activity + 30000)) {
            X11_XResetScreenSaver(data->display);

#ifdef SDL_USE_LIBDBUS
            SDL_DBus_ScreensaverTickle();
#endif

            data->screensaver_activity = now;
        }
    }

    SDL_zero(xevent);

    // Keep processing pending events
    while (X11_PollEvent(data->display, &xevent)) {
        X11_DispatchEvent(_this, &xevent);
    }

#ifdef SDL_USE_LIBDBUS
    SDL_DBus_PumpEvents();
#endif

    // FIXME: Only need to do this when there are pending focus changes
    X11_HandleFocusChanges(_this);

    // FIXME: Only need to do this when there are flashing windows
    for (i = 0; i < data->numwindows; ++i) {
        if (data->windowlist[i] != NULL &&
            data->windowlist[i]->flash_cancel_time &&
            SDL_GetTicks() >= data->windowlist[i]->flash_cancel_time) {
            X11_FlashWindow(_this, data->windowlist[i]->window, SDL_FLASH_CANCEL);
        }
    }

    if (data->xinput_hierarchy_changed) {
        X11_Xinput2UpdateDevices(_this, false);
        data->xinput_hierarchy_changed = false;
    }
}

bool X11_SuspendScreenSaver(SDL_VideoDevice *_this)
{
#ifdef SDL_VIDEO_DRIVER_X11_XSCRNSAVER
    SDL_VideoData *data = _this->internal;
    int dummy;
    int major_version, minor_version;
#endif // SDL_VIDEO_DRIVER_X11_XSCRNSAVER

#ifdef SDL_USE_LIBDBUS
    if (SDL_DBus_ScreensaverInhibit(_this->suspend_screensaver)) {
        return true;
    }

    if (_this->suspend_screensaver) {
        SDL_DBus_ScreensaverTickle();
    }
#endif

#ifdef SDL_VIDEO_DRIVER_X11_XSCRNSAVER
    if (SDL_X11_HAVE_XSS) {
        // X11_XScreenSaverSuspend was introduced in MIT-SCREEN-SAVER 1.1
        if (!X11_XScreenSaverQueryExtension(data->display, &dummy, &dummy) ||
            !X11_XScreenSaverQueryVersion(data->display,
                                          &major_version, &minor_version) ||
            major_version < 1 || (major_version == 1 && minor_version < 1)) {
            return SDL_Unsupported();
        }

        X11_XScreenSaverSuspend(data->display, _this->suspend_screensaver);
        X11_XResetScreenSaver(data->display);
        return true;
    }
#endif
    return SDL_Unsupported();
}

#endif // SDL_VIDEO_DRIVER_X11
