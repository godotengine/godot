//========================================================================
// GLFW 3.4 Wayland - www.glfw.org
//------------------------------------------------------------------------
// Copyright (c) 2014 Jonas Ådahl <jadahl@gmail.com>
//
// This software is provided 'as-is', without any express or implied
// warranty. In no event will the authors be held liable for any damages
// arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose,
// including commercial applications, and to alter it and redistribute it
// freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not
//    claim that you wrote the original software. If you use this software
//    in a product, an acknowledgment in the product documentation would
//    be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not
//    be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source
//    distribution.
//
//========================================================================
// It is fine to use C99 in this file because it will not be built with VS
//========================================================================

#define _POSIX_C_SOURCE 199309L

#include "internal.h"

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <linux/input.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/timerfd.h>
#include <unistd.h>
#include <time.h>
#include <wayland-client.h>


static inline int min(int n1, int n2)
{
    return n1 < n2 ? n1 : n2;
}

static _GLFWwindow* findWindowFromDecorationSurface(struct wl_surface* surface,
                                                    int* which)
{
    int focus;
    _GLFWwindow* window = _glfw.windowListHead;
    if (!which)
        which = &focus;
    while (window)
    {
        if (surface == window->wl.decorations.top.surface)
        {
            *which = topDecoration;
            break;
        }
        if (surface == window->wl.decorations.left.surface)
        {
            *which = leftDecoration;
            break;
        }
        if (surface == window->wl.decorations.right.surface)
        {
            *which = rightDecoration;
            break;
        }
        if (surface == window->wl.decorations.bottom.surface)
        {
            *which = bottomDecoration;
            break;
        }
        window = window->next;
    }
    return window;
}

static void pointerHandleEnter(void* data,
                               struct wl_pointer* pointer,
                               uint32_t serial,
                               struct wl_surface* surface,
                               wl_fixed_t sx,
                               wl_fixed_t sy)
{
    // Happens in the case we just destroyed the surface.
    if (!surface)
        return;

    int focus = 0;
    _GLFWwindow* window = wl_surface_get_user_data(surface);
    if (!window)
    {
        window = findWindowFromDecorationSurface(surface, &focus);
        if (!window)
            return;
    }

    window->wl.decorations.focus = focus;
    _glfw.wl.serial = serial;
    _glfw.wl.pointerFocus = window;

    window->wl.hovered = GLFW_TRUE;

    _glfwPlatformSetCursor(window, window->wl.currentCursor);
    _glfwInputCursorEnter(window, GLFW_TRUE);
}

static void pointerHandleLeave(void* data,
                               struct wl_pointer* pointer,
                               uint32_t serial,
                               struct wl_surface* surface)
{
    _GLFWwindow* window = _glfw.wl.pointerFocus;

    if (!window)
        return;

    window->wl.hovered = GLFW_FALSE;

    _glfw.wl.serial = serial;
    _glfw.wl.pointerFocus = NULL;
    _glfwInputCursorEnter(window, GLFW_FALSE);
    _glfw.wl.cursorPreviousName = NULL;
}

static void setCursor(_GLFWwindow* window, const char* name)
{
    struct wl_buffer* buffer;
    struct wl_cursor* cursor;
    struct wl_cursor_image* image;
    struct wl_surface* surface = _glfw.wl.cursorSurface;
    struct wl_cursor_theme* theme = _glfw.wl.cursorTheme;
    int scale = 1;

    if (window->wl.scale > 1 && _glfw.wl.cursorThemeHiDPI)
    {
        // We only support up to scale=2 for now, since libwayland-cursor
        // requires us to load a different theme for each size.
        scale = 2;
        theme = _glfw.wl.cursorThemeHiDPI;
    }

    cursor = wl_cursor_theme_get_cursor(theme, name);
    if (!cursor)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Standard cursor not found");
        return;
    }
    // TODO: handle animated cursors too.
    image = cursor->images[0];

    if (!image)
        return;

    buffer = wl_cursor_image_get_buffer(image);
    if (!buffer)
        return;
    wl_pointer_set_cursor(_glfw.wl.pointer, _glfw.wl.serial,
                          surface,
                          image->hotspot_x / scale,
                          image->hotspot_y / scale);
    wl_surface_set_buffer_scale(surface, scale);
    wl_surface_attach(surface, buffer, 0, 0);
    wl_surface_damage(surface, 0, 0,
                      image->width, image->height);
    wl_surface_commit(surface);
    _glfw.wl.cursorPreviousName = name;
}

static void pointerHandleMotion(void* data,
                                struct wl_pointer* pointer,
                                uint32_t time,
                                wl_fixed_t sx,
                                wl_fixed_t sy)
{
    _GLFWwindow* window = _glfw.wl.pointerFocus;
    const char* cursorName = NULL;
    double x, y;

    if (!window)
        return;

    if (window->cursorMode == GLFW_CURSOR_DISABLED)
        return;
    x = wl_fixed_to_double(sx);
    y = wl_fixed_to_double(sy);

    switch (window->wl.decorations.focus)
    {
        case mainWindow:
            window->wl.cursorPosX = x;
            window->wl.cursorPosY = y;
            _glfwInputCursorPos(window, x, y);
            _glfw.wl.cursorPreviousName = NULL;
            return;
        case topDecoration:
            if (y < _GLFW_DECORATION_WIDTH)
                cursorName = "n-resize";
            else
                cursorName = "left_ptr";
            break;
        case leftDecoration:
            if (y < _GLFW_DECORATION_WIDTH)
                cursorName = "nw-resize";
            else
                cursorName = "w-resize";
            break;
        case rightDecoration:
            if (y < _GLFW_DECORATION_WIDTH)
                cursorName = "ne-resize";
            else
                cursorName = "e-resize";
            break;
        case bottomDecoration:
            if (x < _GLFW_DECORATION_WIDTH)
                cursorName = "sw-resize";
            else if (x > window->wl.width + _GLFW_DECORATION_WIDTH)
                cursorName = "se-resize";
            else
                cursorName = "s-resize";
            break;
        default:
            assert(0);
    }
    if (_glfw.wl.cursorPreviousName != cursorName)
        setCursor(window, cursorName);
}

static void pointerHandleButton(void* data,
                                struct wl_pointer* pointer,
                                uint32_t serial,
                                uint32_t time,
                                uint32_t button,
                                uint32_t state)
{
    _GLFWwindow* window = _glfw.wl.pointerFocus;
    int glfwButton;
    uint32_t edges = XDG_TOPLEVEL_RESIZE_EDGE_NONE;

    if (!window)
        return;
    if (button == BTN_LEFT)
    {
        switch (window->wl.decorations.focus)
        {
            case mainWindow:
                break;
            case topDecoration:
                if (window->wl.cursorPosY < _GLFW_DECORATION_WIDTH)
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_TOP;
                else
                {
                    xdg_toplevel_move(window->wl.xdg.toplevel, _glfw.wl.seat, serial);
                }
                break;
            case leftDecoration:
                if (window->wl.cursorPosY < _GLFW_DECORATION_WIDTH)
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_TOP_LEFT;
                else
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_LEFT;
                break;
            case rightDecoration:
                if (window->wl.cursorPosY < _GLFW_DECORATION_WIDTH)
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_TOP_RIGHT;
                else
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_RIGHT;
                break;
            case bottomDecoration:
                if (window->wl.cursorPosX < _GLFW_DECORATION_WIDTH)
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_BOTTOM_LEFT;
                else if (window->wl.cursorPosX > window->wl.width + _GLFW_DECORATION_WIDTH)
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_BOTTOM_RIGHT;
                else
                    edges = XDG_TOPLEVEL_RESIZE_EDGE_BOTTOM;
                break;
            default:
                assert(0);
        }
        if (edges != XDG_TOPLEVEL_RESIZE_EDGE_NONE)
        {
            xdg_toplevel_resize(window->wl.xdg.toplevel, _glfw.wl.seat,
                                serial, edges);
        }
    }
    else if (button == BTN_RIGHT)
    {
        if (window->wl.decorations.focus != mainWindow && window->wl.xdg.toplevel)
        {
            xdg_toplevel_show_window_menu(window->wl.xdg.toplevel,
                                          _glfw.wl.seat, serial,
                                          window->wl.cursorPosX,
                                          window->wl.cursorPosY);
            return;
        }
    }

    // Don’t pass the button to the user if it was related to a decoration.
    if (window->wl.decorations.focus != mainWindow)
        return;

    _glfw.wl.serial = serial;

    /* Makes left, right and middle 0, 1 and 2. Overall order follows evdev
     * codes. */
    glfwButton = button - BTN_LEFT;

    _glfwInputMouseClick(window,
                         glfwButton,
                         state == WL_POINTER_BUTTON_STATE_PRESSED
                                ? GLFW_PRESS
                                : GLFW_RELEASE,
                         _glfw.wl.xkb.modifiers);
}

static void pointerHandleAxis(void* data,
                              struct wl_pointer* pointer,
                              uint32_t time,
                              uint32_t axis,
                              wl_fixed_t value)
{
    _GLFWwindow* window = _glfw.wl.pointerFocus;
    double x = 0.0, y = 0.0;
    // Wayland scroll events are in pointer motion coordinate space (think two
    // finger scroll).  The factor 10 is commonly used to convert to "scroll
    // step means 1.0.
    const double scrollFactor = 1.0 / 10.0;

    if (!window)
        return;

    assert(axis == WL_POINTER_AXIS_HORIZONTAL_SCROLL ||
           axis == WL_POINTER_AXIS_VERTICAL_SCROLL);

    if (axis == WL_POINTER_AXIS_HORIZONTAL_SCROLL)
        x = wl_fixed_to_double(value) * scrollFactor;
    else if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL)
        y = wl_fixed_to_double(value) * scrollFactor;

    _glfwInputScroll(window, x, y);
}

static const struct wl_pointer_listener pointerListener = {
    pointerHandleEnter,
    pointerHandleLeave,
    pointerHandleMotion,
    pointerHandleButton,
    pointerHandleAxis,
};

static void keyboardHandleKeymap(void* data,
                                 struct wl_keyboard* keyboard,
                                 uint32_t format,
                                 int fd,
                                 uint32_t size)
{
    struct xkb_keymap* keymap;
    struct xkb_state* state;

#ifdef HAVE_XKBCOMMON_COMPOSE_H
    struct xkb_compose_table* composeTable;
    struct xkb_compose_state* composeState;
#endif

    char* mapStr;
    const char* locale;

    if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1)
    {
        close(fd);
        return;
    }

    mapStr = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapStr == MAP_FAILED) {
        close(fd);
        return;
    }

    keymap = xkb_keymap_new_from_string(_glfw.wl.xkb.context,
                                        mapStr,
                                        XKB_KEYMAP_FORMAT_TEXT_V1,
                                        0);
    munmap(mapStr, size);
    close(fd);

    if (!keymap)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to compile keymap");
        return;
    }

    state = xkb_state_new(keymap);
    if (!state)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to create XKB state");
        xkb_keymap_unref(keymap);
        return;
    }

    // Look up the preferred locale, falling back to "C" as default.
    locale = getenv("LC_ALL");
    if (!locale)
        locale = getenv("LC_CTYPE");
    if (!locale)
        locale = getenv("LANG");
    if (!locale)
        locale = "C";

#ifdef HAVE_XKBCOMMON_COMPOSE_H
    composeTable =
        xkb_compose_table_new_from_locale(_glfw.wl.xkb.context, locale,
                                          XKB_COMPOSE_COMPILE_NO_FLAGS);
    if (composeTable)
    {
        composeState =
            xkb_compose_state_new(composeTable, XKB_COMPOSE_STATE_NO_FLAGS);
        xkb_compose_table_unref(composeTable);
        if (composeState)
            _glfw.wl.xkb.composeState = composeState;
        else
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Failed to create XKB compose state");
    }
    else
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to create XKB compose table");
    }
#endif

    xkb_keymap_unref(_glfw.wl.xkb.keymap);
    xkb_state_unref(_glfw.wl.xkb.state);
    _glfw.wl.xkb.keymap = keymap;
    _glfw.wl.xkb.state = state;

    _glfw.wl.xkb.controlMask =
        1 << xkb_keymap_mod_get_index(_glfw.wl.xkb.keymap, "Control");
    _glfw.wl.xkb.altMask =
        1 << xkb_keymap_mod_get_index(_glfw.wl.xkb.keymap, "Mod1");
    _glfw.wl.xkb.shiftMask =
        1 << xkb_keymap_mod_get_index(_glfw.wl.xkb.keymap, "Shift");
    _glfw.wl.xkb.superMask =
        1 << xkb_keymap_mod_get_index(_glfw.wl.xkb.keymap, "Mod4");
    _glfw.wl.xkb.capsLockMask =
        1 << xkb_keymap_mod_get_index(_glfw.wl.xkb.keymap, "Lock");
    _glfw.wl.xkb.numLockMask =
        1 << xkb_keymap_mod_get_index(_glfw.wl.xkb.keymap, "Mod2");
}

static void keyboardHandleEnter(void* data,
                                struct wl_keyboard* keyboard,
                                uint32_t serial,
                                struct wl_surface* surface,
                                struct wl_array* keys)
{
    // Happens in the case we just destroyed the surface.
    if (!surface)
        return;

    _GLFWwindow* window = wl_surface_get_user_data(surface);
    if (!window)
    {
        window = findWindowFromDecorationSurface(surface, NULL);
        if (!window)
            return;
    }

    _glfw.wl.serial = serial;
    _glfw.wl.keyboardFocus = window;
    _glfwInputWindowFocus(window, GLFW_TRUE);
}

static void keyboardHandleLeave(void* data,
                                struct wl_keyboard* keyboard,
                                uint32_t serial,
                                struct wl_surface* surface)
{
    _GLFWwindow* window = _glfw.wl.keyboardFocus;

    if (!window)
        return;

    _glfw.wl.serial = serial;
    _glfw.wl.keyboardFocus = NULL;
    _glfwInputWindowFocus(window, GLFW_FALSE);
}

static int toGLFWKeyCode(uint32_t key)
{
    if (key < sizeof(_glfw.wl.keycodes) / sizeof(_glfw.wl.keycodes[0]))
        return _glfw.wl.keycodes[key];

    return GLFW_KEY_UNKNOWN;
}

#ifdef HAVE_XKBCOMMON_COMPOSE_H
static xkb_keysym_t composeSymbol(xkb_keysym_t sym)
{
    if (sym == XKB_KEY_NoSymbol || !_glfw.wl.xkb.composeState)
        return sym;
    if (xkb_compose_state_feed(_glfw.wl.xkb.composeState, sym)
            != XKB_COMPOSE_FEED_ACCEPTED)
        return sym;
    switch (xkb_compose_state_get_status(_glfw.wl.xkb.composeState))
    {
        case XKB_COMPOSE_COMPOSED:
            return xkb_compose_state_get_one_sym(_glfw.wl.xkb.composeState);
        case XKB_COMPOSE_COMPOSING:
        case XKB_COMPOSE_CANCELLED:
            return XKB_KEY_NoSymbol;
        case XKB_COMPOSE_NOTHING:
        default:
            return sym;
    }
}
#endif

static GLFWbool inputChar(_GLFWwindow* window, uint32_t key)
{
    uint32_t code, numSyms;
    long cp;
    const xkb_keysym_t *syms;
    xkb_keysym_t sym;

    code = key + 8;
    numSyms = xkb_state_key_get_syms(_glfw.wl.xkb.state, code, &syms);

    if (numSyms == 1)
    {
#ifdef HAVE_XKBCOMMON_COMPOSE_H
        sym = composeSymbol(syms[0]);
#else
        sym = syms[0];
#endif
        cp = _glfwKeySym2Unicode(sym);
        if (cp != -1)
        {
            const int mods = _glfw.wl.xkb.modifiers;
            const int plain = !(mods & (GLFW_MOD_CONTROL | GLFW_MOD_ALT));
            _glfwInputChar(window, cp, mods, plain);
        }
    }

    return xkb_keymap_key_repeats(_glfw.wl.xkb.keymap, syms[0]);
}

static void keyboardHandleKey(void* data,
                              struct wl_keyboard* keyboard,
                              uint32_t serial,
                              uint32_t time,
                              uint32_t key,
                              uint32_t state)
{
    int keyCode;
    int action;
    _GLFWwindow* window = _glfw.wl.keyboardFocus;
    GLFWbool shouldRepeat;
    struct itimerspec timer = {};

    if (!window)
        return;

    keyCode = toGLFWKeyCode(key);
    action = state == WL_KEYBOARD_KEY_STATE_PRESSED
            ? GLFW_PRESS : GLFW_RELEASE;

    _glfw.wl.serial = serial;
    _glfwInputKey(window, keyCode, key, action,
                  _glfw.wl.xkb.modifiers);

    if (action == GLFW_PRESS)
    {
        shouldRepeat = inputChar(window, key);

        if (shouldRepeat && _glfw.wl.keyboardRepeatRate > 0)
        {
            _glfw.wl.keyboardLastKey = keyCode;
            _glfw.wl.keyboardLastScancode = key;
            if (_glfw.wl.keyboardRepeatRate > 1)
                timer.it_interval.tv_nsec = 1000000000 / _glfw.wl.keyboardRepeatRate;
            else
                timer.it_interval.tv_sec = 1;
            timer.it_value.tv_sec = _glfw.wl.keyboardRepeatDelay / 1000;
            timer.it_value.tv_nsec = (_glfw.wl.keyboardRepeatDelay % 1000) * 1000000;
        }
    }
    timerfd_settime(_glfw.wl.timerfd, 0, &timer, NULL);
}

static void keyboardHandleModifiers(void* data,
                                    struct wl_keyboard* keyboard,
                                    uint32_t serial,
                                    uint32_t modsDepressed,
                                    uint32_t modsLatched,
                                    uint32_t modsLocked,
                                    uint32_t group)
{
    xkb_mod_mask_t mask;
    unsigned int modifiers = 0;

    _glfw.wl.serial = serial;

    if (!_glfw.wl.xkb.keymap)
        return;

    xkb_state_update_mask(_glfw.wl.xkb.state,
                          modsDepressed,
                          modsLatched,
                          modsLocked,
                          0,
                          0,
                          group);

    mask = xkb_state_serialize_mods(_glfw.wl.xkb.state,
                                    XKB_STATE_MODS_DEPRESSED |
                                    XKB_STATE_LAYOUT_DEPRESSED |
                                    XKB_STATE_MODS_LATCHED |
                                    XKB_STATE_LAYOUT_LATCHED);
    if (mask & _glfw.wl.xkb.controlMask)
        modifiers |= GLFW_MOD_CONTROL;
    if (mask & _glfw.wl.xkb.altMask)
        modifiers |= GLFW_MOD_ALT;
    if (mask & _glfw.wl.xkb.shiftMask)
        modifiers |= GLFW_MOD_SHIFT;
    if (mask & _glfw.wl.xkb.superMask)
        modifiers |= GLFW_MOD_SUPER;
    if (mask & _glfw.wl.xkb.capsLockMask)
        modifiers |= GLFW_MOD_CAPS_LOCK;
    if (mask & _glfw.wl.xkb.numLockMask)
        modifiers |= GLFW_MOD_NUM_LOCK;
    _glfw.wl.xkb.modifiers = modifiers;
}

#ifdef WL_KEYBOARD_REPEAT_INFO_SINCE_VERSION
static void keyboardHandleRepeatInfo(void* data,
                                     struct wl_keyboard* keyboard,
                                     int32_t rate,
                                     int32_t delay)
{
    if (keyboard != _glfw.wl.keyboard)
        return;

    _glfw.wl.keyboardRepeatRate = rate;
    _glfw.wl.keyboardRepeatDelay = delay;
}
#endif

static const struct wl_keyboard_listener keyboardListener = {
    keyboardHandleKeymap,
    keyboardHandleEnter,
    keyboardHandleLeave,
    keyboardHandleKey,
    keyboardHandleModifiers,
#ifdef WL_KEYBOARD_REPEAT_INFO_SINCE_VERSION
    keyboardHandleRepeatInfo,
#endif
};

static void seatHandleCapabilities(void* data,
                                   struct wl_seat* seat,
                                   enum wl_seat_capability caps)
{
    if ((caps & WL_SEAT_CAPABILITY_POINTER) && !_glfw.wl.pointer)
    {
        _glfw.wl.pointer = wl_seat_get_pointer(seat);
        wl_pointer_add_listener(_glfw.wl.pointer, &pointerListener, NULL);
    }
    else if (!(caps & WL_SEAT_CAPABILITY_POINTER) && _glfw.wl.pointer)
    {
        wl_pointer_destroy(_glfw.wl.pointer);
        _glfw.wl.pointer = NULL;
    }

    if ((caps & WL_SEAT_CAPABILITY_KEYBOARD) && !_glfw.wl.keyboard)
    {
        _glfw.wl.keyboard = wl_seat_get_keyboard(seat);
        wl_keyboard_add_listener(_glfw.wl.keyboard, &keyboardListener, NULL);
    }
    else if (!(caps & WL_SEAT_CAPABILITY_KEYBOARD) && _glfw.wl.keyboard)
    {
        wl_keyboard_destroy(_glfw.wl.keyboard);
        _glfw.wl.keyboard = NULL;
    }
}

static void seatHandleName(void* data,
                           struct wl_seat* seat,
                           const char* name)
{
}

static const struct wl_seat_listener seatListener = {
    seatHandleCapabilities,
    seatHandleName,
};

static void dataOfferHandleOffer(void* data,
                                 struct wl_data_offer* dataOffer,
                                 const char* mimeType)
{
}

static const struct wl_data_offer_listener dataOfferListener = {
    dataOfferHandleOffer,
};

static void dataDeviceHandleDataOffer(void* data,
                                      struct wl_data_device* dataDevice,
                                      struct wl_data_offer* id)
{
    if (_glfw.wl.dataOffer)
        wl_data_offer_destroy(_glfw.wl.dataOffer);

    _glfw.wl.dataOffer = id;
    wl_data_offer_add_listener(_glfw.wl.dataOffer, &dataOfferListener, NULL);
}

static void dataDeviceHandleEnter(void* data,
                                  struct wl_data_device* dataDevice,
                                  uint32_t serial,
                                  struct wl_surface *surface,
                                  wl_fixed_t x,
                                  wl_fixed_t y,
                                  struct wl_data_offer *id)
{
}

static void dataDeviceHandleLeave(void* data,
                                  struct wl_data_device* dataDevice)
{
}

static void dataDeviceHandleMotion(void* data,
                                   struct wl_data_device* dataDevice,
                                   uint32_t time,
                                   wl_fixed_t x,
                                   wl_fixed_t y)
{
}

static void dataDeviceHandleDrop(void* data,
                                 struct wl_data_device* dataDevice)
{
}

static void dataDeviceHandleSelection(void* data,
                                      struct wl_data_device* dataDevice,
                                      struct wl_data_offer* id)
{
}

static const struct wl_data_device_listener dataDeviceListener = {
    dataDeviceHandleDataOffer,
    dataDeviceHandleEnter,
    dataDeviceHandleLeave,
    dataDeviceHandleMotion,
    dataDeviceHandleDrop,
    dataDeviceHandleSelection,
};

static void wmBaseHandlePing(void* data,
                             struct xdg_wm_base* wmBase,
                             uint32_t serial)
{
    xdg_wm_base_pong(wmBase, serial);
}

static const struct xdg_wm_base_listener wmBaseListener = {
    wmBaseHandlePing
};

static void registryHandleGlobal(void* data,
                                 struct wl_registry* registry,
                                 uint32_t name,
                                 const char* interface,
                                 uint32_t version)
{
    if (strcmp(interface, "wl_compositor") == 0)
    {
        _glfw.wl.compositorVersion = min(3, version);
        _glfw.wl.compositor =
            wl_registry_bind(registry, name, &wl_compositor_interface,
                             _glfw.wl.compositorVersion);
    }
    else if (strcmp(interface, "wl_subcompositor") == 0)
    {
        _glfw.wl.subcompositor =
            wl_registry_bind(registry, name, &wl_subcompositor_interface, 1);
    }
    else if (strcmp(interface, "wl_shm") == 0)
    {
        _glfw.wl.shm =
            wl_registry_bind(registry, name, &wl_shm_interface, 1);
    }
    else if (strcmp(interface, "wl_output") == 0)
    {
        _glfwAddOutputWayland(name, version);
    }
    else if (strcmp(interface, "wl_seat") == 0)
    {
        if (!_glfw.wl.seat)
        {
            _glfw.wl.seatVersion = min(4, version);
            _glfw.wl.seat =
                wl_registry_bind(registry, name, &wl_seat_interface,
                                 _glfw.wl.seatVersion);
            wl_seat_add_listener(_glfw.wl.seat, &seatListener, NULL);
        }
    }
    else if (strcmp(interface, "wl_data_device_manager") == 0)
    {
        if (!_glfw.wl.dataDeviceManager)
        {
            _glfw.wl.dataDeviceManager =
                wl_registry_bind(registry, name,
                                 &wl_data_device_manager_interface, 1);
        }
    }
    else if (strcmp(interface, "xdg_wm_base") == 0)
    {
        _glfw.wl.wmBase =
            wl_registry_bind(registry, name, &xdg_wm_base_interface, 1);
        xdg_wm_base_add_listener(_glfw.wl.wmBase, &wmBaseListener, NULL);
    }
    else if (strcmp(interface, "zxdg_decoration_manager_v1") == 0)
    {
        _glfw.wl.decorationManager =
            wl_registry_bind(registry, name,
                             &zxdg_decoration_manager_v1_interface,
                             1);
    }
    else if (strcmp(interface, "wp_viewporter") == 0)
    {
        _glfw.wl.viewporter =
            wl_registry_bind(registry, name, &wp_viewporter_interface, 1);
    }
    else if (strcmp(interface, "zwp_relative_pointer_manager_v1") == 0)
    {
        _glfw.wl.relativePointerManager =
            wl_registry_bind(registry, name,
                             &zwp_relative_pointer_manager_v1_interface,
                             1);
    }
    else if (strcmp(interface, "zwp_pointer_constraints_v1") == 0)
    {
        _glfw.wl.pointerConstraints =
            wl_registry_bind(registry, name,
                             &zwp_pointer_constraints_v1_interface,
                             1);
    }
    else if (strcmp(interface, "zwp_idle_inhibit_manager_v1") == 0)
    {
        _glfw.wl.idleInhibitManager =
            wl_registry_bind(registry, name,
                             &zwp_idle_inhibit_manager_v1_interface,
                             1);
    }
}

static void registryHandleGlobalRemove(void *data,
                                       struct wl_registry *registry,
                                       uint32_t name)
{
    int i;
    _GLFWmonitor* monitor;

    for (i = 0; i < _glfw.monitorCount; ++i)
    {
        monitor = _glfw.monitors[i];
        if (monitor->wl.name == name)
        {
            _glfwInputMonitor(monitor, GLFW_DISCONNECTED, 0);
            return;
        }
    }
}


static const struct wl_registry_listener registryListener = {
    registryHandleGlobal,
    registryHandleGlobalRemove
};

// Create key code translation tables
//
static void createKeyTables(void)
{
    int scancode;

    memset(_glfw.wl.keycodes, -1, sizeof(_glfw.wl.keycodes));
    memset(_glfw.wl.scancodes, -1, sizeof(_glfw.wl.scancodes));

    _glfw.wl.keycodes[KEY_GRAVE]      = GLFW_KEY_GRAVE_ACCENT;
    _glfw.wl.keycodes[KEY_1]          = GLFW_KEY_1;
    _glfw.wl.keycodes[KEY_2]          = GLFW_KEY_2;
    _glfw.wl.keycodes[KEY_3]          = GLFW_KEY_3;
    _glfw.wl.keycodes[KEY_4]          = GLFW_KEY_4;
    _glfw.wl.keycodes[KEY_5]          = GLFW_KEY_5;
    _glfw.wl.keycodes[KEY_6]          = GLFW_KEY_6;
    _glfw.wl.keycodes[KEY_7]          = GLFW_KEY_7;
    _glfw.wl.keycodes[KEY_8]          = GLFW_KEY_8;
    _glfw.wl.keycodes[KEY_9]          = GLFW_KEY_9;
    _glfw.wl.keycodes[KEY_0]          = GLFW_KEY_0;
    _glfw.wl.keycodes[KEY_SPACE]      = GLFW_KEY_SPACE;
    _glfw.wl.keycodes[KEY_MINUS]      = GLFW_KEY_MINUS;
    _glfw.wl.keycodes[KEY_EQUAL]      = GLFW_KEY_EQUAL;
    _glfw.wl.keycodes[KEY_Q]          = GLFW_KEY_Q;
    _glfw.wl.keycodes[KEY_W]          = GLFW_KEY_W;
    _glfw.wl.keycodes[KEY_E]          = GLFW_KEY_E;
    _glfw.wl.keycodes[KEY_R]          = GLFW_KEY_R;
    _glfw.wl.keycodes[KEY_T]          = GLFW_KEY_T;
    _glfw.wl.keycodes[KEY_Y]          = GLFW_KEY_Y;
    _glfw.wl.keycodes[KEY_U]          = GLFW_KEY_U;
    _glfw.wl.keycodes[KEY_I]          = GLFW_KEY_I;
    _glfw.wl.keycodes[KEY_O]          = GLFW_KEY_O;
    _glfw.wl.keycodes[KEY_P]          = GLFW_KEY_P;
    _glfw.wl.keycodes[KEY_LEFTBRACE]  = GLFW_KEY_LEFT_BRACKET;
    _glfw.wl.keycodes[KEY_RIGHTBRACE] = GLFW_KEY_RIGHT_BRACKET;
    _glfw.wl.keycodes[KEY_A]          = GLFW_KEY_A;
    _glfw.wl.keycodes[KEY_S]          = GLFW_KEY_S;
    _glfw.wl.keycodes[KEY_D]          = GLFW_KEY_D;
    _glfw.wl.keycodes[KEY_F]          = GLFW_KEY_F;
    _glfw.wl.keycodes[KEY_G]          = GLFW_KEY_G;
    _glfw.wl.keycodes[KEY_H]          = GLFW_KEY_H;
    _glfw.wl.keycodes[KEY_J]          = GLFW_KEY_J;
    _glfw.wl.keycodes[KEY_K]          = GLFW_KEY_K;
    _glfw.wl.keycodes[KEY_L]          = GLFW_KEY_L;
    _glfw.wl.keycodes[KEY_SEMICOLON]  = GLFW_KEY_SEMICOLON;
    _glfw.wl.keycodes[KEY_APOSTROPHE] = GLFW_KEY_APOSTROPHE;
    _glfw.wl.keycodes[KEY_Z]          = GLFW_KEY_Z;
    _glfw.wl.keycodes[KEY_X]          = GLFW_KEY_X;
    _glfw.wl.keycodes[KEY_C]          = GLFW_KEY_C;
    _glfw.wl.keycodes[KEY_V]          = GLFW_KEY_V;
    _glfw.wl.keycodes[KEY_B]          = GLFW_KEY_B;
    _glfw.wl.keycodes[KEY_N]          = GLFW_KEY_N;
    _glfw.wl.keycodes[KEY_M]          = GLFW_KEY_M;
    _glfw.wl.keycodes[KEY_COMMA]      = GLFW_KEY_COMMA;
    _glfw.wl.keycodes[KEY_DOT]        = GLFW_KEY_PERIOD;
    _glfw.wl.keycodes[KEY_SLASH]      = GLFW_KEY_SLASH;
    _glfw.wl.keycodes[KEY_BACKSLASH]  = GLFW_KEY_BACKSLASH;
    _glfw.wl.keycodes[KEY_ESC]        = GLFW_KEY_ESCAPE;
    _glfw.wl.keycodes[KEY_TAB]        = GLFW_KEY_TAB;
    _glfw.wl.keycodes[KEY_LEFTSHIFT]  = GLFW_KEY_LEFT_SHIFT;
    _glfw.wl.keycodes[KEY_RIGHTSHIFT] = GLFW_KEY_RIGHT_SHIFT;
    _glfw.wl.keycodes[KEY_LEFTCTRL]   = GLFW_KEY_LEFT_CONTROL;
    _glfw.wl.keycodes[KEY_RIGHTCTRL]  = GLFW_KEY_RIGHT_CONTROL;
    _glfw.wl.keycodes[KEY_LEFTALT]    = GLFW_KEY_LEFT_ALT;
    _glfw.wl.keycodes[KEY_RIGHTALT]   = GLFW_KEY_RIGHT_ALT;
    _glfw.wl.keycodes[KEY_LEFTMETA]   = GLFW_KEY_LEFT_SUPER;
    _glfw.wl.keycodes[KEY_RIGHTMETA]  = GLFW_KEY_RIGHT_SUPER;
    _glfw.wl.keycodes[KEY_MENU]       = GLFW_KEY_MENU;
    _glfw.wl.keycodes[KEY_NUMLOCK]    = GLFW_KEY_NUM_LOCK;
    _glfw.wl.keycodes[KEY_CAPSLOCK]   = GLFW_KEY_CAPS_LOCK;
    _glfw.wl.keycodes[KEY_PRINT]      = GLFW_KEY_PRINT_SCREEN;
    _glfw.wl.keycodes[KEY_SCROLLLOCK] = GLFW_KEY_SCROLL_LOCK;
    _glfw.wl.keycodes[KEY_PAUSE]      = GLFW_KEY_PAUSE;
    _glfw.wl.keycodes[KEY_DELETE]     = GLFW_KEY_DELETE;
    _glfw.wl.keycodes[KEY_BACKSPACE]  = GLFW_KEY_BACKSPACE;
    _glfw.wl.keycodes[KEY_ENTER]      = GLFW_KEY_ENTER;
    _glfw.wl.keycodes[KEY_HOME]       = GLFW_KEY_HOME;
    _glfw.wl.keycodes[KEY_END]        = GLFW_KEY_END;
    _glfw.wl.keycodes[KEY_PAGEUP]     = GLFW_KEY_PAGE_UP;
    _glfw.wl.keycodes[KEY_PAGEDOWN]   = GLFW_KEY_PAGE_DOWN;
    _glfw.wl.keycodes[KEY_INSERT]     = GLFW_KEY_INSERT;
    _glfw.wl.keycodes[KEY_LEFT]       = GLFW_KEY_LEFT;
    _glfw.wl.keycodes[KEY_RIGHT]      = GLFW_KEY_RIGHT;
    _glfw.wl.keycodes[KEY_DOWN]       = GLFW_KEY_DOWN;
    _glfw.wl.keycodes[KEY_UP]         = GLFW_KEY_UP;
    _glfw.wl.keycodes[KEY_F1]         = GLFW_KEY_F1;
    _glfw.wl.keycodes[KEY_F2]         = GLFW_KEY_F2;
    _glfw.wl.keycodes[KEY_F3]         = GLFW_KEY_F3;
    _glfw.wl.keycodes[KEY_F4]         = GLFW_KEY_F4;
    _glfw.wl.keycodes[KEY_F5]         = GLFW_KEY_F5;
    _glfw.wl.keycodes[KEY_F6]         = GLFW_KEY_F6;
    _glfw.wl.keycodes[KEY_F7]         = GLFW_KEY_F7;
    _glfw.wl.keycodes[KEY_F8]         = GLFW_KEY_F8;
    _glfw.wl.keycodes[KEY_F9]         = GLFW_KEY_F9;
    _glfw.wl.keycodes[KEY_F10]        = GLFW_KEY_F10;
    _glfw.wl.keycodes[KEY_F11]        = GLFW_KEY_F11;
    _glfw.wl.keycodes[KEY_F12]        = GLFW_KEY_F12;
    _glfw.wl.keycodes[KEY_F13]        = GLFW_KEY_F13;
    _glfw.wl.keycodes[KEY_F14]        = GLFW_KEY_F14;
    _glfw.wl.keycodes[KEY_F15]        = GLFW_KEY_F15;
    _glfw.wl.keycodes[KEY_F16]        = GLFW_KEY_F16;
    _glfw.wl.keycodes[KEY_F17]        = GLFW_KEY_F17;
    _glfw.wl.keycodes[KEY_F18]        = GLFW_KEY_F18;
    _glfw.wl.keycodes[KEY_F19]        = GLFW_KEY_F19;
    _glfw.wl.keycodes[KEY_F20]        = GLFW_KEY_F20;
    _glfw.wl.keycodes[KEY_F21]        = GLFW_KEY_F21;
    _glfw.wl.keycodes[KEY_F22]        = GLFW_KEY_F22;
    _glfw.wl.keycodes[KEY_F23]        = GLFW_KEY_F23;
    _glfw.wl.keycodes[KEY_F24]        = GLFW_KEY_F24;
    _glfw.wl.keycodes[KEY_KPSLASH]    = GLFW_KEY_KP_DIVIDE;
    _glfw.wl.keycodes[KEY_KPDOT]      = GLFW_KEY_KP_MULTIPLY;
    _glfw.wl.keycodes[KEY_KPMINUS]    = GLFW_KEY_KP_SUBTRACT;
    _glfw.wl.keycodes[KEY_KPPLUS]     = GLFW_KEY_KP_ADD;
    _glfw.wl.keycodes[KEY_KP0]        = GLFW_KEY_KP_0;
    _glfw.wl.keycodes[KEY_KP1]        = GLFW_KEY_KP_1;
    _glfw.wl.keycodes[KEY_KP2]        = GLFW_KEY_KP_2;
    _glfw.wl.keycodes[KEY_KP3]        = GLFW_KEY_KP_3;
    _glfw.wl.keycodes[KEY_KP4]        = GLFW_KEY_KP_4;
    _glfw.wl.keycodes[KEY_KP5]        = GLFW_KEY_KP_5;
    _glfw.wl.keycodes[KEY_KP6]        = GLFW_KEY_KP_6;
    _glfw.wl.keycodes[KEY_KP7]        = GLFW_KEY_KP_7;
    _glfw.wl.keycodes[KEY_KP8]        = GLFW_KEY_KP_8;
    _glfw.wl.keycodes[KEY_KP9]        = GLFW_KEY_KP_9;
    _glfw.wl.keycodes[KEY_KPCOMMA]    = GLFW_KEY_KP_DECIMAL;
    _glfw.wl.keycodes[KEY_KPEQUAL]    = GLFW_KEY_KP_EQUAL;
    _glfw.wl.keycodes[KEY_KPENTER]    = GLFW_KEY_KP_ENTER;

    for (scancode = 0;  scancode < 256;  scancode++)
    {
        if (_glfw.wl.keycodes[scancode] > 0)
            _glfw.wl.scancodes[_glfw.wl.keycodes[scancode]] = scancode;
    }
}


//////////////////////////////////////////////////////////////////////////
//////                       GLFW platform API                      //////
//////////////////////////////////////////////////////////////////////////

int _glfwPlatformInit(void)
{
    const char *cursorTheme;
    const char *cursorSizeStr;
    char *cursorSizeEnd;
    long cursorSizeLong;
    int cursorSize;

    _glfw.wl.cursor.handle = _glfw_dlopen("libwayland-cursor.so.0");
    if (!_glfw.wl.cursor.handle)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to open libwayland-cursor");
        return GLFW_FALSE;
    }

    _glfw.wl.cursor.theme_load = (PFN_wl_cursor_theme_load)
        _glfw_dlsym(_glfw.wl.cursor.handle, "wl_cursor_theme_load");
    _glfw.wl.cursor.theme_destroy = (PFN_wl_cursor_theme_destroy)
        _glfw_dlsym(_glfw.wl.cursor.handle, "wl_cursor_theme_destroy");
    _glfw.wl.cursor.theme_get_cursor = (PFN_wl_cursor_theme_get_cursor)
        _glfw_dlsym(_glfw.wl.cursor.handle, "wl_cursor_theme_get_cursor");
    _glfw.wl.cursor.image_get_buffer = (PFN_wl_cursor_image_get_buffer)
        _glfw_dlsym(_glfw.wl.cursor.handle, "wl_cursor_image_get_buffer");

    _glfw.wl.egl.handle = _glfw_dlopen("libwayland-egl.so.1");
    if (!_glfw.wl.egl.handle)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to open libwayland-egl");
        return GLFW_FALSE;
    }

    _glfw.wl.egl.window_create = (PFN_wl_egl_window_create)
        _glfw_dlsym(_glfw.wl.egl.handle, "wl_egl_window_create");
    _glfw.wl.egl.window_destroy = (PFN_wl_egl_window_destroy)
        _glfw_dlsym(_glfw.wl.egl.handle, "wl_egl_window_destroy");
    _glfw.wl.egl.window_resize = (PFN_wl_egl_window_resize)
        _glfw_dlsym(_glfw.wl.egl.handle, "wl_egl_window_resize");

    _glfw.wl.xkb.handle = _glfw_dlopen("libxkbcommon.so.0");
    if (!_glfw.wl.xkb.handle)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to open libxkbcommon");
        return GLFW_FALSE;
    }

    _glfw.wl.xkb.context_new = (PFN_xkb_context_new)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_context_new");
    _glfw.wl.xkb.context_unref = (PFN_xkb_context_unref)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_context_unref");
    _glfw.wl.xkb.keymap_new_from_string = (PFN_xkb_keymap_new_from_string)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_keymap_new_from_string");
    _glfw.wl.xkb.keymap_unref = (PFN_xkb_keymap_unref)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_keymap_unref");
    _glfw.wl.xkb.keymap_mod_get_index = (PFN_xkb_keymap_mod_get_index)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_keymap_mod_get_index");
    _glfw.wl.xkb.keymap_key_repeats = (PFN_xkb_keymap_key_repeats)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_keymap_key_repeats");
    _glfw.wl.xkb.state_new = (PFN_xkb_state_new)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_state_new");
    _glfw.wl.xkb.state_unref = (PFN_xkb_state_unref)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_state_unref");
    _glfw.wl.xkb.state_key_get_syms = (PFN_xkb_state_key_get_syms)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_state_key_get_syms");
    _glfw.wl.xkb.state_update_mask = (PFN_xkb_state_update_mask)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_state_update_mask");
    _glfw.wl.xkb.state_serialize_mods = (PFN_xkb_state_serialize_mods)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_state_serialize_mods");

#ifdef HAVE_XKBCOMMON_COMPOSE_H
    _glfw.wl.xkb.compose_table_new_from_locale = (PFN_xkb_compose_table_new_from_locale)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_table_new_from_locale");
    _glfw.wl.xkb.compose_table_unref = (PFN_xkb_compose_table_unref)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_table_unref");
    _glfw.wl.xkb.compose_state_new = (PFN_xkb_compose_state_new)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_state_new");
    _glfw.wl.xkb.compose_state_unref = (PFN_xkb_compose_state_unref)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_state_unref");
    _glfw.wl.xkb.compose_state_feed = (PFN_xkb_compose_state_feed)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_state_feed");
    _glfw.wl.xkb.compose_state_get_status = (PFN_xkb_compose_state_get_status)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_state_get_status");
    _glfw.wl.xkb.compose_state_get_one_sym = (PFN_xkb_compose_state_get_one_sym)
        _glfw_dlsym(_glfw.wl.xkb.handle, "xkb_compose_state_get_one_sym");
#endif

    _glfw.wl.display = wl_display_connect(NULL);
    if (!_glfw.wl.display)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to connect to display");
        return GLFW_FALSE;
    }

    _glfw.wl.registry = wl_display_get_registry(_glfw.wl.display);
    wl_registry_add_listener(_glfw.wl.registry, &registryListener, NULL);

    createKeyTables();

    _glfw.wl.xkb.context = xkb_context_new(0);
    if (!_glfw.wl.xkb.context)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to initialize xkb context");
        return GLFW_FALSE;
    }

    // Sync so we got all registry objects
    wl_display_roundtrip(_glfw.wl.display);

    // Sync so we got all initial output events
    wl_display_roundtrip(_glfw.wl.display);

#ifdef __linux__
    if (!_glfwInitJoysticksLinux())
        return GLFW_FALSE;
#endif

    _glfwInitTimerPOSIX();

    _glfw.wl.timerfd = -1;
    if (_glfw.wl.seatVersion >= 4)
        _glfw.wl.timerfd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);

    if (!_glfw.wl.wmBase)
    {
        _glfwInputError(GLFW_PLATFORM_ERROR,
                        "Wayland: Failed to find xdg-shell in your compositor");
        return GLFW_FALSE;
    }

    if (_glfw.wl.pointer && _glfw.wl.shm)
    {
        cursorTheme = getenv("XCURSOR_THEME");
        cursorSizeStr = getenv("XCURSOR_SIZE");
        cursorSize = 32;
        if (cursorSizeStr)
        {
            errno = 0;
            cursorSizeLong = strtol(cursorSizeStr, &cursorSizeEnd, 10);
            if (!*cursorSizeEnd && !errno && cursorSizeLong > 0 && cursorSizeLong <= INT_MAX)
                cursorSize = (int)cursorSizeLong;
        }
        _glfw.wl.cursorTheme =
            wl_cursor_theme_load(cursorTheme, cursorSize, _glfw.wl.shm);
        if (!_glfw.wl.cursorTheme)
        {
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Unable to load default cursor theme");
            return GLFW_FALSE;
        }
        // If this happens to be NULL, we just fallback to the scale=1 version.
        _glfw.wl.cursorThemeHiDPI =
            wl_cursor_theme_load(cursorTheme, 2 * cursorSize, _glfw.wl.shm);
        _glfw.wl.cursorSurface =
            wl_compositor_create_surface(_glfw.wl.compositor);
        _glfw.wl.cursorTimerfd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    }

    if (_glfw.wl.seat && _glfw.wl.dataDeviceManager)
    {
        _glfw.wl.dataDevice =
            wl_data_device_manager_get_data_device(_glfw.wl.dataDeviceManager,
                                                   _glfw.wl.seat);
        wl_data_device_add_listener(_glfw.wl.dataDevice, &dataDeviceListener, NULL);
        _glfw.wl.clipboardString = malloc(4096);
        if (!_glfw.wl.clipboardString)
        {
            _glfwInputError(GLFW_PLATFORM_ERROR,
                            "Wayland: Unable to allocate clipboard memory");
            return GLFW_FALSE;
        }
        _glfw.wl.clipboardSize = 4096;
    }

    return GLFW_TRUE;
}

void _glfwPlatformTerminate(void)
{
#ifdef __linux__
    _glfwTerminateJoysticksLinux();
#endif
    _glfwTerminateEGL();
    if (_glfw.wl.egl.handle)
    {
        _glfw_dlclose(_glfw.wl.egl.handle);
        _glfw.wl.egl.handle = NULL;
    }

#ifdef HAVE_XKBCOMMON_COMPOSE_H
    if (_glfw.wl.xkb.composeState)
        xkb_compose_state_unref(_glfw.wl.xkb.composeState);
#endif
    if (_glfw.wl.xkb.keymap)
        xkb_keymap_unref(_glfw.wl.xkb.keymap);
    if (_glfw.wl.xkb.state)
        xkb_state_unref(_glfw.wl.xkb.state);
    if (_glfw.wl.xkb.context)
        xkb_context_unref(_glfw.wl.xkb.context);
    if (_glfw.wl.xkb.handle)
    {
        _glfw_dlclose(_glfw.wl.xkb.handle);
        _glfw.wl.xkb.handle = NULL;
    }

    if (_glfw.wl.cursorTheme)
        wl_cursor_theme_destroy(_glfw.wl.cursorTheme);
    if (_glfw.wl.cursorThemeHiDPI)
        wl_cursor_theme_destroy(_glfw.wl.cursorThemeHiDPI);
    if (_glfw.wl.cursor.handle)
    {
        _glfw_dlclose(_glfw.wl.cursor.handle);
        _glfw.wl.cursor.handle = NULL;
    }

    if (_glfw.wl.cursorSurface)
        wl_surface_destroy(_glfw.wl.cursorSurface);
    if (_glfw.wl.subcompositor)
        wl_subcompositor_destroy(_glfw.wl.subcompositor);
    if (_glfw.wl.compositor)
        wl_compositor_destroy(_glfw.wl.compositor);
    if (_glfw.wl.shm)
        wl_shm_destroy(_glfw.wl.shm);
    if (_glfw.wl.viewporter)
        wp_viewporter_destroy(_glfw.wl.viewporter);
    if (_glfw.wl.decorationManager)
        zxdg_decoration_manager_v1_destroy(_glfw.wl.decorationManager);
    if (_glfw.wl.wmBase)
        xdg_wm_base_destroy(_glfw.wl.wmBase);
    if (_glfw.wl.dataSource)
        wl_data_source_destroy(_glfw.wl.dataSource);
    if (_glfw.wl.dataDevice)
        wl_data_device_destroy(_glfw.wl.dataDevice);
    if (_glfw.wl.dataOffer)
        wl_data_offer_destroy(_glfw.wl.dataOffer);
    if (_glfw.wl.dataDeviceManager)
        wl_data_device_manager_destroy(_glfw.wl.dataDeviceManager);
    if (_glfw.wl.pointer)
        wl_pointer_destroy(_glfw.wl.pointer);
    if (_glfw.wl.keyboard)
        wl_keyboard_destroy(_glfw.wl.keyboard);
    if (_glfw.wl.seat)
        wl_seat_destroy(_glfw.wl.seat);
    if (_glfw.wl.relativePointerManager)
        zwp_relative_pointer_manager_v1_destroy(_glfw.wl.relativePointerManager);
    if (_glfw.wl.pointerConstraints)
        zwp_pointer_constraints_v1_destroy(_glfw.wl.pointerConstraints);
    if (_glfw.wl.idleInhibitManager)
        zwp_idle_inhibit_manager_v1_destroy(_glfw.wl.idleInhibitManager);
    if (_glfw.wl.registry)
        wl_registry_destroy(_glfw.wl.registry);
    if (_glfw.wl.display)
    {
        wl_display_flush(_glfw.wl.display);
        wl_display_disconnect(_glfw.wl.display);
    }

    if (_glfw.wl.timerfd >= 0)
        close(_glfw.wl.timerfd);
    if (_glfw.wl.cursorTimerfd >= 0)
        close(_glfw.wl.cursorTimerfd);

    if (_glfw.wl.clipboardString)
        free(_glfw.wl.clipboardString);
    if (_glfw.wl.clipboardSendString)
        free(_glfw.wl.clipboardSendString);
}

const char* _glfwPlatformGetVersionString(void)
{
    return _GLFW_VERSION_NUMBER " Wayland EGL OSMesa"
#if defined(_POSIX_TIMERS) && defined(_POSIX_MONOTONIC_CLOCK)
        " clock_gettime"
#else
        " gettimeofday"
#endif
        " evdev"
#if defined(_GLFW_BUILD_DLL)
        " shared"
#endif
        ;
}
