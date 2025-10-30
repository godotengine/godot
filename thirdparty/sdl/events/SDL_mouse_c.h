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

#ifndef SDL_mouse_c_h_
#define SDL_mouse_c_h_

// Mouse events not associated with a specific input device
#define SDL_GLOBAL_MOUSE_ID     0

// The default mouse input device, for platforms that don't have multiple mice
#define SDL_DEFAULT_MOUSE_ID    1

typedef struct SDL_CursorData SDL_CursorData;

struct SDL_Cursor
{
    struct SDL_Cursor *next;
    SDL_CursorData *internal;
};

typedef struct
{
    Uint64 last_timestamp;
    double click_motion_x;
    double click_motion_y;
    Uint8 click_count;
} SDL_MouseClickState;

typedef struct
{
    SDL_MouseID mouseID;
    Uint32 buttonstate;

    // Data for double-click tracking
    int num_clickstates;
    SDL_MouseClickState *clickstate;
} SDL_MouseInputSource;

typedef struct
{
    // Create a cursor from a surface
    SDL_Cursor *(*CreateCursor)(SDL_Surface *surface, int hot_x, int hot_y);

    // Create a system cursor
    SDL_Cursor *(*CreateSystemCursor)(SDL_SystemCursor id);

    // Show the specified cursor, or hide if cursor is NULL
    bool (*ShowCursor)(SDL_Cursor *cursor);

    // This is called when a mouse motion event occurs
    bool (*MoveCursor)(SDL_Cursor *cursor);

    // Free a window manager cursor
    void (*FreeCursor)(SDL_Cursor *cursor);

    // Warp the mouse to (x,y) within a window
    bool (*WarpMouse)(SDL_Window *window, float x, float y);

    // Warp the mouse to (x,y) in screen space
    bool (*WarpMouseGlobal)(float x, float y);

    // Set relative mode
    bool (*SetRelativeMouseMode)(bool enabled);

    // Set mouse capture
    bool (*CaptureMouse)(SDL_Window *window);

    // Get absolute mouse coordinates. (x) and (y) are never NULL and set to zero before call.
    SDL_MouseButtonFlags (*GetGlobalMouseState)(float *x, float *y);

    // Platform-specific system mouse transform
    void (*ApplySystemScale)(void *internal, Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, float *x, float *y);
    void *system_scale_data;

    // integer mode data
    Uint8 integer_mode_flags; // 1 to enable mouse quantization, 2 to enable wheel quantization
    float integer_mode_residual_motion_x;
    float integer_mode_residual_motion_y;

    // Data common to all mice
    SDL_Window *focus;
    float x;
    float y;
    float x_accu;
    float y_accu;
    float last_x, last_y; // the last reported x and y coordinates
    float residual_scroll_x;
    float residual_scroll_y;
    double click_motion_x;
    double click_motion_y;
    bool has_position;
    bool relative_mode;
    bool relative_mode_warp_motion;
    bool relative_mode_cursor_visible;
    bool relative_mode_center;
    bool warp_emulation_hint;
    bool warp_emulation_active;
    bool warp_emulation_prohibited;
    Uint64 last_center_warp_time_ns;
    bool enable_normal_speed_scale;
    float normal_speed_scale;
    bool enable_relative_speed_scale;
    float relative_speed_scale;
    bool enable_relative_system_scale;
    Uint32 double_click_time;
    int double_click_radius;
    bool touch_mouse_events;
    bool mouse_touch_events;
    bool pen_mouse_events;
    bool pen_touch_events;
    bool was_touch_mouse_events; // Was a touch-mouse event pending?
    bool added_mouse_touch_device;  // did we SDL_AddTouch() a virtual touch device for the mouse?
    bool added_pen_touch_device;  // did we SDL_AddTouch() a virtual touch device for pens?
#ifdef SDL_PLATFORM_VITA
    Uint8 vita_touch_mouse_device;
#endif
    bool auto_capture;
    bool capture_desired;
    SDL_Window *capture_window;

    // Data for input source state
    int num_sources;
    SDL_MouseInputSource *sources;

    SDL_Cursor *cursors;
    SDL_Cursor *def_cursor;
    SDL_Cursor *cur_cursor;
    bool cursor_shown;

    // Driver-dependent data.
    void *internal;
} SDL_Mouse;

// Initialize the mouse subsystem, called before the main video driver is initialized
extern bool SDL_PreInitMouse(void);

// Finish initializing the mouse subsystem, called after the main video driver was initialized
extern void SDL_PostInitMouse(void);

// Return whether a device is actually a mouse
extern bool SDL_IsMouse(Uint16 vendor, Uint16 product);

// A mouse has been added to the system
extern void SDL_AddMouse(SDL_MouseID mouseID, const char *name, bool send_event);

// A mouse has been removed from the system
extern void SDL_RemoveMouse(SDL_MouseID mouseID, bool send_event);

// Get the mouse state structure
extern SDL_Mouse *SDL_GetMouse(void);

// Set the default mouse cursor
extern void SDL_SetDefaultCursor(SDL_Cursor *cursor);

// Get the preferred default system cursor
extern SDL_SystemCursor SDL_GetDefaultSystemCursor(void);

// Set the mouse focus window
extern void SDL_SetMouseFocus(SDL_Window *window);

// Update the mouse capture window
extern bool SDL_UpdateMouseCapture(bool force_release);

// Send a mouse motion event
extern void SDL_SendMouseMotion(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, bool relative, float x, float y);

// Send a mouse button event
extern void SDL_SendMouseButton(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, Uint8 button, bool down);

// Send a mouse button event with a click count
extern void SDL_SendMouseButtonClicks(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, Uint8 button, bool down, int clicks);

// Send a mouse wheel event
extern void SDL_SendMouseWheel(Uint64 timestamp, SDL_Window *window, SDL_MouseID mouseID, float x, float y, SDL_MouseWheelDirection direction);

// Warp the mouse within the window, potentially overriding relative mode
extern void SDL_PerformWarpMouseInWindow(SDL_Window *window, float x, float y, bool ignore_relative_mode);

// Relative mouse mode
extern bool SDL_SetRelativeMouseMode(bool enabled);
extern bool SDL_GetRelativeMouseMode(void);
extern void SDL_UpdateRelativeMouseMode(void);
extern void SDL_DisableMouseWarpEmulation(void);

// TODO RECONNECT: Set mouse state to "zero"
#if 0
extern void SDL_ResetMouse(void);
#endif // 0

// Check if mouse position is within window or captured by window
extern bool SDL_MousePositionInWindow(SDL_Window *window, float x, float y);

// Shutdown the mouse subsystem
extern void SDL_QuitMouse(void);

#endif // SDL_mouse_c_h_
