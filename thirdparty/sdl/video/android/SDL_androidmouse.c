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

#ifdef SDL_VIDEO_DRIVER_ANDROID

#include "SDL_androidmouse.h"

#include "../../events/SDL_mouse_c.h"

#include "../../core/android/SDL_android.h"

// See Android's MotionEvent class for constants
#define ACTION_DOWN       0
#define ACTION_UP         1
#define ACTION_MOVE       2
#define ACTION_HOVER_MOVE 7
#define ACTION_SCROLL     8
#define BUTTON_PRIMARY    1
#define BUTTON_SECONDARY  2
#define BUTTON_TERTIARY   4
#define BUTTON_BACK       8
#define BUTTON_FORWARD    16

struct SDL_CursorData
{
    int custom_cursor;
    int system_cursor;
};

// Last known Android mouse button state (includes all buttons)
static int last_state;

// Blank cursor
static SDL_Cursor *empty_cursor;

static SDL_Cursor *Android_WrapCursor(int custom_cursor, int system_cursor)
{
    SDL_Cursor *cursor;

    cursor = SDL_calloc(1, sizeof(*cursor));
    if (cursor) {
        SDL_CursorData *data = (SDL_CursorData *)SDL_calloc(1, sizeof(*data));
        if (data) {
            data->custom_cursor = custom_cursor;
            data->system_cursor = system_cursor;
            cursor->internal = data;
        } else {
            SDL_free(cursor);
            cursor = NULL;
        }
    }

    return cursor;
}

static SDL_Cursor *Android_CreateDefaultCursor(void)
{
    SDL_SystemCursor id = SDL_GetDefaultSystemCursor();
    return Android_WrapCursor(0, id);
}

static SDL_Cursor *Android_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    int custom_cursor;
    SDL_Surface *converted;

    converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ARGB8888);
    if (!converted) {
        return NULL;
    }
    custom_cursor = Android_JNI_CreateCustomCursor(converted, hot_x, hot_y);
    SDL_DestroySurface(converted);
    if (!custom_cursor) {
        SDL_Unsupported();
        return NULL;
    }
    return Android_WrapCursor(custom_cursor, 0);
}

static SDL_Cursor *Android_CreateSystemCursor(SDL_SystemCursor id)
{
    return Android_WrapCursor(0, id);
}

static void Android_FreeCursor(SDL_Cursor *cursor)
{
    SDL_CursorData *data = cursor->internal;
    if (data->custom_cursor != 0) {
        Android_JNI_DestroyCustomCursor(data->custom_cursor);
    }
    SDL_free(cursor->internal);
    SDL_free(cursor);
}

static SDL_Cursor *Android_CreateEmptyCursor(void)
{
    if (!empty_cursor) {
        SDL_Surface *empty_surface = SDL_CreateSurface(1, 1, SDL_PIXELFORMAT_ARGB8888);
        if (empty_surface) {
            SDL_memset(empty_surface->pixels, 0, (size_t)empty_surface->h * empty_surface->pitch);
            empty_cursor = Android_CreateCursor(empty_surface, 0, 0);
            SDL_DestroySurface(empty_surface);
        }
    }
    return empty_cursor;
}

static void Android_DestroyEmptyCursor(void)
{
    if (empty_cursor) {
        Android_FreeCursor(empty_cursor);
        empty_cursor = NULL;
    }
}

static bool Android_ShowCursor(SDL_Cursor *cursor)
{
    if (!cursor) {
        cursor = Android_CreateEmptyCursor();
    }
    if (cursor) {
        SDL_CursorData *data = cursor->internal;
        if (data->custom_cursor) {
            if (!Android_JNI_SetCustomCursor(data->custom_cursor)) {
                return SDL_Unsupported();
            }
        } else {
            if (!Android_JNI_SetSystemCursor(data->system_cursor)) {
                return SDL_Unsupported();
            }
        }
        return true;
    } else {
        // SDL error set inside Android_CreateEmptyCursor()
        return false;
    }
}

static bool Android_SetRelativeMouseMode(bool enabled)
{
    if (!Android_JNI_SupportsRelativeMouse()) {
        return SDL_Unsupported();
    }

    if (!Android_JNI_SetRelativeMouseEnabled(enabled)) {
        return SDL_Unsupported();
    }

    return true;
}

void Android_InitMouse(void)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    mouse->CreateCursor = Android_CreateCursor;
    mouse->CreateSystemCursor = Android_CreateSystemCursor;
    mouse->ShowCursor = Android_ShowCursor;
    mouse->FreeCursor = Android_FreeCursor;
    mouse->SetRelativeMouseMode = Android_SetRelativeMouseMode;

    SDL_SetDefaultCursor(Android_CreateDefaultCursor());

    last_state = 0;
}

void Android_QuitMouse(void)
{
    Android_DestroyEmptyCursor();
}

// Translate Android mouse button state to SDL mouse button
static Uint8 TranslateButton(int state)
{
    if (state & BUTTON_PRIMARY) {
        return SDL_BUTTON_LEFT;
    } else if (state & BUTTON_SECONDARY) {
        return SDL_BUTTON_RIGHT;
    } else if (state & BUTTON_TERTIARY) {
        return SDL_BUTTON_MIDDLE;
    } else if (state & BUTTON_FORWARD) {
        return SDL_BUTTON_X1;
    } else if (state & BUTTON_BACK) {
        return SDL_BUTTON_X2;
    } else {
        return 0;
    }
}

void Android_OnMouse(SDL_Window *window, int state, int action, float x, float y, bool relative)
{
    int changes;
    Uint8 button;

    if (!window) {
        return;
    }

    switch (action) {
    case ACTION_DOWN:
        changes = state & ~last_state;
        button = TranslateButton(changes);
        last_state = state;
        SDL_SendMouseMotion(0, window, SDL_DEFAULT_MOUSE_ID, relative, x, y);
        SDL_SendMouseButton(0, window, SDL_DEFAULT_MOUSE_ID, button, true);
        break;

    case ACTION_UP:
        changes = last_state & ~state;
        button = TranslateButton(changes);
        last_state = state;
        SDL_SendMouseMotion(0, window, SDL_DEFAULT_MOUSE_ID, relative, x, y);
        SDL_SendMouseButton(0, window, SDL_DEFAULT_MOUSE_ID, button, false);
        break;

    case ACTION_MOVE:
    case ACTION_HOVER_MOVE:
        SDL_SendMouseMotion(0, window, SDL_DEFAULT_MOUSE_ID, relative, x, y);
        break;

    case ACTION_SCROLL:
        SDL_SendMouseWheel(0, window, SDL_DEFAULT_MOUSE_ID, x, y, SDL_MOUSEWHEEL_NORMAL);
        break;

    default:
        break;
    }
}

#endif // SDL_VIDEO_DRIVER_ANDROID
