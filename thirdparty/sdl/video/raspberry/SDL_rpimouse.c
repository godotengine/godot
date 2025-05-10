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

#ifdef SDL_VIDEO_DRIVER_RPI

#include "SDL_rpivideo.h"
#include "SDL_rpimouse.h"

#include "../SDL_sysvideo.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/default_cursor.h"

// Copied from vc_vchi_dispmanx.h which is bugged and tries to include a non existing file
// Attributes changes flag mask
#define ELEMENT_CHANGE_LAYER         (1 << 0)
#define ELEMENT_CHANGE_OPACITY       (1 << 1)
#define ELEMENT_CHANGE_DEST_RECT     (1 << 2)
#define ELEMENT_CHANGE_SRC_RECT      (1 << 3)
#define ELEMENT_CHANGE_MASK_RESOURCE (1 << 4)
#define ELEMENT_CHANGE_TRANSFORM     (1 << 5)
// End copied from vc_vchi_dispmanx.h

static SDL_Cursor *RPI_CreateDefaultCursor(void);
static SDL_Cursor *RPI_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y);
static bool RPI_ShowCursor(SDL_Cursor *cursor);
static bool RPI_MoveCursor(SDL_Cursor *cursor);
static void RPI_FreeCursor(SDL_Cursor *cursor);

static SDL_Cursor *global_cursor;

static SDL_Cursor *RPI_CreateDefaultCursor(void)
{
    return SDL_CreateCursor(default_cdata, default_cmask, DEFAULT_CWIDTH, DEFAULT_CHEIGHT, DEFAULT_CHOTX, DEFAULT_CHOTY);
}

// Create a cursor from a surface
static SDL_Cursor *RPI_CreateCursor(SDL_Surface *surface, int hot_x, int hot_y)
{
    int rc;
    SDL_CursorData *curdata;
    SDL_Cursor *cursor;
    VC_RECT_T dst_rect;
    Uint32 dummy;

    SDL_assert(surface->format == SDL_PIXELFORMAT_ARGB8888);
    SDL_assert(surface->pitch == surface->w * 4);

    cursor = (SDL_Cursor *)SDL_calloc(1, sizeof(*cursor));
    if (!cursor) {
        return NULL;
    }
    curdata = (SDL_CursorData *)SDL_calloc(1, sizeof(*curdata));
    if (!curdata) {
        SDL_free(cursor);
        return NULL;
    }

    curdata->hot_x = hot_x;
    curdata->hot_y = hot_y;
    curdata->w = surface->w;
    curdata->h = surface->h;

    // This usage is inspired by Wayland/Weston RPI code, how they figured this out is anyone's guess
    curdata->resource = vc_dispmanx_resource_create(VC_IMAGE_ARGB8888, surface->w | (surface->pitch << 16), surface->h | (surface->h << 16), &dummy);
    SDL_assert(curdata->resource);
    vc_dispmanx_rect_set(&dst_rect, 0, 0, curdata->w, curdata->h);
    /* A note from Weston:
     * vc_dispmanx_resource_write_data() ignores ifmt,
     * rect.x, rect.width, and uses stride only for computing
     * the size of the transfer as rect.height * stride.
     * Therefore we can only write rows starting at x=0.
     */
    rc = vc_dispmanx_resource_write_data(curdata->resource, VC_IMAGE_ARGB8888, surface->pitch, surface->pixels, &dst_rect);
    SDL_assert(rc == DISPMANX_SUCCESS);

    cursor->internal = curdata;

    return cursor;
}

// Show the specified cursor, or hide if cursor is NULL
static bool RPI_ShowCursor(SDL_Cursor *cursor)
{
    int rc;
    DISPMANX_UPDATE_HANDLE_T update;
    SDL_CursorData *curdata;
    VC_RECT_T src_rect, dst_rect;
    SDL_Mouse *mouse = SDL_GetMouse();
    SDL_DisplayData *data;
    VC_DISPMANX_ALPHA_T alpha = { DISPMANX_FLAGS_ALPHA_FROM_SOURCE /* flags */, 255 /*opacity 0->255*/, 0 /* mask */ };
    uint32_t layer = SDL_RPI_MOUSELAYER;
    const char *hint;

    if (cursor != global_cursor) {
        if (global_cursor) {
            curdata = global_cursor->internal;
            if (curdata && curdata->element > DISPMANX_NO_HANDLE) {
                update = vc_dispmanx_update_start(0);
                SDL_assert(update);
                rc = vc_dispmanx_element_remove(update, curdata->element);
                SDL_assert(rc == DISPMANX_SUCCESS);
                rc = vc_dispmanx_update_submit_sync(update);
                SDL_assert(rc == DISPMANX_SUCCESS);
                curdata->element = DISPMANX_NO_HANDLE;
            }
        }
        global_cursor = cursor;
    }

    if (!cursor) {
        return true;
    }

    curdata = cursor->internal;
    if (!curdata) {
        return false;
    }

    if (!mouse->focus) {
        return false;
    }

    data = SDL_GetDisplayDriverDataForWindow(mouse->focus);
    if (!data) {
        return false;
    }

    if (curdata->element == DISPMANX_NO_HANDLE) {
        vc_dispmanx_rect_set(&src_rect, 0, 0, curdata->w << 16, curdata->h << 16);
        vc_dispmanx_rect_set(&dst_rect, mouse->x - curdata->hot_x, mouse->y - curdata->hot_y, curdata->w, curdata->h);

        update = vc_dispmanx_update_start(0);
        SDL_assert(update);

        hint = SDL_GetHint(SDL_HINT_RPI_VIDEO_LAYER);
        if (hint) {
            layer = SDL_atoi(hint) + 1;
        }

        curdata->element = vc_dispmanx_element_add(update,
                                                   data->dispman_display,
                                                   layer,
                                                   &dst_rect,
                                                   curdata->resource,
                                                   &src_rect,
                                                   DISPMANX_PROTECTION_NONE,
                                                   &alpha,
                                                   DISPMANX_NO_HANDLE, // clamp
                                                   DISPMANX_NO_ROTATE);
        SDL_assert(curdata->element > DISPMANX_NO_HANDLE);
        rc = vc_dispmanx_update_submit_sync(update);
        SDL_assert(rc == DISPMANX_SUCCESS);
    }

    return true;
}

// Free a window manager cursor
static void RPI_FreeCursor(SDL_Cursor *cursor)
{
    int rc;
    DISPMANX_UPDATE_HANDLE_T update;
    SDL_CursorData *curdata;

    if (cursor) {
        curdata = cursor->internal;

        if (curdata) {
            if (curdata->element != DISPMANX_NO_HANDLE) {
                update = vc_dispmanx_update_start(0);
                SDL_assert(update);
                rc = vc_dispmanx_element_remove(update, curdata->element);
                SDL_assert(rc == DISPMANX_SUCCESS);
                rc = vc_dispmanx_update_submit_sync(update);
                SDL_assert(rc == DISPMANX_SUCCESS);
            }

            if (curdata->resource != DISPMANX_NO_HANDLE) {
                rc = vc_dispmanx_resource_delete(curdata->resource);
                SDL_assert(rc == DISPMANX_SUCCESS);
            }

            SDL_free(cursor->internal);
        }
        SDL_free(cursor);
        if (cursor == global_cursor) {
            global_cursor = NULL;
        }
    }
}

static bool RPI_WarpMouseGlobalGraphically(float x, float y)
{
    int rc;
    SDL_CursorData *curdata;
    DISPMANX_UPDATE_HANDLE_T update;
    VC_RECT_T dst_rect;
    VC_RECT_T src_rect;
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse || !mouse->cur_cursor || !mouse->cur_cursor->internal) {
        return true;
    }

    curdata = mouse->cur_cursor->internal;
    if (curdata->element == DISPMANX_NO_HANDLE) {
        return true;
    }

    update = vc_dispmanx_update_start(0);
    if (!update) {
        return true;
    }

    src_rect.x = 0;
    src_rect.y = 0;
    src_rect.width = curdata->w << 16;
    src_rect.height = curdata->h << 16;
    dst_rect.x = (int)x - curdata->hot_x;
    dst_rect.y = (int)y - curdata->hot_y;
    dst_rect.width = curdata->w;
    dst_rect.height = curdata->h;

    rc = vc_dispmanx_element_change_attributes(
        update,
        curdata->element,
        0,
        0,
        0,
        &dst_rect,
        &src_rect,
        DISPMANX_NO_HANDLE,
        DISPMANX_NO_ROTATE);
    if (rc != DISPMANX_SUCCESS) {
        return SDL_SetError("vc_dispmanx_element_change_attributes() failed");
    }

    // Submit asynchronously, otherwise the performance suffers a lot
    rc = vc_dispmanx_update_submit(update, 0, NULL);
    if (rc != DISPMANX_SUCCESS) {
        return SDL_SetError("vc_dispmanx_update_submit() failed");
    }
    return true;
}

static bool RPI_WarpMouseGlobal(float x, float y)
{
    SDL_Mouse *mouse = SDL_GetMouse();

    if (!mouse || !mouse->cur_cursor || !mouse->cur_cursor->internal) {
        return true;
    }

    // Update internal mouse position.
    SDL_SendMouseMotion(0, mouse->focus, SDL_GLOBAL_MOUSE_ID, false, x, y);

    return RPI_WarpMouseGlobalGraphically(x, y);
}

static bool RPI_WarpMouse(SDL_Window *window, float x, float y)
{
    return RPI_WarpMouseGlobal(x, y);
}

void RPI_InitMouse(SDL_VideoDevice *_this)
{
    /* FIXME: Using UDEV it should be possible to scan all mice
     * but there's no point in doing so as there's no multimice support...yet!
     */
    SDL_Mouse *mouse = SDL_GetMouse();

    mouse->CreateCursor = RPI_CreateCursor;
    mouse->ShowCursor = RPI_ShowCursor;
    mouse->MoveCursor = RPI_MoveCursor;
    mouse->FreeCursor = RPI_FreeCursor;
    mouse->WarpMouse = RPI_WarpMouse;
    mouse->WarpMouseGlobal = RPI_WarpMouseGlobal;

    SDL_SetDefaultCursor(RPI_CreateDefaultCursor());
}

void RPI_QuitMouse(SDL_VideoDevice *_this)
{
}

// This is called when a mouse motion event occurs
static bool RPI_MoveCursor(SDL_Cursor *cursor)
{
    SDL_Mouse *mouse = SDL_GetMouse();
    /* We must NOT call SDL_SendMouseMotion() on the next call or we will enter recursivity,
     * so we create a version of WarpMouseGlobal without it. */
    return RPI_WarpMouseGlobalGraphically(mouse->x, mouse->y);
}

#endif // SDL_VIDEO_DRIVER_RPI
