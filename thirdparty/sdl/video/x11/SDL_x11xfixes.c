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

#if defined(SDL_VIDEO_DRIVER_X11) && defined(SDL_VIDEO_DRIVER_X11_XFIXES)

#include "SDL_x11video.h"
#include "SDL_x11xfixes.h"
#include "../../events/SDL_mouse_c.h"
#include "../../events/SDL_touch_c.h"

static bool xfixes_initialized = true;
static int xfixes_selection_notify_event = 0;

static int query_xfixes_version(Display *display, int major, int minor)
{
    // We don't care if this fails, so long as it sets major/minor on it's way out the door.
    X11_XFixesQueryVersion(display, &major, &minor);
    return (major * 1000) + minor;
}

static bool xfixes_version_atleast(const int version, const int wantmajor, const int wantminor)
{
    return version >= ((wantmajor * 1000) + wantminor);
}

void X11_InitXfixes(SDL_VideoDevice *_this)
{
    SDL_VideoData *data = _this->internal;

    int version = 0;
    int event, error;
    int fixes_opcode;

    Atom XA_CLIPBOARD = data->atoms.CLIPBOARD;

    if (!SDL_X11_HAVE_XFIXES ||
        !X11_XQueryExtension(data->display, "XFIXES", &fixes_opcode, &event, &error)) {
        return;
    }

    // Selection tracking is available in all versions of XFixes
    xfixes_selection_notify_event = event + XFixesSelectionNotify;
    X11_XFixesSelectSelectionInput(data->display, DefaultRootWindow(data->display),
            XA_CLIPBOARD, XFixesSetSelectionOwnerNotifyMask);
    X11_XFixesSelectSelectionInput(data->display, DefaultRootWindow(data->display),
            XA_PRIMARY, XFixesSetSelectionOwnerNotifyMask);

    // We need at least 5.0 for barriers.
    version = query_xfixes_version(data->display, 5, 0);
    if (!xfixes_version_atleast(version, 5, 0)) {
        return; // X server does not support the version we want at all.
    }

    xfixes_initialized = 1;
}

bool X11_XfixesIsInitialized(void)
{
    return xfixes_initialized;
}

int X11_GetXFixesSelectionNotifyEvent(void)
{
    return xfixes_selection_notify_event;
}

bool X11_SetWindowMouseRect(SDL_VideoDevice *_this, SDL_Window *window)
{
    if (SDL_RectEmpty(&window->mouse_rect)) {
        X11_ConfineCursorWithFlags(_this, window, NULL, 0);
    } else {
        if (window->flags & SDL_WINDOW_INPUT_FOCUS) {
            X11_ConfineCursorWithFlags(_this, window, &window->mouse_rect, 0);
        } else {
            // Save the state for when we get focus again
            SDL_WindowData *wdata = window->internal;

            SDL_memcpy(&wdata->barrier_rect, &window->mouse_rect, sizeof(wdata->barrier_rect));

            wdata->pointer_barrier_active = true;
        }
    }

    return true;
}

bool X11_ConfineCursorWithFlags(SDL_VideoDevice *_this, SDL_Window *window, const SDL_Rect *rect, int flags)
{
    /* Yaakuro: For some reason Xfixes when confining inside a rect where the
     * edges exactly match, a rectangle the cursor 'slips' out of the barrier.
     * To prevent that the lines for the barriers will span the whole screen.
     */
    SDL_VideoData *data = _this->internal;
    SDL_WindowData *wdata;

    if (!X11_XfixesIsInitialized()) {
        return SDL_Unsupported();
    }

    // If there is already a set of barriers active, disable them.
    if (data->active_cursor_confined_window) {
        X11_DestroyPointerBarrier(_this, data->active_cursor_confined_window);
    }

    SDL_assert(window != NULL);
    wdata = window->internal;

    /* If user did not specify an area to confine, destroy the barrier that was/is assigned to
     * this window it was assigned */
    if (rect) {
        int x1, y1, x2, y2;
        SDL_Rect bounds;
        SDL_GetWindowPosition(window, &bounds.x, &bounds.y);
        SDL_GetWindowSize(window, &bounds.w, &bounds.h);

        /** Negative values are not allowed. Clip values relative to the specified window. */
        x1 = bounds.x + SDL_max(rect->x, 0);
        y1 = bounds.y + SDL_max(rect->y, 0);
        x2 = SDL_min(bounds.x + rect->x + rect->w, bounds.x + bounds.w);
        y2 = SDL_min(bounds.y + rect->y + rect->h, bounds.y + bounds.h);

        if ((wdata->barrier_rect.x != rect->x) ||
            (wdata->barrier_rect.y != rect->y) ||
            (wdata->barrier_rect.w != rect->w) ||
            (wdata->barrier_rect.h != rect->h)) {
            wdata->barrier_rect = *rect;
        }

        // Use the display bounds to ensure the barriers don't have corner gaps
        SDL_GetDisplayBounds(SDL_GetDisplayForWindow(window), &bounds);

        /** Create the left barrier */
        wdata->barrier[0] = X11_XFixesCreatePointerBarrier(data->display, wdata->xwindow,
                                                           x1, bounds.y,
                                                           x1, bounds.y + bounds.h,
                                                           BarrierPositiveX,
                                                           0, NULL);
        /** Create the right barrier */
        wdata->barrier[1] = X11_XFixesCreatePointerBarrier(data->display, wdata->xwindow,
                                                           x2, bounds.y,
                                                           x2, bounds.y + bounds.h,
                                                           BarrierNegativeX,
                                                           0, NULL);
        /** Create the top barrier */
        wdata->barrier[2] = X11_XFixesCreatePointerBarrier(data->display, wdata->xwindow,
                                                           bounds.x, y1,
                                                           bounds.x + bounds.w, y1,
                                                           BarrierPositiveY,
                                                           0, NULL);
        /** Create the bottom barrier */
        wdata->barrier[3] = X11_XFixesCreatePointerBarrier(data->display, wdata->xwindow,
                                                           bounds.x, y2,
                                                           bounds.x + bounds.w, y2,
                                                           BarrierNegativeY,
                                                           0, NULL);

        X11_XFlush(data->display);

        // Lets remember current active confined window.
        data->active_cursor_confined_window = window;

        /* User activated the confinement for this window. We use this later to reactivate
         * the confinement if it got deactivated by FocusOut or UnmapNotify */
        wdata->pointer_barrier_active = true;
    } else {
        X11_DestroyPointerBarrier(_this, window);

        // Only set barrier inactive when user specified NULL and not handled by focus out.
        if (flags != X11_BARRIER_HANDLED_BY_EVENT) {
            wdata->pointer_barrier_active = false;
        }
    }
    return true;
}

void X11_DestroyPointerBarrier(SDL_VideoDevice *_this, SDL_Window *window)
{
    int i;
    SDL_VideoData *data = _this->internal;
    if (window) {
        SDL_WindowData *wdata = window->internal;

        for (i = 0; i < 4; i++) {
            if (wdata->barrier[i] > 0) {
                X11_XFixesDestroyPointerBarrier(data->display, wdata->barrier[i]);
                wdata->barrier[i] = 0;
            }
        }
        X11_XFlush(data->display);
    }
    data->active_cursor_confined_window = NULL;
}

#endif // SDL_VIDEO_DRIVER_X11 && SDL_VIDEO_DRIVER_X11_XFIXES
