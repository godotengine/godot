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

#include "SDL_x11video.h"
#include "SDL_x11framebuffer.h"
#include "SDL_x11xsync.h"

#ifndef NO_SHARED_MEMORY

// Shared memory error handler routine
static int shm_error;
static int (*X_handler)(Display *, XErrorEvent *) = NULL;
static int shm_errhandler(Display *d, XErrorEvent *e)
{
    if (e->error_code == BadAccess) {
        shm_error = True;
        return 0;
    }
    return X_handler(d, e);
}

static bool have_mitshm(Display *dpy)
{
    // Only use shared memory on local X servers
    return X11_XShmQueryExtension(dpy) ? SDL_X11_HAVE_SHM : false;
}

#endif // !NO_SHARED_MEMORY

bool X11_CreateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, SDL_PixelFormat *format,
                                void **pixels, int *pitch)
{
    SDL_WindowData *data = window->internal;
    Display *display = data->videodata->display;
    XGCValues gcv;
    XVisualInfo vinfo;
    int w, h;

    SDL_GetWindowSizeInPixels(window, &w, &h);

    // Free the old framebuffer surface
    X11_DestroyWindowFramebuffer(_this, window);

    // Create the graphics context for drawing
    gcv.graphics_exposures = False;
    data->gc = X11_XCreateGC(display, data->xwindow, GCGraphicsExposures, &gcv);
    if (!data->gc) {
        return SDL_SetError("Couldn't create graphics context");
    }

    // Find out the pixel format and depth
    if (!X11_GetVisualInfoFromVisual(display, data->visual, &vinfo)) {
        return SDL_SetError("Couldn't get window visual information");
    }

    *format = X11_GetPixelFormatFromVisualInfo(display, &vinfo);
    if (*format == SDL_PIXELFORMAT_UNKNOWN) {
        return SDL_SetError("Unknown window pixel format");
    }

    // Calculate pitch
    *pitch = (((w * SDL_BYTESPERPIXEL(*format)) + 3) & ~3);

    // Create the actual image
#ifndef NO_SHARED_MEMORY
    if (have_mitshm(display)) {
        XShmSegmentInfo *shminfo = &data->shminfo;

        shminfo->shmid = shmget(IPC_PRIVATE, (size_t)h * (*pitch), IPC_CREAT | 0777);
        if (shminfo->shmid >= 0) {
            shminfo->shmaddr = (char *)shmat(shminfo->shmid, 0, 0);
            shminfo->readOnly = False;
            if (shminfo->shmaddr != (char *)-1) {
                shm_error = False;
                X_handler = X11_XSetErrorHandler(shm_errhandler);
                X11_XShmAttach(display, shminfo);
                X11_XSync(display, False);
                X11_XSetErrorHandler(X_handler);
                if (shm_error) {
                    shmdt(shminfo->shmaddr);
                }
            } else {
                shm_error = True;
            }
            shmctl(shminfo->shmid, IPC_RMID, NULL);
        } else {
            shm_error = True;
        }
        if (!shm_error) {
            data->ximage = X11_XShmCreateImage(display, data->visual,
                                               vinfo.depth, ZPixmap,
                                               shminfo->shmaddr, shminfo,
                                               w, h);
            if (!data->ximage) {
                X11_XShmDetach(display, shminfo);
                X11_XSync(display, False);
                shmdt(shminfo->shmaddr);
            } else {
                // Done!
                data->ximage->byte_order = (SDL_BYTEORDER == SDL_BIG_ENDIAN) ? MSBFirst : LSBFirst;
                data->use_mitshm = true;
                *pixels = shminfo->shmaddr;
                return true;
            }
        }
    }
#endif // not NO_SHARED_MEMORY

    *pixels = SDL_malloc((size_t)h * (*pitch));
    if (!*pixels) {
        return false;
    }

    data->ximage = X11_XCreateImage(display, data->visual,
                                    vinfo.depth, ZPixmap, 0, (char *)(*pixels),
                                    w, h, 32, 0);
    if (!data->ximage) {
        SDL_free(*pixels);
        return SDL_SetError("Couldn't create XImage");
    }
    data->ximage->byte_order = (SDL_BYTEORDER == SDL_BIG_ENDIAN) ? MSBFirst : LSBFirst;
    return true;
}

bool X11_UpdateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, const SDL_Rect *rects,
                                int numrects)
{
    SDL_WindowData *data = window->internal;
    Display *display = data->videodata->display;
    int i;
    int x, y, w, h;
    int window_w, window_h;

    SDL_GetWindowSizeInPixels(window, &window_w, &window_h);

#ifndef NO_SHARED_MEMORY
    if (data->use_mitshm) {
        for (i = 0; i < numrects; ++i) {
            x = rects[i].x;
            y = rects[i].y;
            w = rects[i].w;
            h = rects[i].h;

            if (w <= 0 || h <= 0 || (x + w) <= 0 || (y + h) <= 0) {
                // Clipped?
                continue;
            }
            if (x < 0) {
                x += w;
                w += rects[i].x;
            }
            if (y < 0) {
                y += h;
                h += rects[i].y;
            }
            if (x + w > window_w) {
                w = window_w - x;
            }
            if (y + h > window_h) {
                h = window_h - y;
            }

            X11_XShmPutImage(display, data->xwindow, data->gc, data->ximage,
                             x, y, x, y, w, h, False);
        }
    } else
#endif // !NO_SHARED_MEMORY
    {
        for (i = 0; i < numrects; ++i) {
            x = rects[i].x;
            y = rects[i].y;
            w = rects[i].w;
            h = rects[i].h;

            if (w <= 0 || h <= 0 || (x + w) <= 0 || (y + h) <= 0) {
                // Clipped?
                continue;
            }
            if (x < 0) {
                x += w;
                w += rects[i].x;
            }
            if (y < 0) {
                y += h;
                h += rects[i].y;
            }
            if (x + w > window_w) {
                w = window_w - x;
            }
            if (y + h > window_h) {
                h = window_h - y;
            }

            X11_XPutImage(display, data->xwindow, data->gc, data->ximage,
                          x, y, x, y, w, h);
        }
    }

#ifdef SDL_VIDEO_DRIVER_X11_XSYNC
    X11_HandlePresent(data->window);
#endif /* SDL_VIDEO_DRIVER_X11_XSYNC */

    X11_XSync(display, False);

    return true;
}

void X11_DestroyWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;
    Display *display;

    if (!data) {
        // The window wasn't fully initialized
        return;
    }

    display = data->videodata->display;

    if (data->ximage) {
        XDestroyImage(data->ximage);

#ifndef NO_SHARED_MEMORY
        if (data->use_mitshm) {
            X11_XShmDetach(display, &data->shminfo);
            X11_XSync(display, False);
            shmdt(data->shminfo.shmaddr);
            data->use_mitshm = false;
        }
#endif // !NO_SHARED_MEMORY

        data->ximage = NULL;
    }
    if (data->gc) {
        X11_XFreeGC(display, data->gc);
        data->gc = NULL;
    }
}

#endif // SDL_VIDEO_DRIVER_X11
