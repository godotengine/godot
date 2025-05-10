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

#ifdef SDL_VIDEO_DRIVER_N3DS

#include "../SDL_sysvideo.h"
#include "../../SDL_properties_c.h"
#include "SDL_n3dsframebuffer_c.h"
#include "SDL_n3dsvideo.h"

#define N3DS_SURFACE "SDL.internal.window.surface"

typedef struct
{
    int width, height;
} Dimensions;

static void CopyFramebuffertoN3DS_16(u16 *dest, const Dimensions dest_dim, const u16 *source, const Dimensions source_dim);
static void CopyFramebuffertoN3DS_24(u8  *dest, const Dimensions dest_dim, const u8  *source, const Dimensions source_dim);
static void CopyFramebuffertoN3DS_32(u32 *dest, const Dimensions dest_dim, const u32 *source, const Dimensions source_dim);
static int GetDestOffset(int x, int y, int dest_width);
static int GetSourceOffset(int x, int y, int source_width);
static void FlushN3DSBuffer(const void *buffer, u32 bufsize, gfxScreen_t screen);


bool SDL_N3DS_CreateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, SDL_PixelFormat *format, void **pixels, int *pitch)
{
    SDL_Surface *framebuffer;
    const SDL_DisplayMode *mode;
    int w, h;

    SDL_N3DS_DestroyWindowFramebuffer(_this, window);

    mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window));
    SDL_GetWindowSizeInPixels(window, &w, &h);
    framebuffer = SDL_CreateSurface(w, h, mode->format);

    if (!framebuffer) {
        return false;
    }

    SDL_SetSurfaceProperty(SDL_GetWindowProperties(window), N3DS_SURFACE, framebuffer);
    *format = mode->format;
    *pixels = framebuffer->pixels;
    *pitch = framebuffer->pitch;
    return true;
}

bool SDL_N3DS_UpdateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_WindowData *drv_data = window->internal;
    SDL_Surface *surface;
    u16 width, height;
    void *framebuffer;
    u32 bufsize;

    surface = (SDL_Surface *)SDL_GetPointerProperty(SDL_GetWindowProperties(window), N3DS_SURFACE, NULL);
    if (!surface) {
        return SDL_SetError("%s: Unable to get the window surface.", __func__);
    }

    // Get the N3DS internal framebuffer and its size
    framebuffer = gfxGetFramebuffer(drv_data->screen, GFX_LEFT, &width, &height);
    bufsize = width * height * 4;

    if (SDL_BYTESPERPIXEL(surface->format) == 2)
        CopyFramebuffertoN3DS_16(framebuffer, (Dimensions){ width, height },
                                 surface->pixels, (Dimensions){ surface->w, surface->h });
    else if (SDL_BYTESPERPIXEL(surface->format) == 3)
        CopyFramebuffertoN3DS_24(framebuffer, (Dimensions){ width, height },
                                 surface->pixels, (Dimensions){ surface->w, surface->h });
    else
        CopyFramebuffertoN3DS_32(framebuffer, (Dimensions){ width, height },
                                 surface->pixels, (Dimensions){ surface->w, surface->h });
    FlushN3DSBuffer(framebuffer, bufsize, drv_data->screen);

    return true;
}

static void CopyFramebuffertoN3DS_16(u16 *dest, const Dimensions dest_dim, const u16 *source, const Dimensions source_dim)
{
    int rows = SDL_min(dest_dim.width, source_dim.height);
    int cols = SDL_min(dest_dim.height, source_dim.width);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const u16 *s = source + GetSourceOffset(x, y, source_dim.width);
            u16 *d = dest + GetDestOffset(x, y, dest_dim.width);
            *d = *s;
        }
    }
}

static void CopyFramebuffertoN3DS_24(u8 *dest, const Dimensions dest_dim, const u8 *source, const Dimensions source_dim)
{
    int rows = SDL_min(dest_dim.width, source_dim.height);
    int cols = SDL_min(dest_dim.height, source_dim.width);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const u8 *s = source + GetSourceOffset(x, y, source_dim.width) * 3;
            u8 *d = dest + GetDestOffset(x, y, dest_dim.width) * 3;
            d[0] = s[0];
            d[1] = s[1];
            d[2] = s[2];
        }
    }
}

static void CopyFramebuffertoN3DS_32(u32 *dest, const Dimensions dest_dim, const u32 *source, const Dimensions source_dim)
{
    int rows = SDL_min(dest_dim.width, source_dim.height);
    int cols = SDL_min(dest_dim.height, source_dim.width);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const u32 *s = source + GetSourceOffset(x, y, source_dim.width);
            u32 *d = dest + GetDestOffset(x, y, dest_dim.width);
            *d = *s;
        }
    }
}

static int GetDestOffset(int x, int y, int dest_width)
{
    return dest_width - y - 1 + dest_width * x;
}

static int GetSourceOffset(int x, int y, int source_width)
{
    return x + y * source_width;
}

static void FlushN3DSBuffer(const void *buffer, u32 bufsize, gfxScreen_t screen)
{
    GSPGPU_FlushDataCache(buffer, bufsize);
    gfxScreenSwapBuffers(screen, false);
}

void SDL_N3DS_DestroyWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_ClearProperty(SDL_GetWindowProperties(window), N3DS_SURFACE);
}

#endif // SDL_VIDEO_DRIVER_N3DS
