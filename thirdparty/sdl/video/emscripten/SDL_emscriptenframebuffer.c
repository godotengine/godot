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

#include "SDL_emscriptenvideo.h"
#include "SDL_emscriptenframebuffer.h"

#include <emscripten/threading.h>

bool Emscripten_CreateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, SDL_PixelFormat *format, void **pixels, int *pitch)
{
    SDL_Surface *surface;
    const SDL_PixelFormat surface_format = SDL_PIXELFORMAT_XBGR8888;
    int w, h;

    // Free the old framebuffer surface
    SDL_WindowData *data = window->internal;
    surface = data->surface;
    SDL_DestroySurface(surface);

    // Create a new one
    SDL_GetWindowSizeInPixels(window, &w, &h);

    surface = SDL_CreateSurface(w, h, surface_format);
    if (!surface) {
        return false;
    }

    // Save the info and return!
    data->surface = surface;
    *format = surface_format;
    *pixels = surface->pixels;
    *pitch = surface->pitch;
    return true;
}

bool Emscripten_UpdateWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window, const SDL_Rect *rects, int numrects)
{
    SDL_Surface *surface;

    SDL_WindowData *data = window->internal;
    surface = data->surface;
    if (!surface) {
        return SDL_SetError("Couldn't find framebuffer surface for window");
    }

    // Send the data to the display

    /* *INDENT-OFF* */ // clang-format off
    MAIN_THREAD_EM_ASM({
        var w = $0;
        var h = $1;
        var pixels = $2;
        var canvasId = UTF8ToString($3);
        var canvas = document.querySelector(canvasId);

        //TODO: this should store a context per canvas
        if (!Module['SDL3']) Module['SDL3'] = {};
        var SDL3 = Module['SDL3'];
        if (SDL3.ctxCanvas !== canvas) {
            SDL3.ctx = Module['createContext'](canvas, false, true);
            SDL3.ctxCanvas = canvas;
        }
        if (SDL3.w !== w || SDL3.h !== h || SDL3.imageCtx !== SDL3.ctx) {
            SDL3.image = SDL3.ctx.createImageData(w, h);
            SDL3.w = w;
            SDL3.h = h;
            SDL3.imageCtx = SDL3.ctx;
        }
        var data = SDL3.image.data;
        var src = pixels / 4;
        var dst = 0;
        var num;

        if (SDL3.data32Data !== data) {
            SDL3.data32 = new Int32Array(data.buffer);
            SDL3.data8 = new Uint8Array(data.buffer);
            SDL3.data32Data = data;
        }
        var data32 = SDL3.data32;
        num = data32.length;
        // logically we need to do
        //      while (dst < num) {
        //          data32[dst++] = HEAP32[src++] | 0xff000000
        //      }
        // the following code is faster though, because
        // .set() is almost free - easily 10x faster due to
        // native SDL_memcpy efficiencies, and the remaining loop
        // just stores, not load + store, so it is faster
        data32.set(HEAP32.subarray(src, src + num));
        var data8 = SDL3.data8;
        var i = 3;
        var j = i + 4*num;
        if (num % 8 == 0) {
            // unrolling gives big speedups
            while (i < j) {
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
              data8[i] = 0xff;
              i = i + 4 | 0;
            }
         } else {
            while (i < j) {
              data8[i] = 0xff;
              i = i + 4 | 0;
            }
        }

        SDL3.ctx.putImageData(SDL3.image, 0, 0);
    }, surface->w, surface->h, surface->pixels, data->canvas_id);
    /* *INDENT-ON* */ // clang-format on

    if (emscripten_has_asyncify() && SDL_GetHintBoolean(SDL_HINT_EMSCRIPTEN_ASYNCIFY, true)) {
        // give back control to browser for screen refresh
        emscripten_sleep(0);
    }

    return true;
}

void Emscripten_DestroyWindowFramebuffer(SDL_VideoDevice *_this, SDL_Window *window)
{
    SDL_WindowData *data = window->internal;

    SDL_DestroySurface(data->surface);
    data->surface = NULL;
}

#endif // SDL_VIDEO_DRIVER_EMSCRIPTEN
