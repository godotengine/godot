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

// This is the software implementation of the YUV texture support

#ifdef SDL_HAVE_YUV

#include "SDL_yuv_sw_c.h"
#include "../video/SDL_surface_c.h"
#include "../video/SDL_yuv_c.h"

SDL_SW_YUVTexture *SDL_SW_CreateYUVTexture(SDL_PixelFormat format, SDL_Colorspace colorspace, int w, int h)
{
    SDL_SW_YUVTexture *swdata;

    switch (format) {
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_YUY2:
    case SDL_PIXELFORMAT_UYVY:
    case SDL_PIXELFORMAT_YVYU:
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        break;
    default:
        SDL_SetError("Unsupported YUV format");
        return NULL;
    }

    swdata = (SDL_SW_YUVTexture *)SDL_calloc(1, sizeof(*swdata));
    if (!swdata) {
        return NULL;
    }

    swdata->format = format;
    swdata->colorspace = colorspace;
    swdata->target_format = SDL_PIXELFORMAT_UNKNOWN;
    swdata->w = w;
    swdata->h = h;
    {
        size_t dst_size;
        if (!SDL_CalculateYUVSize(format, w, h, &dst_size, NULL)) {
            SDL_SW_DestroyYUVTexture(swdata);
            return NULL;
        }
        swdata->pixels = (Uint8 *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), dst_size);
        if (!swdata->pixels) {
            SDL_SW_DestroyYUVTexture(swdata);
            return NULL;
        }
    }

    // Find the pitch and offset values for the texture
    switch (format) {
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
        swdata->pitches[0] = w;
        swdata->pitches[1] = (swdata->pitches[0] + 1) / 2;
        swdata->pitches[2] = (swdata->pitches[0] + 1) / 2;
        swdata->planes[0] = swdata->pixels;
        swdata->planes[1] = swdata->planes[0] + swdata->pitches[0] * h;
        swdata->planes[2] = swdata->planes[1] + swdata->pitches[1] * ((h + 1) / 2);
        break;
    case SDL_PIXELFORMAT_YUY2:
    case SDL_PIXELFORMAT_UYVY:
    case SDL_PIXELFORMAT_YVYU:
        swdata->pitches[0] = ((w + 1) / 2) * 4;
        swdata->planes[0] = swdata->pixels;
        break;

    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        swdata->pitches[0] = w;
        swdata->pitches[1] = 2 * ((swdata->pitches[0] + 1) / 2);
        swdata->planes[0] = swdata->pixels;
        swdata->planes[1] = swdata->planes[0] + swdata->pitches[0] * h;
        break;

    default:
        SDL_assert(!"We should never get here (caught above)");
        break;
    }

    // We're all done..
    return swdata;
}

bool SDL_SW_QueryYUVTexturePixels(SDL_SW_YUVTexture *swdata, void **pixels,
                                 int *pitch)
{
    *pixels = swdata->planes[0];
    *pitch = swdata->pitches[0];
    return true;
}

bool SDL_SW_UpdateYUVTexture(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect,
                            const void *pixels, int pitch)
{
    switch (swdata->format) {
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
        if (rect->x == 0 && rect->y == 0 &&
            rect->w == swdata->w && rect->h == swdata->h) {
            SDL_memcpy(swdata->pixels, pixels,
                       (size_t)(swdata->h * swdata->w) + 2 * ((swdata->h + 1) / 2) * ((swdata->w + 1) / 2));
        } else {
            Uint8 *src, *dst;
            int row;
            size_t length;

            // Copy the Y plane
            src = (Uint8 *)pixels;
            dst = swdata->pixels + rect->y * swdata->w + rect->x;
            length = rect->w;
            for (row = 0; row < rect->h; ++row) {
                SDL_memcpy(dst, src, length);
                src += pitch;
                dst += swdata->w;
            }

            // Copy the next plane
            src = (Uint8 *)pixels + rect->h * pitch;
            dst = swdata->pixels + swdata->h * swdata->w;
            dst += rect->y / 2 * ((swdata->w + 1) / 2) + rect->x / 2;
            length = (rect->w + 1) / 2;
            for (row = 0; row < (rect->h + 1) / 2; ++row) {
                SDL_memcpy(dst, src, length);
                src += (pitch + 1) / 2;
                dst += (swdata->w + 1) / 2;
            }

            // Copy the next plane
            src = (Uint8 *)pixels + rect->h * pitch + ((rect->h + 1) / 2) * ((pitch + 1) / 2);
            dst = swdata->pixels + swdata->h * swdata->w +
                  ((swdata->h + 1) / 2) * ((swdata->w + 1) / 2);
            dst += rect->y / 2 * ((swdata->w + 1) / 2) + rect->x / 2;
            length = (rect->w + 1) / 2;
            for (row = 0; row < (rect->h + 1) / 2; ++row) {
                SDL_memcpy(dst, src, length);
                src += (pitch + 1) / 2;
                dst += (swdata->w + 1) / 2;
            }
        }
        break;
    case SDL_PIXELFORMAT_YUY2:
    case SDL_PIXELFORMAT_UYVY:
    case SDL_PIXELFORMAT_YVYU:
    {
        Uint8 *src, *dst;
        int row;
        size_t length;

        src = (Uint8 *)pixels;
        dst =
            swdata->planes[0] + rect->y * swdata->pitches[0] +
            rect->x * 2;
        length = 4 * (((size_t)rect->w + 1) / 2);
        for (row = 0; row < rect->h; ++row) {
            SDL_memcpy(dst, src, length);
            src += pitch;
            dst += swdata->pitches[0];
        }
    } break;
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
    {
        if (rect->x == 0 && rect->y == 0 && rect->w == swdata->w && rect->h == swdata->h) {
            SDL_memcpy(swdata->pixels, pixels,
                       (size_t)(swdata->h * swdata->w) + 2 * ((swdata->h + 1) / 2) * ((swdata->w + 1) / 2));
        } else {

            Uint8 *src, *dst;
            int row;
            size_t length;

            // Copy the Y plane
            src = (Uint8 *)pixels;
            dst = swdata->pixels + rect->y * swdata->w + rect->x;
            length = rect->w;
            for (row = 0; row < rect->h; ++row) {
                SDL_memcpy(dst, src, length);
                src += pitch;
                dst += swdata->w;
            }

            // Copy the next plane
            src = (Uint8 *)pixels + rect->h * pitch;
            dst = swdata->pixels + swdata->h * swdata->w;
            dst += 2 * ((rect->y + 1) / 2) * ((swdata->w + 1) / 2) + 2 * (rect->x / 2);
            length = 2 * (((size_t)rect->w + 1) / 2);
            for (row = 0; row < (rect->h + 1) / 2; ++row) {
                SDL_memcpy(dst, src, length);
                src += 2 * ((pitch + 1) / 2);
                dst += 2 * ((swdata->w + 1) / 2);
            }
        }
    } break;
    default:
        return SDL_SetError("Unsupported YUV format");
    }
    return true;
}

bool SDL_SW_UpdateYUVTexturePlanar(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect,
                                  const Uint8 *Yplane, int Ypitch,
                                  const Uint8 *Uplane, int Upitch,
                                  const Uint8 *Vplane, int Vpitch)
{
    const Uint8 *src;
    Uint8 *dst;
    int row;
    size_t length;

    // Copy the Y plane
    src = Yplane;
    dst = swdata->pixels + rect->y * swdata->w + rect->x;
    length = rect->w;
    for (row = 0; row < rect->h; ++row) {
        SDL_memcpy(dst, src, length);
        src += Ypitch;
        dst += swdata->w;
    }

    // Copy the U plane
    src = Uplane;
    if (swdata->format == SDL_PIXELFORMAT_IYUV) {
        dst = swdata->pixels + swdata->h * swdata->w;
    } else {
        dst = swdata->pixels + swdata->h * swdata->w +
              ((swdata->h + 1) / 2) * ((swdata->w + 1) / 2);
    }
    dst += rect->y / 2 * ((swdata->w + 1) / 2) + rect->x / 2;
    length = (rect->w + 1) / 2;
    for (row = 0; row < (rect->h + 1) / 2; ++row) {
        SDL_memcpy(dst, src, length);
        src += Upitch;
        dst += (swdata->w + 1) / 2;
    }

    // Copy the V plane
    src = Vplane;
    if (swdata->format == SDL_PIXELFORMAT_YV12) {
        dst = swdata->pixels + swdata->h * swdata->w;
    } else {
        dst = swdata->pixels + swdata->h * swdata->w +
              ((swdata->h + 1) / 2) * ((swdata->w + 1) / 2);
    }
    dst += rect->y / 2 * ((swdata->w + 1) / 2) + rect->x / 2;
    length = (rect->w + 1) / 2;
    for (row = 0; row < (rect->h + 1) / 2; ++row) {
        SDL_memcpy(dst, src, length);
        src += Vpitch;
        dst += (swdata->w + 1) / 2;
    }
    return true;
}

bool SDL_SW_UpdateNVTexturePlanar(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect,
                                 const Uint8 *Yplane, int Ypitch,
                                 const Uint8 *UVplane, int UVpitch)
{
    const Uint8 *src;
    Uint8 *dst;
    int row;
    size_t length;

    // Copy the Y plane
    src = Yplane;
    dst = swdata->pixels + rect->y * swdata->w + rect->x;
    length = rect->w;
    for (row = 0; row < rect->h; ++row) {
        SDL_memcpy(dst, src, length);
        src += Ypitch;
        dst += swdata->w;
    }

    // Copy the UV or VU plane
    src = UVplane;
    dst = swdata->pixels + swdata->h * swdata->w;
    dst += rect->y * ((swdata->w + 1) / 2) + rect->x;
    length = (rect->w + 1) / 2;
    length *= 2;
    for (row = 0; row < (rect->h + 1) / 2; ++row) {
        SDL_memcpy(dst, src, length);
        src += UVpitch;
        dst += 2 * ((swdata->w + 1) / 2);
    }

    return true;
}

bool SDL_SW_LockYUVTexture(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect,
                          void **pixels, int *pitch)
{
    switch (swdata->format) {
    case SDL_PIXELFORMAT_YV12:
    case SDL_PIXELFORMAT_IYUV:
    case SDL_PIXELFORMAT_NV12:
    case SDL_PIXELFORMAT_NV21:
        if (rect && (rect->x != 0 || rect->y != 0 || rect->w != swdata->w || rect->h != swdata->h)) {
            return SDL_SetError("YV12, IYUV, NV12, NV21 textures only support full surface locks");
        }
        break;
    default:
        return SDL_SetError("Unsupported YUV format");
    }

    if (rect) {
        *pixels = swdata->planes[0] + rect->y * swdata->pitches[0] + rect->x * 2;
    } else {
        *pixels = swdata->planes[0];
    }
    *pitch = swdata->pitches[0];
    return true;
}

void SDL_SW_UnlockYUVTexture(SDL_SW_YUVTexture *swdata)
{
}

bool SDL_SW_CopyYUVToRGB(SDL_SW_YUVTexture *swdata, const SDL_Rect *srcrect, SDL_PixelFormat target_format, int w, int h, void *pixels, int pitch)
{
    int stretch;

    // Make sure we're set up to display in the desired format
    if (target_format != swdata->target_format && swdata->display) {
        SDL_DestroySurface(swdata->display);
        swdata->display = NULL;
    }

    stretch = 0;
    if (srcrect->x || srcrect->y || srcrect->w < swdata->w || srcrect->h < swdata->h) {
        /* The source rectangle has been clipped.
           Using a scratch surface is easier than adding clipped
           source support to all the blitters, plus that would
           slow them down in the general unclipped case.
         */
        stretch = 1;
    } else if ((srcrect->w != w) || (srcrect->h != h)) {
        stretch = 1;
    }
    if (stretch) {
        if (swdata->display) {
            swdata->display->w = w;
            swdata->display->h = h;
            swdata->display->pixels = pixels;
            swdata->display->pitch = pitch;
        } else {
            swdata->display = SDL_CreateSurfaceFrom(w, h, target_format, pixels, pitch);
            if (!swdata->display) {
                return false;
            }
            swdata->target_format = target_format;
        }
        if (!swdata->stretch) {
            swdata->stretch = SDL_CreateSurface(swdata->w, swdata->h, target_format);
            if (!swdata->stretch) {
                return false;
            }
        }
        pixels = swdata->stretch->pixels;
        pitch = swdata->stretch->pitch;
    }
    if (!SDL_ConvertPixelsAndColorspace(swdata->w, swdata->h, swdata->format, swdata->colorspace, 0, swdata->planes[0], swdata->pitches[0], target_format, SDL_COLORSPACE_SRGB, 0, pixels, pitch)) {
        return false;
    }
    if (stretch) {
        SDL_Rect rect = *srcrect;
        return SDL_StretchSurface(swdata->stretch, &rect, swdata->display, NULL, SDL_SCALEMODE_NEAREST);
    } else {
        return true;
    }
}

void SDL_SW_DestroyYUVTexture(SDL_SW_YUVTexture *swdata)
{
    if (swdata) {
        SDL_aligned_free(swdata->pixels);
        SDL_DestroySurface(swdata->stretch);
        SDL_DestroySurface(swdata->display);
        SDL_free(swdata);
    }
}

#endif // SDL_HAVE_YUV
