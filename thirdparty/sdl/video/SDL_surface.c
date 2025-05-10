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

#include "SDL_sysvideo.h"
#include "SDL_video_c.h"
#include "SDL_RLEaccel_c.h"
#include "SDL_pixels_c.h"
#include "SDL_stb_c.h"
#include "SDL_yuv_c.h"
#include "../render/SDL_sysrender.h"

#include "SDL_surface_c.h"


// Check to make sure we can safely check multiplication of surface w and pitch and it won't overflow size_t
SDL_COMPILE_TIME_ASSERT(surface_size_assumptions,
                        sizeof(int) == sizeof(Sint32) && sizeof(size_t) >= sizeof(Sint32));

SDL_COMPILE_TIME_ASSERT(can_indicate_overflow, SDL_SIZE_MAX > SDL_MAX_SINT32);

// Magic!
static char SDL_surface_magic;

// Public routines

bool SDL_SurfaceValid(SDL_Surface *surface)
{
    return (surface && surface->reserved == &SDL_surface_magic);
}

void SDL_UpdateSurfaceLockFlag(SDL_Surface *surface)
{
    if (SDL_SurfaceHasRLE(surface)) {
        surface->flags |= SDL_SURFACE_LOCK_NEEDED;
    } else {
        surface->flags &= ~SDL_SURFACE_LOCK_NEEDED;
    }
}

/*
 * Calculate the pad-aligned scanline width of a surface.
 *
 * for FOURCC, use SDL_CalculateYUVSize()
 */
static bool SDL_CalculateRGBSize(Uint32 format, size_t width, size_t height, size_t *size, size_t *pitch, bool minimal)
{
    if (SDL_BITSPERPIXEL(format) >= 8) {
        if (!SDL_size_mul_check_overflow(width, SDL_BYTESPERPIXEL(format), pitch)) {
            return SDL_SetError("width * bpp would overflow");
        }
    } else {
        if (!SDL_size_mul_check_overflow(width, SDL_BITSPERPIXEL(format), pitch)) {
            return SDL_SetError("width * bpp would overflow");
        }
        if (!SDL_size_add_check_overflow(*pitch, 7, pitch)) {
            return SDL_SetError("aligning pitch would overflow");
        }
        *pitch /= 8;
    }
    if (!minimal) {
        // 4-byte aligning for speed
        if (!SDL_size_add_check_overflow(*pitch, 3, pitch)) {
            return SDL_SetError("aligning pitch would overflow");
        }
        *pitch &= ~3;
    }

    if (!SDL_size_mul_check_overflow(height, *pitch, size)) {
        return SDL_SetError("height * pitch would overflow");
    }

    return true;
}

bool SDL_CalculateSurfaceSize(SDL_PixelFormat format, int width, int height, size_t *size, size_t *pitch, bool minimalPitch)
{
    size_t p = 0, sz = 0;

    if (size) {
        *size = 0;
    }

    if (pitch) {
        *pitch = 0;
    }

    if (SDL_ISPIXELFORMAT_FOURCC(format)) {
        if (format == SDL_PIXELFORMAT_MJPG) {
            // We don't know in advance what it will be, we'll figure it out later.
            return true;
        }

        if (!SDL_CalculateYUVSize(format, width, height, &sz, &p)) {
            // Overflow...
            return false;
        }
    } else {
        if (!SDL_CalculateRGBSize(format, width, height, &sz, &p, minimalPitch)) {
            // Overflow...
            return false;
        }
    }

    if (size) {
        *size = sz;
    }

    if (pitch) {
        *pitch = p;
    }

    return true;
}

static bool SDL_InitializeSurface(SDL_Surface *surface, int width, int height, SDL_PixelFormat format, SDL_Colorspace colorspace, SDL_PropertiesID props, void *pixels, int pitch, bool onstack)
{
    SDL_zerop(surface);

    surface->flags = SDL_SURFACE_PREALLOCATED;
    surface->format = format;
    surface->w = width;
    surface->h = height;
    surface->pixels = pixels;
    surface->pitch = pitch;
    surface->reserved = &SDL_surface_magic;

    if (onstack) {
        surface->internal_flags |= SDL_INTERNAL_SURFACE_STACK;
    }

    surface->fmt = SDL_GetPixelFormatDetails(format);
    if (!surface->fmt) {
        SDL_DestroySurface(surface);
        return false;
    }

    // Initialize the clip rect
    surface->clip_rect.w = width;
    surface->clip_rect.h = height;

    // Allocate an empty mapping
    surface->map.info.r = 0xFF;
    surface->map.info.g = 0xFF;
    surface->map.info.b = 0xFF;
    surface->map.info.a = 0xFF;

    if (colorspace == SDL_COLORSPACE_UNKNOWN) {
        surface->colorspace = SDL_GetDefaultColorspaceForFormat(format);
    } else {
        surface->colorspace = colorspace;
    }

    if (props) {
        if (!SDL_CopyProperties(props, SDL_GetSurfaceProperties(surface))) {
            SDL_DestroySurface(surface);
            return false;
        }
    }

    // By default surfaces with an alpha mask are set up for blending
    if (SDL_ISPIXELFORMAT_ALPHA(surface->format)) {
        SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_BLEND);
    }

    // The surface is ready to go
    surface->refcount = 1;
    return true;
}

/*
 * Create an empty surface of the appropriate depth using the given format
 */
SDL_Surface *SDL_CreateSurface(int width, int height, SDL_PixelFormat format)
{
    size_t pitch, size;
    SDL_Surface *surface;

    if (width < 0) {
        SDL_InvalidParamError("width");
        return NULL;
    }

    if (height < 0) {
        SDL_InvalidParamError("height");
        return NULL;
    }

    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        SDL_InvalidParamError("format");
        return NULL;
    }

    if (!SDL_CalculateSurfaceSize(format, width, height, &size, &pitch, false /* not minimal pitch */)) {
        // Overflow...
        return NULL;
    }

    // Allocate and initialize the surface
    surface = (SDL_Surface *)SDL_malloc(sizeof(*surface));
    if (!surface) {
        return NULL;
    }

    if (!SDL_InitializeSurface(surface, width, height, format, SDL_COLORSPACE_UNKNOWN, 0, NULL, (int)pitch, false)) {
        return NULL;
    }

    if (surface->w && surface->h && format != SDL_PIXELFORMAT_MJPG) {
        surface->flags &= ~SDL_SURFACE_PREALLOCATED;
        surface->pixels = SDL_aligned_alloc(SDL_GetSIMDAlignment(), size);
        if (!surface->pixels) {
            SDL_DestroySurface(surface);
            return NULL;
        }
        surface->flags |= SDL_SURFACE_SIMD_ALIGNED;

        // This is important for bitmaps
        SDL_memset(surface->pixels, 0, size);
    }
    return surface;
}

/*
 * Create an RGB surface from an existing memory buffer using the given
 * enum SDL_PIXELFORMAT_* format
 */
SDL_Surface *SDL_CreateSurfaceFrom(int width, int height, SDL_PixelFormat format, void *pixels, int pitch)
{
    if (width < 0) {
        SDL_InvalidParamError("width");
        return NULL;
    }

    if (height < 0) {
        SDL_InvalidParamError("height");
        return NULL;
    }

    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        SDL_InvalidParamError("format");
        return NULL;
    }

    if (pitch == 0 && !pixels) {
        // The application will fill these in later with valid values
    } else {
        size_t minimalPitch;

        if (!SDL_CalculateSurfaceSize(format, width, height, NULL, &minimalPitch, true /* minimal pitch */)) {
            // Overflow...
            return NULL;
        }

        if (pitch < 0 || (size_t)pitch < minimalPitch) {
            SDL_InvalidParamError("pitch");
            return NULL;
        }
    }

    // Allocate and initialize the surface
    SDL_Surface *surface = (SDL_Surface *)SDL_malloc(sizeof(*surface));
    if (!surface ||
        !SDL_InitializeSurface(surface, width, height, format, SDL_COLORSPACE_UNKNOWN, 0, pixels, pitch, false)) {
        return NULL;
    }
    return surface;
}

SDL_PropertiesID SDL_GetSurfaceProperties(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return 0;
    }

    if (!surface->props) {
        surface->props = SDL_CreateProperties();
    }
    return surface->props;
}

bool SDL_SetSurfaceColorspace(SDL_Surface *surface, SDL_Colorspace colorspace)
{
    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    surface->colorspace = colorspace;
    return true;
}

SDL_Colorspace SDL_GetSurfaceColorspace(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return SDL_COLORSPACE_UNKNOWN;
    }

    return surface->colorspace;
}

float SDL_GetDefaultSDRWhitePoint(SDL_Colorspace colorspace)
{
    return SDL_GetSurfaceSDRWhitePoint(NULL, colorspace);
}

float SDL_GetSurfaceSDRWhitePoint(SDL_Surface *surface, SDL_Colorspace colorspace)
{
    SDL_TransferCharacteristics transfer = SDL_COLORSPACETRANSFER(colorspace);

    if (transfer == SDL_TRANSFER_CHARACTERISTICS_LINEAR ||
        transfer == SDL_TRANSFER_CHARACTERISTICS_PQ) {
        SDL_PropertiesID props;
        float default_value = 1.0f;

        if (SDL_SurfaceValid(surface)) {
            props = surface->props;
        } else {
            props = 0;
        }
        if (transfer == SDL_TRANSFER_CHARACTERISTICS_PQ) {
            /* The older standards use an SDR white point of 100 nits.
             * ITU-R BT.2408-6 recommends using an SDR white point of 203 nits.
             * This is the default Chrome uses, and what a lot of game content
             * assumes, so we'll go with that.
             */
            const float DEFAULT_PQ_SDR_WHITE_POINT = 203.0f;
            default_value = DEFAULT_PQ_SDR_WHITE_POINT;
        }
        return SDL_GetFloatProperty(props, SDL_PROP_SURFACE_SDR_WHITE_POINT_FLOAT, default_value);
    }
    return 1.0f;
}

float SDL_GetDefaultHDRHeadroom(SDL_Colorspace colorspace)
{
    return SDL_GetSurfaceHDRHeadroom(NULL, colorspace);
}

float SDL_GetSurfaceHDRHeadroom(SDL_Surface *surface, SDL_Colorspace colorspace)
{
    SDL_TransferCharacteristics transfer = SDL_COLORSPACETRANSFER(colorspace);

    if (transfer == SDL_TRANSFER_CHARACTERISTICS_LINEAR ||
        transfer == SDL_TRANSFER_CHARACTERISTICS_PQ) {
        SDL_PropertiesID props;
        float default_value = 0.0f;

        if (SDL_SurfaceValid(surface)) {
            props = surface->props;
        } else {
            props = 0;
        }
        return SDL_GetFloatProperty(props, SDL_PROP_SURFACE_HDR_HEADROOM_FLOAT, default_value);
    }
    return 1.0f;
}

SDL_Palette *SDL_CreateSurfacePalette(SDL_Surface *surface)
{
    SDL_Palette *palette;

    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    if (!SDL_ISPIXELFORMAT_INDEXED(surface->format)) {
        SDL_SetError("The surface is not indexed format");
        return NULL;
    }

    palette = SDL_CreatePalette((1 << SDL_BITSPERPIXEL(surface->format)));
    if (!palette) {
        return NULL;
    }

    if (palette->ncolors == 2) {
        // Create a black and white bitmap palette
        palette->colors[0].r = 0xFF;
        palette->colors[0].g = 0xFF;
        palette->colors[0].b = 0xFF;
        palette->colors[1].r = 0x00;
        palette->colors[1].g = 0x00;
        palette->colors[1].b = 0x00;
    }

    if (!SDL_SetSurfacePalette(surface, palette)) {
        SDL_DestroyPalette(palette);
        return NULL;
    }

    // The surface has retained the palette, we can remove the reference here
    SDL_assert(palette->refcount == 2);
    SDL_DestroyPalette(palette);
    return palette;
}

bool SDL_SetSurfacePalette(SDL_Surface *surface, SDL_Palette *palette)
{
    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (palette && palette->ncolors > (1 << SDL_BITSPERPIXEL(surface->format))) {
        return SDL_SetError("SDL_SetSurfacePalette() passed a palette that doesn't match the surface format");
    }

    if (palette != surface->palette) {
        if (surface->palette) {
            SDL_DestroyPalette(surface->palette);
        }

        surface->palette = palette;

        if (surface->palette) {
            ++surface->palette->refcount;
        }
    }

    SDL_InvalidateMap(&surface->map);

    return true;
}

SDL_Palette *SDL_GetSurfacePalette(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return NULL;
    }

    return surface->palette;
}

bool SDL_AddSurfaceAlternateImage(SDL_Surface *surface, SDL_Surface *image)
{
    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (!SDL_SurfaceValid(image)) {
        return SDL_InvalidParamError("image");
    }

    SDL_Surface **images = (SDL_Surface **)SDL_realloc(surface->images, (surface->num_images + 1) * sizeof(*images));
    if (!images) {
        return false;
    }
    images[surface->num_images] = image;
    surface->images = images;
    ++surface->num_images;
    ++image->refcount;
    return true;
}

bool SDL_SurfaceHasAlternateImages(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    return (surface->num_images > 0);
}

SDL_Surface **SDL_GetSurfaceImages(SDL_Surface *surface, int *count)
{
    if (count) {
        *count = 0;
    }

    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    int num_images = 1 + surface->num_images;
    SDL_Surface **images = (SDL_Surface **)SDL_malloc((num_images + 1) * sizeof(*images));
    if (!images) {
        return NULL;
    }
    images[0] = surface;
    if (surface->num_images > 0) {
        SDL_memcpy(&images[1], surface->images, (surface->num_images * sizeof(images[1])));
    }
    images[num_images] = NULL;

    if (count) {
        *count = num_images;
    }
    return images;
}

SDL_Surface *SDL_GetSurfaceImage(SDL_Surface *surface, float display_scale)
{
    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    if (!SDL_SurfaceHasAlternateImages(surface)) {
        ++surface->refcount;
        return surface;
    }

    // This surface has high DPI images, pick the best one available, or scale one to the correct size
    SDL_Surface **images = SDL_GetSurfaceImages(surface, NULL);
    if (!images) {
        // Failure, fall back to the existing surface
        ++surface->refcount;
        return surface;
    }

    // Find closest image. Images that are larger than the
    // desired size are preferred over images that are smaller.
    SDL_Surface *closest = NULL;
    int desired_w = (int)SDL_round(surface->w * display_scale);
    int desired_h = (int)SDL_round(surface->h * display_scale);
    int desired_size = desired_w * desired_h;
    int closest_distance = -1;
    int closest_size = -1;
    for (int i = 0; images[i]; ++i) {
        SDL_Surface *candidate = images[i];
        int size = candidate->w * candidate->h;
        int delta_w = (candidate->w - desired_w);
        int delta_h = (candidate->h - desired_h);
        int distance = (delta_w * delta_w) + (delta_h * delta_h);
        if (closest_distance < 0 || distance < closest_distance ||
            (size > desired_size && closest_size < desired_size)) {
            closest = candidate;
            closest_distance = distance;
            closest_size = size;
        }
    }
    SDL_free(images);
    SDL_assert(closest != NULL);    // We should always have at least one surface

    if (closest->w == desired_w && closest->h == desired_h) {
        ++closest->refcount;
        return closest;
    }

    // We need to scale the image to the correct size. To maintain good image quality, downscaling
    // is done in steps, never reducing the width and height by more than half each time.
    SDL_Surface *scaled = closest;
    do {
        int next_scaled_w = SDL_max(desired_w, (scaled->w + 1) / 2);
        int next_scaled_h = SDL_max(desired_h, (scaled->h + 1) / 2);
        SDL_Surface *next_scaled = SDL_ScaleSurface(scaled, next_scaled_w, next_scaled_h, SDL_SCALEMODE_LINEAR);
        if (scaled != closest) {
            SDL_DestroySurface(scaled);
        }
        scaled = next_scaled;
        if (!scaled) {
            // Failure, fall back to the closest surface
            ++closest->refcount;
            return closest;
        }
    } while (scaled->w != desired_w || scaled->h != desired_h);

    return scaled;
}

void SDL_RemoveSurfaceAlternateImages(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return;
    }

    if (surface->num_images > 0) {
        for (int i = 0; i < surface->num_images; ++i) {
            SDL_DestroySurface(surface->images[i]);
        }
        SDL_free(surface->images);
        surface->images = NULL;
        surface->num_images = 0;
    }
}

bool SDL_SetSurfaceRLE(SDL_Surface *surface, bool enabled)
{
    int flags;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    flags = surface->map.info.flags;
    if (enabled) {
        surface->map.info.flags |= SDL_COPY_RLE_DESIRED;
    } else {
        surface->map.info.flags &= ~SDL_COPY_RLE_DESIRED;
    }
    if (surface->map.info.flags != flags) {
        SDL_InvalidateMap(&surface->map);
    }
    SDL_UpdateSurfaceLockFlag(surface);
    return true;
}

bool SDL_SurfaceHasRLE(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    if (!(surface->map.info.flags & SDL_COPY_RLE_DESIRED)) {
        return false;
    }

    return true;
}

bool SDL_SetSurfaceColorKey(SDL_Surface *surface, bool enabled, Uint32 key)
{
    int flags;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (surface->palette && key >= ((Uint32)surface->palette->ncolors)) {
        return SDL_InvalidParamError("key");
    }

    flags = surface->map.info.flags;
    if (enabled) {
        surface->map.info.flags |= SDL_COPY_COLORKEY;
        surface->map.info.colorkey = key;
    } else {
        surface->map.info.flags &= ~SDL_COPY_COLORKEY;
    }
    if (surface->map.info.flags != flags) {
        SDL_InvalidateMap(&surface->map);
    }

    return true;
}

bool SDL_SurfaceHasColorKey(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    if (!(surface->map.info.flags & SDL_COPY_COLORKEY)) {
        return false;
    }

    return true;
}

bool SDL_GetSurfaceColorKey(SDL_Surface *surface, Uint32 *key)
{
    if (key) {
        *key = 0;
    }

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (!(surface->map.info.flags & SDL_COPY_COLORKEY)) {
        return SDL_SetError("Surface doesn't have a colorkey");
    }

    if (key) {
        *key = surface->map.info.colorkey;
    }
    return true;
}

/* This is a fairly slow function to switch from colorkey to alpha
   NB: it doesn't handle bpp 1 or 3, because they have no alpha channel */
static void SDL_ConvertColorkeyToAlpha(SDL_Surface *surface, bool ignore_alpha)
{
    int x, y, bpp;

    if (!SDL_SurfaceValid(surface)) {
        return;
    }

    if (!(surface->map.info.flags & SDL_COPY_COLORKEY) ||
        !SDL_ISPIXELFORMAT_ALPHA(surface->format)) {
        return;
    }

    bpp = SDL_BYTESPERPIXEL(surface->format);

    SDL_LockSurface(surface);

    if (bpp == 2) {
        Uint16 *row, *spot;
        Uint16 ckey = (Uint16)surface->map.info.colorkey;
        Uint16 mask = (Uint16)(~surface->fmt->Amask);

        // Ignore, or not, alpha in colorkey comparison
        if (ignore_alpha) {
            ckey &= mask;
            row = (Uint16 *)surface->pixels;
            for (y = surface->h; y--;) {
                spot = row;
                for (x = surface->w; x--;) {
                    if ((*spot & mask) == ckey) {
                        *spot &= mask;
                    }
                    ++spot;
                }
                row += surface->pitch / 2;
            }
        } else {
            row = (Uint16 *)surface->pixels;
            for (y = surface->h; y--;) {
                spot = row;
                for (x = surface->w; x--;) {
                    if (*spot == ckey) {
                        *spot &= mask;
                    }
                    ++spot;
                }
                row += surface->pitch / 2;
            }
        }
    } else if (bpp == 4) {
        Uint32 *row, *spot;
        Uint32 ckey = surface->map.info.colorkey;
        Uint32 mask = ~surface->fmt->Amask;

        // Ignore, or not, alpha in colorkey comparison
        if (ignore_alpha) {
            ckey &= mask;
            row = (Uint32 *)surface->pixels;
            for (y = surface->h; y--;) {
                spot = row;
                for (x = surface->w; x--;) {
                    if ((*spot & mask) == ckey) {
                        *spot &= mask;
                    }
                    ++spot;
                }
                row += surface->pitch / 4;
            }
        } else {
            row = (Uint32 *)surface->pixels;
            for (y = surface->h; y--;) {
                spot = row;
                for (x = surface->w; x--;) {
                    if (*spot == ckey) {
                        *spot &= mask;
                    }
                    ++spot;
                }
                row += surface->pitch / 4;
            }
        }
    }

    SDL_UnlockSurface(surface);

    SDL_SetSurfaceColorKey(surface, false, 0);
    SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_BLEND);
}

bool SDL_SetSurfaceColorMod(SDL_Surface *surface, Uint8 r, Uint8 g, Uint8 b)
{
    int flags;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    surface->map.info.r = r;
    surface->map.info.g = g;
    surface->map.info.b = b;

    flags = surface->map.info.flags;
    if (r != 0xFF || g != 0xFF || b != 0xFF) {
        surface->map.info.flags |= SDL_COPY_MODULATE_COLOR;
    } else {
        surface->map.info.flags &= ~SDL_COPY_MODULATE_COLOR;
    }
    if (surface->map.info.flags != flags) {
        SDL_InvalidateMap(&surface->map);
    }
    return true;
}

bool SDL_GetSurfaceColorMod(SDL_Surface *surface, Uint8 *r, Uint8 *g, Uint8 *b)
{
    if (!SDL_SurfaceValid(surface)) {
        if (r) {
            *r = 255;
        }
        if (g) {
            *g = 255;
        }
        if (b) {
            *b = 255;
        }
        return SDL_InvalidParamError("surface");
    }

    if (r) {
        *r = surface->map.info.r;
    }
    if (g) {
        *g = surface->map.info.g;
    }
    if (b) {
        *b = surface->map.info.b;
    }
    return true;
}

bool SDL_SetSurfaceAlphaMod(SDL_Surface *surface, Uint8 alpha)
{
    int flags;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    surface->map.info.a = alpha;

    flags = surface->map.info.flags;
    if (alpha != 0xFF) {
        surface->map.info.flags |= SDL_COPY_MODULATE_ALPHA;
    } else {
        surface->map.info.flags &= ~SDL_COPY_MODULATE_ALPHA;
    }
    if (surface->map.info.flags != flags) {
        SDL_InvalidateMap(&surface->map);
    }
    return true;
}

bool SDL_GetSurfaceAlphaMod(SDL_Surface *surface, Uint8 *alpha)
{
    if (!SDL_SurfaceValid(surface)) {
        if (alpha) {
            *alpha = 255;
        }
        return SDL_InvalidParamError("surface");
    }

    if (alpha) {
        *alpha = surface->map.info.a;
    }
    return true;
}

bool SDL_SetSurfaceBlendMode(SDL_Surface *surface, SDL_BlendMode blendMode)
{
    int flags;
    bool result = true;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (blendMode == SDL_BLENDMODE_INVALID) {
        return SDL_InvalidParamError("blendMode");
    }

    flags = surface->map.info.flags;
    surface->map.info.flags &= ~SDL_COPY_BLEND_MASK;
    switch (blendMode) {
    case SDL_BLENDMODE_NONE:
        break;
    case SDL_BLENDMODE_BLEND:
        surface->map.info.flags |= SDL_COPY_BLEND;
        break;
    case SDL_BLENDMODE_BLEND_PREMULTIPLIED:
        surface->map.info.flags |= SDL_COPY_BLEND_PREMULTIPLIED;
        break;
    case SDL_BLENDMODE_ADD:
        surface->map.info.flags |= SDL_COPY_ADD;
        break;
    case SDL_BLENDMODE_ADD_PREMULTIPLIED:
        surface->map.info.flags |= SDL_COPY_ADD_PREMULTIPLIED;
        break;
    case SDL_BLENDMODE_MOD:
        surface->map.info.flags |= SDL_COPY_MOD;
        break;
    case SDL_BLENDMODE_MUL:
        surface->map.info.flags |= SDL_COPY_MUL;
        break;
    default:
        result = SDL_Unsupported();
        break;
    }

    if (surface->map.info.flags != flags) {
        SDL_InvalidateMap(&surface->map);
    }

    return result;
}

bool SDL_GetSurfaceBlendMode(SDL_Surface *surface, SDL_BlendMode *blendMode)
{
    if (blendMode) {
        *blendMode = SDL_BLENDMODE_INVALID;
    }

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (!blendMode) {
        return true;
    }

    switch (surface->map.info.flags & SDL_COPY_BLEND_MASK) {
    case SDL_COPY_BLEND:
        *blendMode = SDL_BLENDMODE_BLEND;
        break;
    case SDL_COPY_BLEND_PREMULTIPLIED:
        *blendMode = SDL_BLENDMODE_BLEND_PREMULTIPLIED;
        break;
    case SDL_COPY_ADD:
        *blendMode = SDL_BLENDMODE_ADD;
        break;
    case SDL_COPY_ADD_PREMULTIPLIED:
        *blendMode = SDL_BLENDMODE_ADD_PREMULTIPLIED;
        break;
    case SDL_COPY_MOD:
        *blendMode = SDL_BLENDMODE_MOD;
        break;
    case SDL_COPY_MUL:
        *blendMode = SDL_BLENDMODE_MUL;
        break;
    default:
        *blendMode = SDL_BLENDMODE_NONE;
        break;
    }
    return true;
}

bool SDL_SetSurfaceClipRect(SDL_Surface *surface, const SDL_Rect *rect)
{
    SDL_Rect full_rect;

    // Don't do anything if there's no surface to act on
    if (!SDL_SurfaceValid(surface)) {
        return false;
    }

    // Set up the full surface rectangle
    full_rect.x = 0;
    full_rect.y = 0;
    full_rect.w = surface->w;
    full_rect.h = surface->h;

    // Set the clipping rectangle
    if (!rect) {
        surface->clip_rect = full_rect;
        return true;
    }
    return SDL_GetRectIntersection(rect, &full_rect, &surface->clip_rect);
}

bool SDL_GetSurfaceClipRect(SDL_Surface *surface, SDL_Rect *rect)
{
    if (!SDL_SurfaceValid(surface)) {
        if (rect) {
            SDL_zerop(rect);
        }
        return SDL_InvalidParamError("surface");
    }
    if (!rect) {
        return SDL_InvalidParamError("rect");
    }
    *rect = surface->clip_rect;
    return true;
}

/*
 * Set up a blit between two surfaces -- split into three parts:
 * The upper part, SDL_BlitSurface(), performs clipping and rectangle
 * verification.  The lower part is a pointer to a low level
 * accelerated blitting function.
 *
 * These parts are separated out and each used internally by this
 * library in the optimum places.  They are exported so that if
 * you know exactly what you are doing, you can optimize your code
 * by calling the one(s) you need.
 */
bool SDL_BlitSurfaceUnchecked(SDL_Surface *src, const SDL_Rect *srcrect,
                             SDL_Surface *dst, const SDL_Rect *dstrect)
{
    // Check to make sure the blit mapping is valid
    if (!SDL_ValidateMap(src, dst)) {
        return false;
    }
    return src->map.blit(src, srcrect, dst, dstrect);
}

bool SDL_BlitSurface(SDL_Surface *src, const SDL_Rect *srcrect, SDL_Surface *dst, const SDL_Rect *dstrect)
{
    SDL_Rect r_src, r_dst;

    // Make sure the surfaces aren't locked
    if (!SDL_SurfaceValid(src)) {
        return SDL_InvalidParamError("src");
    } else if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("dst");
    } else if ((src->flags & SDL_SURFACE_LOCKED) || (dst->flags & SDL_SURFACE_LOCKED)) {
        return SDL_SetError("Surfaces must not be locked during blit");
    }

    // Full src surface
    r_src.x = 0;
    r_src.y = 0;
    r_src.w = src->w;
    r_src.h = src->h;

    if (dstrect) {
        r_dst.x = dstrect->x;
        r_dst.y = dstrect->y;
    } else {
        r_dst.x = 0;
        r_dst.y = 0;
    }

    // clip the source rectangle to the source surface
    if (srcrect) {
        SDL_Rect tmp;
        if (SDL_GetRectIntersection(srcrect, &r_src, &tmp) == false) {
            return true;
        }

        // Shift dstrect, if srcrect origin has changed
        r_dst.x += tmp.x - srcrect->x;
        r_dst.y += tmp.y - srcrect->y;

        // Update srcrect
        r_src = tmp;
    }

    // There're no dstrect.w/h parameters. It's the same as srcrect
    r_dst.w = r_src.w;
    r_dst.h = r_src.h;

    // clip the destination rectangle against the clip rectangle
    {
        SDL_Rect tmp;
        if (SDL_GetRectIntersection(&r_dst, &dst->clip_rect, &tmp) == false) {
            return true;
        }

        // Shift srcrect, if dstrect has changed
        r_src.x += tmp.x - r_dst.x;
        r_src.y += tmp.y - r_dst.y;
        r_src.w = tmp.w;
        r_src.h = tmp.h;

        // Update dstrect
        r_dst = tmp;
    }

    if (r_dst.w <= 0 || r_dst.h <= 0) {
        // No-op.
        return true;
    }

    // Switch back to a fast blit if we were previously stretching
    if (src->map.info.flags & SDL_COPY_NEAREST) {
        src->map.info.flags &= ~SDL_COPY_NEAREST;
        SDL_InvalidateMap(&src->map);
    }

    return SDL_BlitSurfaceUnchecked(src, &r_src, dst, &r_dst);
}

bool SDL_BlitSurfaceScaled(SDL_Surface *src, const SDL_Rect *srcrect, SDL_Surface *dst, const SDL_Rect *dstrect, SDL_ScaleMode scaleMode)
{
    SDL_Rect *clip_rect;
    double src_x0, src_y0, src_x1, src_y1;
    double dst_x0, dst_y0, dst_x1, dst_y1;
    SDL_Rect final_src, final_dst;
    double scaling_w, scaling_h;
    int src_w, src_h;
    int dst_w, dst_h;

    // Make sure the surfaces aren't locked
    if (!SDL_SurfaceValid(src) || !src->pixels) {
        return SDL_InvalidParamError("src");
    } else if (!SDL_SurfaceValid(dst) || !dst->pixels) {
        return SDL_InvalidParamError("dst");
    } else if ((src->flags & SDL_SURFACE_LOCKED) || (dst->flags & SDL_SURFACE_LOCKED)) {
        return SDL_SetError("Surfaces must not be locked during blit");
    }

    switch (scaleMode) {
    case SDL_SCALEMODE_NEAREST:
        break;
    case SDL_SCALEMODE_LINEAR:
        break;
    case SDL_SCALEMODE_PIXELART:
        scaleMode = SDL_SCALEMODE_NEAREST;
        break;
    default:
        return SDL_InvalidParamError("scaleMode");
    }

    if (!srcrect) {
        src_w = src->w;
        src_h = src->h;
    } else {
        src_w = srcrect->w;
        src_h = srcrect->h;
    }

    if (!dstrect) {
        dst_w = dst->w;
        dst_h = dst->h;
    } else {
        dst_w = dstrect->w;
        dst_h = dstrect->h;
    }

    if (dst_w == src_w && dst_h == src_h) {
        // No scaling, defer to regular blit
        return SDL_BlitSurface(src, srcrect, dst, dstrect);
    }

    if (src_w == 0) {
        src_w = 1;
    }
    if (src_h == 0) {
        src_h = 1;
    }

    scaling_w = (double)dst_w / src_w;
    scaling_h = (double)dst_h / src_h;

    if (!dstrect) {
        dst_x0 = 0;
        dst_y0 = 0;
        dst_x1 = dst_w;
        dst_y1 = dst_h;
    } else {
        dst_x0 = dstrect->x;
        dst_y0 = dstrect->y;
        dst_x1 = dst_x0 + dst_w;
        dst_y1 = dst_y0 + dst_h;
    }

    if (!srcrect) {
        src_x0 = 0;
        src_y0 = 0;
        src_x1 = src_w;
        src_y1 = src_h;
    } else {
        src_x0 = srcrect->x;
        src_y0 = srcrect->y;
        src_x1 = src_x0 + src_w;
        src_y1 = src_y0 + src_h;

        // Clip source rectangle to the source surface

        if (src_x0 < 0) {
            dst_x0 -= src_x0 * scaling_w;
            src_x0 = 0;
        }

        if (src_x1 > src->w) {
            dst_x1 -= (src_x1 - src->w) * scaling_w;
            src_x1 = src->w;
        }

        if (src_y0 < 0) {
            dst_y0 -= src_y0 * scaling_h;
            src_y0 = 0;
        }

        if (src_y1 > src->h) {
            dst_y1 -= (src_y1 - src->h) * scaling_h;
            src_y1 = src->h;
        }
    }

    // Clip destination rectangle to the clip rectangle
    clip_rect = &dst->clip_rect;

    // Translate to clip space for easier calculations
    dst_x0 -= clip_rect->x;
    dst_x1 -= clip_rect->x;
    dst_y0 -= clip_rect->y;
    dst_y1 -= clip_rect->y;

    if (dst_x0 < 0) {
        src_x0 -= dst_x0 / scaling_w;
        dst_x0 = 0;
    }

    if (dst_x1 > clip_rect->w) {
        src_x1 -= (dst_x1 - clip_rect->w) / scaling_w;
        dst_x1 = clip_rect->w;
    }

    if (dst_y0 < 0) {
        src_y0 -= dst_y0 / scaling_h;
        dst_y0 = 0;
    }

    if (dst_y1 > clip_rect->h) {
        src_y1 -= (dst_y1 - clip_rect->h) / scaling_h;
        dst_y1 = clip_rect->h;
    }

    // Translate back to surface coordinates
    dst_x0 += clip_rect->x;
    dst_x1 += clip_rect->x;
    dst_y0 += clip_rect->y;
    dst_y1 += clip_rect->y;

    final_src.x = (int)SDL_round(src_x0);
    final_src.y = (int)SDL_round(src_y0);
    final_src.w = (int)SDL_round(src_x1 - src_x0);
    final_src.h = (int)SDL_round(src_y1 - src_y0);

    final_dst.x = (int)SDL_round(dst_x0);
    final_dst.y = (int)SDL_round(dst_y0);
    final_dst.w = (int)SDL_round(dst_x1 - dst_x0);
    final_dst.h = (int)SDL_round(dst_y1 - dst_y0);

    // Clip again
    {
        SDL_Rect tmp;
        tmp.x = 0;
        tmp.y = 0;
        tmp.w = src->w;
        tmp.h = src->h;
        SDL_GetRectIntersection(&tmp, &final_src, &final_src);
    }

    // Clip again
    SDL_GetRectIntersection(clip_rect, &final_dst, &final_dst);

    if (final_dst.w == 0 || final_dst.h == 0 ||
        final_src.w < 0 || final_src.h < 0) {
        // No-op.
        return true;
    }

    return SDL_BlitSurfaceUncheckedScaled(src, &final_src, dst, &final_dst, scaleMode);
}

/**
 *  This is a semi-private blit function and it performs low-level surface
 *  scaled blitting only.
 */
bool SDL_BlitSurfaceUncheckedScaled(SDL_Surface *src, const SDL_Rect *srcrect, SDL_Surface *dst, const SDL_Rect *dstrect, SDL_ScaleMode scaleMode)
{
    static const Uint32 complex_copy_flags = (SDL_COPY_MODULATE_MASK | SDL_COPY_BLEND_MASK | SDL_COPY_COLORKEY);

    if (srcrect->w > SDL_MAX_UINT16 || srcrect->h > SDL_MAX_UINT16 ||
        dstrect->w > SDL_MAX_UINT16 || dstrect->h > SDL_MAX_UINT16) {
        return SDL_SetError("Size too large for scaling");
    }

    if (!(src->map.info.flags & SDL_COPY_NEAREST)) {
        src->map.info.flags |= SDL_COPY_NEAREST;
        SDL_InvalidateMap(&src->map);
    }

    if (scaleMode == SDL_SCALEMODE_NEAREST || scaleMode == SDL_SCALEMODE_PIXELART) {
        if (!(src->map.info.flags & complex_copy_flags) &&
            src->format == dst->format &&
            !SDL_ISPIXELFORMAT_INDEXED(src->format) &&
            SDL_BYTESPERPIXEL(src->format) <= 4) {
            return SDL_StretchSurface(src, srcrect, dst, dstrect, SDL_SCALEMODE_NEAREST);
        } else if (SDL_BITSPERPIXEL(src->format) < 8) {
            // Scaling bitmap not yet supported, convert to RGBA for blit
            bool result = false;
            SDL_Surface *tmp = SDL_ConvertSurface(src, SDL_PIXELFORMAT_ARGB8888);
            if (tmp) {
                result = SDL_BlitSurfaceUncheckedScaled(tmp, srcrect, dst, dstrect, SDL_SCALEMODE_NEAREST);
                SDL_DestroySurface(tmp);
            }
            return result;
        } else {
            return SDL_BlitSurfaceUnchecked(src, srcrect, dst, dstrect);
        }
    } else {
        if (!(src->map.info.flags & complex_copy_flags) &&
            src->format == dst->format &&
            !SDL_ISPIXELFORMAT_INDEXED(src->format) &&
            SDL_BYTESPERPIXEL(src->format) == 4 &&
            src->format != SDL_PIXELFORMAT_ARGB2101010) {
            // fast path
            return SDL_StretchSurface(src, srcrect, dst, dstrect, SDL_SCALEMODE_LINEAR);
        } else if (SDL_BITSPERPIXEL(src->format) < 8) {
            // Scaling bitmap not yet supported, convert to RGBA for blit
            bool result = false;
            SDL_Surface *tmp = SDL_ConvertSurface(src, SDL_PIXELFORMAT_ARGB8888);
            if (tmp) {
                result = SDL_BlitSurfaceUncheckedScaled(tmp, srcrect, dst, dstrect, scaleMode);
                SDL_DestroySurface(tmp);
            }
            return result;
        } else {
            // Use intermediate surface(s)
            SDL_Surface *tmp1 = NULL;
            bool result;
            SDL_Rect srcrect2;
            int is_complex_copy_flags = (src->map.info.flags & complex_copy_flags);

            Uint8 r, g, b;
            Uint8 alpha;
            SDL_BlendMode blendMode;

            // Save source infos
            SDL_GetSurfaceColorMod(src, &r, &g, &b);
            SDL_GetSurfaceAlphaMod(src, &alpha);
            SDL_GetSurfaceBlendMode(src, &blendMode);
            srcrect2.x = srcrect->x;
            srcrect2.y = srcrect->y;
            srcrect2.w = srcrect->w;
            srcrect2.h = srcrect->h;

            // Change source format if not appropriate for scaling
            if (SDL_BYTESPERPIXEL(src->format) != 4 || src->format == SDL_PIXELFORMAT_ARGB2101010) {
                SDL_Rect tmprect;
                SDL_PixelFormat fmt;
                tmprect.x = 0;
                tmprect.y = 0;
                tmprect.w = src->w;
                tmprect.h = src->h;
                if (SDL_BYTESPERPIXEL(dst->format) == 4 && dst->format != SDL_PIXELFORMAT_ARGB2101010) {
                    fmt = dst->format;
                } else {
                    fmt = SDL_PIXELFORMAT_ARGB8888;
                }
                tmp1 = SDL_CreateSurface(src->w, src->h, fmt);
                SDL_BlitSurfaceUnchecked(src, srcrect, tmp1, &tmprect);

                srcrect2.x = 0;
                srcrect2.y = 0;
                SDL_SetSurfaceColorMod(tmp1, r, g, b);
                SDL_SetSurfaceAlphaMod(tmp1, alpha);
                SDL_SetSurfaceBlendMode(tmp1, blendMode);

                src = tmp1;
            }

            // Intermediate scaling
            if (is_complex_copy_flags || src->format != dst->format) {
                SDL_Rect tmprect;
                SDL_Surface *tmp2 = SDL_CreateSurface(dstrect->w, dstrect->h, src->format);
                SDL_StretchSurface(src, &srcrect2, tmp2, NULL, SDL_SCALEMODE_LINEAR);

                SDL_SetSurfaceColorMod(tmp2, r, g, b);
                SDL_SetSurfaceAlphaMod(tmp2, alpha);
                SDL_SetSurfaceBlendMode(tmp2, blendMode);

                tmprect.x = 0;
                tmprect.y = 0;
                tmprect.w = dstrect->w;
                tmprect.h = dstrect->h;
                result = SDL_BlitSurfaceUnchecked(tmp2, &tmprect, dst, dstrect);
                SDL_DestroySurface(tmp2);
            } else {
                result = SDL_StretchSurface(src, &srcrect2, dst, dstrect, SDL_SCALEMODE_LINEAR);
            }

            SDL_DestroySurface(tmp1);
            return result;
        }
    }
}

bool SDL_BlitSurfaceTiled(SDL_Surface *src, const SDL_Rect *srcrect, SDL_Surface *dst, const SDL_Rect *dstrect)
{
    SDL_Rect r_src, r_dst;

    // Make sure the surfaces aren't locked
    if (!SDL_SurfaceValid(src)) {
        return SDL_InvalidParamError("src");
    } else if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("dst");
    } else if ((src->flags & SDL_SURFACE_LOCKED) || (dst->flags & SDL_SURFACE_LOCKED)) {
        return SDL_SetError("Surfaces must not be locked during blit");
    }

    // Full src surface
    r_src.x = 0;
    r_src.y = 0;
    r_src.w = src->w;
    r_src.h = src->h;

    if (dstrect) {
        r_dst.x = dstrect->x;
        r_dst.y = dstrect->y;
        r_dst.w = dstrect->w;
        r_dst.h = dstrect->h;
    } else {
        r_dst.x = 0;
        r_dst.y = 0;
        r_dst.w = dst->w;
        r_dst.h = dst->h;
    }

    // clip the source rectangle to the source surface
    if (srcrect) {
        if (SDL_GetRectIntersection(srcrect, &r_src, &r_src) == false) {
            return true;
        }

        // For tiling we don't adjust the destination rectangle
    }

    // clip the destination rectangle against the clip rectangle
    {
        if (SDL_GetRectIntersection(&r_dst, &dst->clip_rect, &r_dst) == false) {
            return true;
        }

        // For tiling we don't adjust the source rectangle
    }

    // Switch back to a fast blit if we were previously stretching
    if (src->map.info.flags & SDL_COPY_NEAREST) {
        src->map.info.flags &= ~SDL_COPY_NEAREST;
        SDL_InvalidateMap(&src->map);
    }

    int rows = r_dst.h / r_src.h;
    int cols = r_dst.w / r_src.w;
    int remaining_w = r_dst.w % r_src.w;
    int remaining_h = r_dst.h % r_src.h;
    SDL_Rect curr_src, curr_dst;

    SDL_copyp(&curr_src, &r_src);
    curr_dst.y = r_dst.y;
    curr_dst.w = r_src.w;
    curr_dst.h = r_src.h;
    for (int y = 0; y < rows; ++y) {
        curr_dst.x = r_dst.x;
        for (int x = 0; x < cols; ++x) {
            if (!SDL_BlitSurfaceUnchecked(src, &curr_src, dst, &curr_dst)) {
                return false;
            }
            curr_dst.x += curr_dst.w;
        }
        if (remaining_w) {
            curr_src.w = remaining_w;
            curr_dst.w = remaining_w;
            if (!SDL_BlitSurfaceUnchecked(src, &curr_src, dst, &curr_dst)) {
                return false;
            }
            curr_src.w = r_src.w;
            curr_dst.w = r_src.w;
        }
        curr_dst.y += curr_dst.h;
    }
    if (remaining_h) {
        curr_src.h = remaining_h;
        curr_dst.h = remaining_h;
        curr_dst.x = r_dst.x;
        for (int x = 0; x < cols; ++x) {
            if (!SDL_BlitSurfaceUnchecked(src, &curr_src, dst, &curr_dst)) {
                return false;
            }
            curr_dst.x += curr_dst.w;
        }
        if (remaining_w) {
            curr_src.w = remaining_w;
            curr_dst.w = remaining_w;
            if (!SDL_BlitSurfaceUnchecked(src, &curr_src, dst, &curr_dst)) {
                return false;
            }
        }
    }
    return true;
}

bool SDL_BlitSurfaceTiledWithScale(SDL_Surface *src, const SDL_Rect *srcrect, float scale, SDL_ScaleMode scaleMode, SDL_Surface *dst, const SDL_Rect *dstrect)
{
    SDL_Rect r_src, r_dst;

    // Make sure the surfaces aren't locked
    if (!SDL_SurfaceValid(src)) {
        return SDL_InvalidParamError("src");
    } else if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("dst");
    } else if ((src->flags & SDL_SURFACE_LOCKED) || (dst->flags & SDL_SURFACE_LOCKED)) {
        return SDL_SetError("Surfaces must not be locked during blit");
    }

    if (scale <= 0.0f) {
        return SDL_InvalidParamError("scale");
    }

    // Full src surface
    r_src.x = 0;
    r_src.y = 0;
    r_src.w = src->w;
    r_src.h = src->h;

    if (dstrect) {
        r_dst.x = dstrect->x;
        r_dst.y = dstrect->y;
        r_dst.w = dstrect->w;
        r_dst.h = dstrect->h;
    } else {
        r_dst.x = 0;
        r_dst.y = 0;
        r_dst.w = dst->w;
        r_dst.h = dst->h;
    }

    // clip the source rectangle to the source surface
    if (srcrect) {
        if (SDL_GetRectIntersection(srcrect, &r_src, &r_src) == false) {
            return true;
        }

        // For tiling we don't adjust the destination rectangle
    }

    // clip the destination rectangle against the clip rectangle
    {
        if (SDL_GetRectIntersection(&r_dst, &dst->clip_rect, &r_dst) == false) {
            return true;
        }

        // For tiling we don't adjust the source rectangle
    }

    // Switch back to a fast blit if we were previously stretching
    if (src->map.info.flags & SDL_COPY_NEAREST) {
        src->map.info.flags &= ~SDL_COPY_NEAREST;
        SDL_InvalidateMap(&src->map);
    }

    int tile_width = (int)(r_src.w * scale);
    int tile_height = (int)(r_src.h * scale);
    int rows = r_dst.h / tile_height;
    int cols = r_dst.w / tile_width;
    int remaining_dst_w = (r_dst.w - cols * tile_width);
    int remaining_dst_h = (r_dst.h - rows * tile_height);
    int remaining_src_w = (int)(remaining_dst_w / scale);
    int remaining_src_h = (int)(remaining_dst_h / scale);
    SDL_Rect curr_src, curr_dst;

    SDL_copyp(&curr_src, &r_src);
    curr_dst.y = r_dst.y;
    curr_dst.w = tile_width;
    curr_dst.h = tile_height;
    for (int y = 0; y < rows; ++y) {
        curr_dst.x = r_dst.x;
        for (int x = 0; x < cols; ++x) {
            if (!SDL_BlitSurfaceUncheckedScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
                return false;
            }
            curr_dst.x += curr_dst.w;
        }
        if (remaining_dst_w > 0) {
            curr_src.w = remaining_src_w;
            curr_dst.w = remaining_dst_w;
            if (!SDL_BlitSurfaceUncheckedScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
                return false;
            }
            curr_src.w = r_src.w;
            curr_dst.w = tile_width;
        }
        curr_dst.y += curr_dst.h;
    }
    if (remaining_dst_h > 0) {
        curr_src.h = remaining_src_h;
        curr_dst.h = remaining_dst_h;
        curr_dst.x = r_dst.x;
        for (int x = 0; x < cols; ++x) {
            if (!SDL_BlitSurfaceUncheckedScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
                return false;
            }
            curr_dst.x += curr_dst.w;
        }
        if (remaining_dst_w > 0) {
            curr_src.w = remaining_src_w;
            curr_dst.w = remaining_dst_w;
            if (!SDL_BlitSurfaceUncheckedScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
                return false;
            }
        }
    }
    return true;
}

bool SDL_BlitSurface9Grid(SDL_Surface *src, const SDL_Rect *srcrect, int left_width, int right_width, int top_height, int bottom_height, float scale, SDL_ScaleMode scaleMode, SDL_Surface *dst, const SDL_Rect *dstrect)
{
    SDL_Rect full_src, full_dst;
    SDL_Rect curr_src, curr_dst;
    int dst_left_width;
    int dst_right_width;
    int dst_top_height;
    int dst_bottom_height;

    // Make sure the surfaces aren't locked
    if (!SDL_SurfaceValid(src)) {
        return SDL_InvalidParamError("src");
    } else if (!SDL_SurfaceValid(dst)) {
        return SDL_InvalidParamError("dst");
    }

    if (!srcrect) {
        full_src.x = 0;
        full_src.y = 0;
        full_src.w = src->w;
        full_src.h = src->h;
        srcrect = &full_src;
    }

    if (!dstrect) {
        full_dst.x = 0;
        full_dst.y = 0;
        full_dst.w = dst->w;
        full_dst.h = dst->h;
        dstrect = &full_dst;
    }

    if (scale <= 0.0f || scale == 1.0f) {
        dst_left_width = left_width;
        dst_right_width = right_width;
        dst_top_height = top_height;
        dst_bottom_height = bottom_height;
    } else {
        dst_left_width = (int)SDL_roundf(left_width * scale);
        dst_right_width = (int)SDL_roundf(right_width * scale);
        dst_top_height = (int)SDL_roundf(top_height * scale);
        dst_bottom_height = (int)SDL_roundf(bottom_height * scale);
    }

    // Upper-left corner
    curr_src.x = srcrect->x;
    curr_src.y = srcrect->y;
    curr_src.w = left_width;
    curr_src.h = top_height;
    curr_dst.x = dstrect->x;
    curr_dst.y = dstrect->y;
    curr_dst.w = dst_left_width;
    curr_dst.h = dst_top_height;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Upper-right corner
    curr_src.x = srcrect->x + srcrect->w - right_width;
    curr_src.w = right_width;
    curr_dst.x = dstrect->x + dstrect->w - dst_right_width;
    curr_dst.w = dst_right_width;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Lower-right corner
    curr_src.y = srcrect->y + srcrect->h - bottom_height;
    curr_dst.y = dstrect->y + dstrect->h - dst_bottom_height;
    curr_dst.h = dst_bottom_height;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Lower-left corner
    curr_src.x = srcrect->x;
    curr_src.w = left_width;
    curr_dst.x = dstrect->x;
    curr_dst.w = dst_left_width;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Left
    curr_src.y = srcrect->y + top_height;
    curr_src.h = srcrect->h - top_height - bottom_height;
    curr_dst.y = dstrect->y + dst_top_height;
    curr_dst.h = dstrect->h - dst_top_height - dst_bottom_height;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Right
    curr_src.x = srcrect->x + srcrect->w - right_width;
    curr_src.w = right_width;
    curr_dst.x = dstrect->x + dstrect->w - dst_right_width;
    curr_dst.w = dst_right_width;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Top
    curr_src.x = srcrect->x + left_width;
    curr_src.y = srcrect->y;
    curr_src.w = srcrect->w - left_width - right_width;
    curr_src.h = top_height;
    curr_dst.x = dstrect->x + dst_left_width;
    curr_dst.y = dstrect->y;
    curr_dst.w = dstrect->w - dst_left_width - dst_right_width;
    curr_dst.h = dst_top_height;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Bottom
    curr_src.y = srcrect->y + srcrect->h - bottom_height;
    curr_dst.y = dstrect->y + dstrect->h - dst_bottom_height;
    curr_dst.h = dst_bottom_height;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    // Center
    curr_src.x = srcrect->x + left_width;
    curr_src.y = srcrect->y + top_height;
    curr_src.w = srcrect->w - left_width - right_width;
    curr_src.h = srcrect->h - top_height - bottom_height;
    curr_dst.x = dstrect->x + dst_left_width;
    curr_dst.y = dstrect->y + dst_top_height;
    curr_dst.w = dstrect->w - dst_left_width - dst_right_width;
    curr_dst.h = dstrect->h - dst_top_height - dst_bottom_height;
    if (!SDL_BlitSurfaceScaled(src, &curr_src, dst, &curr_dst, scaleMode)) {
        return false;
    }

    return true;
}

/*
 * Lock a surface to directly access the pixels
 */
bool SDL_LockSurface(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    if (!surface->locked) {
#ifdef SDL_HAVE_RLE
        // Perform the lock
        if (surface->internal_flags & SDL_INTERNAL_SURFACE_RLEACCEL) {
            SDL_UnRLESurface(surface, true);
            surface->internal_flags |= SDL_INTERNAL_SURFACE_RLEACCEL; // save accel'd state
        }
#endif
    }

    // Increment the surface lock count, for recursive locks
    ++surface->locked;
    surface->flags |= SDL_SURFACE_LOCKED;

    // Ready to go..
    return true;
}

/*
 * Unlock a previously locked surface
 */
void SDL_UnlockSurface(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return;
    }

    // Only perform an unlock if we are locked
    if (!surface->locked || (--surface->locked > 0)) {
        return;
    }

#ifdef SDL_HAVE_RLE
    // Update RLE encoded surface with new data
    if (surface->internal_flags & SDL_INTERNAL_SURFACE_RLEACCEL) {
        surface->internal_flags &= ~SDL_INTERNAL_SURFACE_RLEACCEL; // stop lying
        SDL_RLESurface(surface);
    }
#endif

    surface->flags &= ~SDL_SURFACE_LOCKED;
}

static bool SDL_FlipSurfaceHorizontal(SDL_Surface *surface)
{
    bool isstack;
    Uint8 *row, *a, *b, *tmp;
    int i, j, bpp;

    if (SDL_BITSPERPIXEL(surface->format) < 8) {
        // We could implement this if needed, but we'd have to flip sets of bits within a byte
        return SDL_Unsupported();
    }

    if (surface->h <= 0) {
        return true;
    }

    if (surface->w <= 1) {
        return true;
    }

    bpp = SDL_BYTESPERPIXEL(surface->format);
    row = (Uint8 *)surface->pixels;
    tmp = SDL_small_alloc(Uint8, surface->pitch, &isstack);
    if (!tmp) {
        return false;
    }
    for (i = surface->h; i--; ) {
        a = row;
        b = a + (surface->w - 1) * bpp;
        for (j = surface->w / 2; j--; ) {
            SDL_memcpy(tmp, a, bpp);
            SDL_memcpy(a, b, bpp);
            SDL_memcpy(b, tmp, bpp);
            a += bpp;
            b -= bpp;
        }
        row += surface->pitch;
    }
    SDL_small_free(tmp, isstack);
    return true;
}

static bool SDL_FlipSurfaceVertical(SDL_Surface *surface)
{
    bool isstack;
    Uint8 *a, *b, *tmp;
    int i;

    if (surface->h <= 1) {
        return true;
    }

    a = (Uint8 *)surface->pixels;
    b = a + (surface->h - 1) * surface->pitch;
    tmp = SDL_small_alloc(Uint8, surface->pitch, &isstack);
    if (!tmp) {
        return false;
    }
    for (i = surface->h / 2; i--; ) {
        SDL_memcpy(tmp, a, surface->pitch);
        SDL_memcpy(a, b, surface->pitch);
        SDL_memcpy(b, tmp, surface->pitch);
        a += surface->pitch;
        b -= surface->pitch;
    }
    SDL_small_free(tmp, isstack);
    return true;
}

bool SDL_FlipSurface(SDL_Surface *surface, SDL_FlipMode flip)
{
    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }
    if (!surface->pixels) {
        return true;
    }

    switch (flip) {
    case SDL_FLIP_HORIZONTAL:
        return SDL_FlipSurfaceHorizontal(surface);
    case SDL_FLIP_VERTICAL:
        return SDL_FlipSurfaceVertical(surface);
    default:
        return SDL_InvalidParamError("flip");
    }
}

SDL_Surface *SDL_ConvertSurfaceAndColorspace(SDL_Surface *surface, SDL_PixelFormat format, SDL_Palette *palette, SDL_Colorspace colorspace, SDL_PropertiesID props)
{
    SDL_Palette *temp_palette = NULL;
    SDL_Surface *convert = NULL;
    SDL_Colorspace src_colorspace;
    SDL_PropertiesID src_properties;
    Uint32 copy_flags;
    SDL_Color copy_color;
    SDL_Rect bounds;
    bool result;
    bool palette_ck_transform = false;
    Uint8 palette_ck_value = 0;
    Uint8 *palette_saved_alpha = NULL;
    int palette_saved_alpha_ncolors = 0;

    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        goto error;
    }

    if (format == SDL_PIXELFORMAT_UNKNOWN) {
        SDL_InvalidParamError("format");
        goto error;
    }

    // Check for empty destination palette! (results in empty image)
    if (palette) {
        int i;
        for (i = 0; i < palette->ncolors; ++i) {
            if ((palette->colors[i].r != 0xFF) || (palette->colors[i].g != 0xFF) || (palette->colors[i].b != 0xFF)) {
                break;
            }
        }
        if (i == palette->ncolors) {
            SDL_SetError("Empty destination palette");
            goto error;
        }
    } else if (SDL_ISPIXELFORMAT_INDEXED(format)) {
        // Create a dither palette for conversion
        temp_palette = SDL_CreatePalette(1 << SDL_BITSPERPIXEL(format));
        if (temp_palette) {
            SDL_DitherPalette(temp_palette);
            palette = temp_palette;
        }
    }

    src_colorspace = surface->colorspace;
    src_properties = surface->props;

    // Create a new surface with the desired format
    convert = SDL_CreateSurface(surface->w, surface->h, format);
    if (!convert) {
        goto error;
    }
    if (SDL_ISPIXELFORMAT_INDEXED(format)) {
        SDL_SetSurfacePalette(convert, palette);
    }

    if (colorspace == SDL_COLORSPACE_UNKNOWN) {
        colorspace = src_colorspace;
    }
    SDL_SetSurfaceColorspace(convert, colorspace);

    if (SDL_ISPIXELFORMAT_FOURCC(format) || SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        if (surface->format == SDL_PIXELFORMAT_MJPG && format == SDL_PIXELFORMAT_MJPG) {
            // Just do a straight pixel copy of the JPEG image
            size_t size = (size_t)surface->pitch;
            convert->pixels = SDL_malloc(size);
            if (!convert->pixels) {
                goto error;
            }
            convert->flags &= ~SDL_SURFACE_PREALLOCATED;
            convert->pitch = surface->pitch;
            SDL_memcpy(convert->pixels, surface->pixels, size);

        } else if (!SDL_ConvertPixelsAndColorspace(surface->w, surface->h, surface->format, src_colorspace, src_properties, surface->pixels, surface->pitch, convert->format, colorspace, props, convert->pixels, convert->pitch)) {
            goto error;
        }

        // Save the original copy flags
        copy_flags = surface->map.info.flags;

        goto end;
    }

    // Save the original copy flags
    copy_flags = surface->map.info.flags;
    copy_color.r = surface->map.info.r;
    copy_color.g = surface->map.info.g;
    copy_color.b = surface->map.info.b;
    copy_color.a = surface->map.info.a;
    surface->map.info.r = 0xFF;
    surface->map.info.g = 0xFF;
    surface->map.info.b = 0xFF;
    surface->map.info.a = 0xFF;
    surface->map.info.flags = (copy_flags & (SDL_COPY_RLE_COLORKEY | SDL_COPY_RLE_ALPHAKEY));
    SDL_InvalidateMap(&surface->map);

    // Copy over the image data
    bounds.x = 0;
    bounds.y = 0;
    bounds.w = surface->w;
    bounds.h = surface->h;

    /* Source surface has a palette with no real alpha (0 or OPAQUE).
     * Destination format has alpha.
     * -> set alpha channel to be opaque */
    if (surface->palette && SDL_ISPIXELFORMAT_ALPHA(format)) {
        bool set_opaque = false;

        bool is_opaque, has_alpha_channel;
        SDL_DetectPalette(surface->palette, &is_opaque, &has_alpha_channel);

        if (is_opaque) {
            if (!has_alpha_channel) {
                set_opaque = true;
            }
        }

        // Set opaque and backup palette alpha values
        if (set_opaque) {
            int i;
            palette_saved_alpha_ncolors = surface->palette->ncolors;
            if (palette_saved_alpha_ncolors > 0) {
                palette_saved_alpha = SDL_stack_alloc(Uint8, palette_saved_alpha_ncolors);
                for (i = 0; i < palette_saved_alpha_ncolors; i++) {
                    palette_saved_alpha[i] = surface->palette->colors[i].a;
                    surface->palette->colors[i].a = SDL_ALPHA_OPAQUE;
                }
            }
        }
    }

    // Transform colorkey to alpha. for cases where source palette has duplicate values, and colorkey is one of them
    if (copy_flags & SDL_COPY_COLORKEY) {
        if (surface->palette && !palette) {
            palette_ck_transform = true;
            palette_ck_value = surface->palette->colors[surface->map.info.colorkey].a;
            surface->palette->colors[surface->map.info.colorkey].a = SDL_ALPHA_TRANSPARENT;
        }
    }

    result = SDL_BlitSurfaceUnchecked(surface, &bounds, convert, &bounds);

    // Restore colorkey alpha value
    if (palette_ck_transform) {
        surface->palette->colors[surface->map.info.colorkey].a = palette_ck_value;
    }

    // Restore palette alpha values
    if (palette_saved_alpha) {
        int i;
        for (i = 0; i < palette_saved_alpha_ncolors; i++) {
            surface->palette->colors[i].a = palette_saved_alpha[i];
        }
        SDL_stack_free(palette_saved_alpha);
    }

    // Clean up the original surface, and update converted surface
    convert->map.info.r = copy_color.r;
    convert->map.info.g = copy_color.g;
    convert->map.info.b = copy_color.b;
    convert->map.info.a = copy_color.a;
    convert->map.info.flags =
        (copy_flags &
         ~(SDL_COPY_COLORKEY | SDL_COPY_BLEND | SDL_COPY_RLE_DESIRED | SDL_COPY_RLE_COLORKEY |
           SDL_COPY_RLE_ALPHAKEY));
    surface->map.info.r = copy_color.r;
    surface->map.info.g = copy_color.g;
    surface->map.info.b = copy_color.b;
    surface->map.info.a = copy_color.a;
    surface->map.info.flags = copy_flags;
    SDL_InvalidateMap(&surface->map);

    // SDL_BlitSurfaceUnchecked failed, and so the conversion
    if (!result) {
        goto error;
    }

    if (copy_flags & SDL_COPY_COLORKEY) {
        bool set_colorkey_by_color = false;
        bool convert_colorkey = true;

        if (surface->palette) {
            if (palette &&
                surface->palette->ncolors <= palette->ncolors &&
                (SDL_memcmp(surface->palette->colors, palette->colors,
                            surface->palette->ncolors * sizeof(SDL_Color)) == 0)) {
                // The palette is identical, just set the same colorkey
                SDL_SetSurfaceColorKey(convert, true, surface->map.info.colorkey);
            } else if (!palette) {
                if (SDL_ISPIXELFORMAT_ALPHA(format)) {
                    // No need to add the colorkey, transparency is in the alpha channel
                } else {
                    // Only set the colorkey information
                    set_colorkey_by_color = true;
                    convert_colorkey = false;
                }
            } else {
                set_colorkey_by_color = true;
            }
        } else {
            set_colorkey_by_color = true;
        }

        if (set_colorkey_by_color) {
            SDL_Surface *tmp;
            SDL_Surface *tmp2;
            int converted_colorkey = 0;

            // Create a dummy surface to get the colorkey converted
            tmp = SDL_CreateSurface(1, 1, surface->format);
            if (!tmp) {
                goto error;
            }

            // Share the palette, if any
            if (surface->palette) {
                SDL_SetSurfacePalette(tmp, surface->palette);
            }

            SDL_FillSurfaceRect(tmp, NULL, surface->map.info.colorkey);

            tmp->map.info.flags &= ~SDL_COPY_COLORKEY;

            // Conversion of the colorkey
            tmp2 = SDL_ConvertSurfaceAndColorspace(tmp, format, palette, colorspace, props);
            if (!tmp2) {
                SDL_DestroySurface(tmp);
                goto error;
            }

            // Get the converted colorkey
            SDL_memcpy(&converted_colorkey, tmp2->pixels, tmp2->fmt->bytes_per_pixel);

            SDL_DestroySurface(tmp);
            SDL_DestroySurface(tmp2);

            // Set the converted colorkey on the new surface
            SDL_SetSurfaceColorKey(convert, true, converted_colorkey);

            // This is needed when converting for 3D texture upload
            if (convert_colorkey) {
                SDL_ConvertColorkeyToAlpha(convert, true);
            }
        }
    }

end:
    if (temp_palette) {
        SDL_DestroyPalette(temp_palette);
    }

    SDL_SetSurfaceClipRect(convert, &surface->clip_rect);

    /* Enable alpha blending by default if the new surface has an
     * alpha channel or alpha modulation */
    if (SDL_ISPIXELFORMAT_ALPHA(format) ||
        (copy_flags & SDL_COPY_MODULATE_ALPHA)) {
        SDL_SetSurfaceBlendMode(convert, SDL_BLENDMODE_BLEND);
    }
    if (copy_flags & SDL_COPY_RLE_DESIRED) {
        SDL_SetSurfaceRLE(convert, true);
    }

    // Copy alternate images
    for (int i = 0; i < surface->num_images; ++i) {
        if (!SDL_AddSurfaceAlternateImage(convert, surface->images[i])) {
            goto error;
        }
    }

    // We're ready to go!
    return convert;

error:
    if (temp_palette) {
        SDL_DestroyPalette(temp_palette);
    }
    if (convert) {
        SDL_DestroySurface(convert);
    }
    return NULL;
}

SDL_Surface *SDL_DuplicateSurface(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    return SDL_ConvertSurfaceAndColorspace(surface, surface->format, surface->palette, surface->colorspace, surface->props);
}

SDL_Surface *SDL_ScaleSurface(SDL_Surface *surface, int width, int height, SDL_ScaleMode scaleMode)
{
    SDL_Surface *convert = NULL;
    Uint32 copy_flags;
    SDL_Color copy_color;
    bool rc;

    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        goto error;
    }

    if (SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        // We can't directly scale a YUV surface (yet!)
        SDL_Surface *tmp = SDL_CreateSurface(surface->w, surface->h, SDL_PIXELFORMAT_ARGB8888);
        if (!tmp) {
            return NULL;
        }

        SDL_Surface *scaled = SDL_ScaleSurface(tmp, width, height, scaleMode);
        SDL_DestroySurface(tmp);
        if (!scaled) {
            return NULL;
        }
        tmp = scaled;

        SDL_Surface *result = SDL_ConvertSurfaceAndColorspace(tmp, surface->format, NULL, surface->colorspace, surface->props);
        SDL_DestroySurface(tmp);
        return result;
    }

    // Create a new surface with the desired size
    convert = SDL_CreateSurface(width, height, surface->format);
    if (!convert) {
        goto error;
    }
    SDL_SetSurfacePalette(convert, surface->palette);
    SDL_SetSurfaceColorspace(convert, surface->colorspace);

    // Save the original copy flags
    copy_flags = surface->map.info.flags;
    copy_color.r = surface->map.info.r;
    copy_color.g = surface->map.info.g;
    copy_color.b = surface->map.info.b;
    copy_color.a = surface->map.info.a;
    surface->map.info.r = 0xFF;
    surface->map.info.g = 0xFF;
    surface->map.info.b = 0xFF;
    surface->map.info.a = 0xFF;
    surface->map.info.flags = (copy_flags & (SDL_COPY_RLE_COLORKEY | SDL_COPY_RLE_ALPHAKEY));
    SDL_InvalidateMap(&surface->map);

    rc = SDL_BlitSurfaceScaled(surface, NULL, convert, NULL, scaleMode);

    // Clean up the original surface, and update converted surface
    convert->map.info.r = copy_color.r;
    convert->map.info.g = copy_color.g;
    convert->map.info.b = copy_color.b;
    convert->map.info.a = copy_color.a;
    convert->map.info.flags = (copy_flags & ~(SDL_COPY_RLE_COLORKEY | SDL_COPY_RLE_ALPHAKEY));
    surface->map.info.r = copy_color.r;
    surface->map.info.g = copy_color.g;
    surface->map.info.b = copy_color.b;
    surface->map.info.a = copy_color.a;
    surface->map.info.flags = copy_flags;
    SDL_InvalidateMap(&surface->map);

    // SDL_BlitSurfaceScaled failed, and so the conversion
    if (!rc) {
        goto error;
    }

    // We're ready to go!
    return convert;

error:
    if (convert) {
        SDL_DestroySurface(convert);
    }
    return NULL;
}

SDL_Surface *SDL_ConvertSurface(SDL_Surface *surface, SDL_PixelFormat format)
{
    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return NULL;
    }

    return SDL_ConvertSurfaceAndColorspace(surface, format, NULL, SDL_GetDefaultColorspaceForFormat(format), surface->props);
}

SDL_Surface *SDL_DuplicatePixels(int width, int height, SDL_PixelFormat format, SDL_Colorspace colorspace, void *pixels, int pitch)
{
    SDL_Surface *surface = SDL_CreateSurface(width, height, format);
    if (surface) {
        int length = width * SDL_BYTESPERPIXEL(format);
        Uint8 *src = (Uint8 *)pixels;
        Uint8 *dst = (Uint8 *)surface->pixels;
        int rows = height;
        while (rows--) {
            SDL_memcpy(dst, src, length);
            dst += surface->pitch;
            src += pitch;
        }

        SDL_SetSurfaceColorspace(surface, colorspace);
    }
    return surface;
}

bool SDL_ConvertPixelsAndColorspace(int width, int height,
                      SDL_PixelFormat src_format, SDL_Colorspace src_colorspace, SDL_PropertiesID src_properties, const void *src, int src_pitch,
                      SDL_PixelFormat dst_format, SDL_Colorspace dst_colorspace, SDL_PropertiesID dst_properties, void *dst, int dst_pitch)
{
    SDL_Surface src_surface;
    SDL_Surface dst_surface;
    SDL_Rect rect;
    void *nonconst_src = (void *)src;
    bool result;

    if (!src) {
        return SDL_InvalidParamError("src");
    }
    if (!src_pitch) {
        return SDL_InvalidParamError("src_pitch");
    }
    if (!dst) {
        return SDL_InvalidParamError("dst");
    }
    if (!dst_pitch) {
        return SDL_InvalidParamError("dst_pitch");
    }

    if (src_colorspace == SDL_COLORSPACE_UNKNOWN) {
        src_colorspace = SDL_GetDefaultColorspaceForFormat(src_format);
    }
    if (dst_colorspace == SDL_COLORSPACE_UNKNOWN) {
        dst_colorspace = SDL_GetDefaultColorspaceForFormat(dst_format);
    }

    if (src_format == SDL_PIXELFORMAT_MJPG) {
        return SDL_ConvertPixels_STB(width, height, src_format, src_colorspace, src_properties, src, src_pitch, dst_format, dst_colorspace, dst_properties, dst, dst_pitch);
    }

#ifdef SDL_HAVE_YUV
    if (SDL_ISPIXELFORMAT_FOURCC(src_format) && SDL_ISPIXELFORMAT_FOURCC(dst_format)) {
        return SDL_ConvertPixels_YUV_to_YUV(width, height, src_format, src_colorspace, src_properties, src, src_pitch, dst_format, dst_colorspace, dst_properties, dst, dst_pitch);
    } else if (SDL_ISPIXELFORMAT_FOURCC(src_format)) {
        return SDL_ConvertPixels_YUV_to_RGB(width, height, src_format, src_colorspace, src_properties, src, src_pitch, dst_format, dst_colorspace, dst_properties, dst, dst_pitch);
    } else if (SDL_ISPIXELFORMAT_FOURCC(dst_format)) {
        return SDL_ConvertPixels_RGB_to_YUV(width, height, src_format, src_colorspace, src_properties, src, src_pitch, dst_format, dst_colorspace, dst_properties, dst, dst_pitch);
    }
#else
    if (SDL_ISPIXELFORMAT_FOURCC(src_format) || SDL_ISPIXELFORMAT_FOURCC(dst_format)) {
        return SDL_SetError("SDL not built with YUV support");
    }
#endif

    // Fast path for same format copy
    if (src_format == dst_format && src_colorspace == dst_colorspace) {
        if (src_pitch == dst_pitch) {
            SDL_memcpy(dst, src, height * src_pitch);
        } else {
            int i;
            const int bpp = SDL_BYTESPERPIXEL(src_format);
            width *= bpp;
            for (i = height; i--;) {
                SDL_memcpy(dst, src, width);
                src = (const Uint8 *)src + src_pitch;
                dst = (Uint8 *)dst + dst_pitch;
            }
        }
        return true;
    }

    if (!SDL_InitializeSurface(&src_surface, width, height, src_format, src_colorspace, src_properties, nonconst_src, src_pitch, true)) {
        return false;
    }
    SDL_SetSurfaceBlendMode(&src_surface, SDL_BLENDMODE_NONE);

    if (!SDL_InitializeSurface(&dst_surface, width, height, dst_format, dst_colorspace, dst_properties, dst, dst_pitch, true)) {
        return false;
    }

    // Set up the rect and go!
    rect.x = 0;
    rect.y = 0;
    rect.w = width;
    rect.h = height;
    result = SDL_BlitSurfaceUnchecked(&src_surface, &rect, &dst_surface, &rect);

    SDL_DestroySurface(&src_surface);
    SDL_DestroySurface(&dst_surface);

    return result;
}

bool SDL_ConvertPixels(int width, int height, SDL_PixelFormat src_format, const void *src, int src_pitch, SDL_PixelFormat dst_format, void *dst, int dst_pitch)
{
    return SDL_ConvertPixelsAndColorspace(width, height,
                      src_format, SDL_COLORSPACE_UNKNOWN, 0, src, src_pitch,
                      dst_format, SDL_COLORSPACE_UNKNOWN, 0, dst, dst_pitch);
}

/*
 * Premultiply the alpha on a block of pixels
 *
 * Here are some ideas for optimization:
 * https://github.com/Wizermil/premultiply_alpha/tree/master/premultiply_alpha
 * https://developer.arm.com/documentation/101964/0201/Pre-multiplied-alpha-channel-data
 */

static void SDL_PremultiplyAlpha_AXYZ8888(int width, int height, const void *src, int src_pitch, void *dst, int dst_pitch)
{
    int c;
    Uint32 srcpixel;
    Uint32 srcR, srcG, srcB, srcA;
    Uint32 dstpixel;
    Uint32 dstR, dstG, dstB, dstA;

    while (height--) {
        const Uint32 *src_px = (const Uint32 *)src;
        Uint32 *dst_px = (Uint32 *)dst;
        for (c = width; c; --c) {
            // Component bytes extraction.
            srcpixel = *src_px++;
            RGBA_FROM_ARGB8888(srcpixel, srcR, srcG, srcB, srcA);

            // Alpha pre-multiplication of each component.
            dstA = srcA;
            dstR = (srcA * srcR) / 255;
            dstG = (srcA * srcG) / 255;
            dstB = (srcA * srcB) / 255;

            // ARGB8888 pixel recomposition.
            ARGB8888_FROM_RGBA(dstpixel, dstR, dstG, dstB, dstA);
            *dst_px++ = dstpixel;
        }
        src = (const Uint8 *)src + src_pitch;
        dst = (Uint8 *)dst + dst_pitch;
    }
}

static void SDL_PremultiplyAlpha_XYZA8888(int width, int height, const void *src, int src_pitch, void *dst, int dst_pitch)
{
    int c;
    Uint32 srcpixel;
    Uint32 srcR, srcG, srcB, srcA;
    Uint32 dstpixel;
    Uint32 dstR, dstG, dstB, dstA;

    while (height--) {
        const Uint32 *src_px = (const Uint32 *)src;
        Uint32 *dst_px = (Uint32 *)dst;
        for (c = width; c; --c) {
            // Component bytes extraction.
            srcpixel = *src_px++;
            RGBA_FROM_RGBA8888(srcpixel, srcR, srcG, srcB, srcA);

            // Alpha pre-multiplication of each component.
            dstA = srcA;
            dstR = (srcA * srcR) / 255;
            dstG = (srcA * srcG) / 255;
            dstB = (srcA * srcB) / 255;

            // RGBA8888 pixel recomposition.
            RGBA8888_FROM_RGBA(dstpixel, dstR, dstG, dstB, dstA);
            *dst_px++ = dstpixel;
        }
        src = (const Uint8 *)src + src_pitch;
        dst = (Uint8 *)dst + dst_pitch;
    }
}

static void SDL_PremultiplyAlpha_AXYZ128(int width, int height, const void *src, int src_pitch, void *dst, int dst_pitch)
{
    int c;
    float flR, flG, flB, flA;

    while (height--) {
        const float *src_px = (const float *)src;
        float *dst_px = (float *)dst;
        for (c = width; c; --c) {
            flA = *src_px++;
            flR = *src_px++;
            flG = *src_px++;
            flB = *src_px++;

            // Alpha pre-multiplication of each component.
            flR *= flA;
            flG *= flA;
            flB *= flA;

            *dst_px++ = flA;
            *dst_px++ = flR;
            *dst_px++ = flG;
            *dst_px++ = flB;
        }
        src = (const Uint8 *)src + src_pitch;
        dst = (Uint8 *)dst + dst_pitch;
    }
}

static bool SDL_PremultiplyAlphaPixelsAndColorspace(int width, int height, SDL_PixelFormat src_format, SDL_Colorspace src_colorspace, SDL_PropertiesID src_properties, const void *src, int src_pitch, SDL_PixelFormat dst_format, SDL_Colorspace dst_colorspace, SDL_PropertiesID dst_properties, void *dst, int dst_pitch, bool linear)
{
    SDL_Surface *convert = NULL;
    void *final_dst = dst;
    int final_dst_pitch = dst_pitch;
    SDL_PixelFormat format;
    SDL_Colorspace colorspace;
    bool result = false;

    if (!src) {
        return SDL_InvalidParamError("src");
    }
    if (!src_pitch) {
        return SDL_InvalidParamError("src_pitch");
    }
    if (!dst) {
        return SDL_InvalidParamError("dst");
    }
    if (!dst_pitch) {
        return SDL_InvalidParamError("dst_pitch");
    }

    // Use a high precision format if we're converting to linear colorspace or using high precision pixel formats
    if (linear ||
        SDL_ISPIXELFORMAT_10BIT(src_format) || SDL_BITSPERPIXEL(src_format) > 32 ||
        SDL_ISPIXELFORMAT_10BIT(dst_format) || SDL_BITSPERPIXEL(dst_format) > 32) {
        if (src_format == SDL_PIXELFORMAT_ARGB128_FLOAT ||
            src_format == SDL_PIXELFORMAT_ABGR128_FLOAT) {
            format = src_format;
        } else {
            format = SDL_PIXELFORMAT_ARGB128_FLOAT;
        }
    } else {
        if (src_format == SDL_PIXELFORMAT_ARGB8888 ||
            src_format == SDL_PIXELFORMAT_ABGR8888 ||
            src_format == SDL_PIXELFORMAT_RGBA8888 ||
            src_format == SDL_PIXELFORMAT_BGRA8888) {
            format = src_format;
        } else {
            format = SDL_PIXELFORMAT_ARGB8888;
        }
    }
    if (linear) {
        colorspace = SDL_COLORSPACE_SRGB_LINEAR;
    } else {
        colorspace = SDL_COLORSPACE_SRGB;
    }

    if (src_format != format || src_colorspace != colorspace) {
        convert = SDL_CreateSurface(width, height, format);
        if (!convert) {
            goto done;
        }
        if (!SDL_ConvertPixelsAndColorspace(width, height, src_format, src_colorspace, src_properties, src, src_pitch, format, colorspace, 0, convert->pixels, convert->pitch)) {
            goto done;
        }

        src = convert->pixels;
        src_pitch = convert->pitch;
        dst = convert->pixels;
        dst_pitch = convert->pitch;

    } else if (dst_format != format || dst_colorspace != colorspace) {
        convert = SDL_CreateSurface(width, height, format);
        if (!convert) {
            goto done;
        }
        dst = convert->pixels;
        dst_pitch = convert->pitch;
    }

    switch (format) {
    case SDL_PIXELFORMAT_ARGB8888:
    case SDL_PIXELFORMAT_ABGR8888:
        SDL_PremultiplyAlpha_AXYZ8888(width, height, src, src_pitch, dst, dst_pitch);
        break;
    case SDL_PIXELFORMAT_RGBA8888:
    case SDL_PIXELFORMAT_BGRA8888:
        SDL_PremultiplyAlpha_XYZA8888(width, height, src, src_pitch, dst, dst_pitch);
        break;
    case SDL_PIXELFORMAT_ARGB128_FLOAT:
    case SDL_PIXELFORMAT_ABGR128_FLOAT:
        SDL_PremultiplyAlpha_AXYZ128(width, height, src, src_pitch, dst, dst_pitch);
        break;
    default:
        SDL_SetError("Unexpected internal pixel format");
        goto done;
    }

    if (dst != final_dst) {
        if (!SDL_ConvertPixelsAndColorspace(width, height, format, colorspace, 0, convert->pixels, convert->pitch, dst_format, dst_colorspace, dst_properties, final_dst, final_dst_pitch)) {
            goto done;
        }
    }
    result = true;

done:
    if (convert) {
        SDL_DestroySurface(convert);
    }
    return result;
}

bool SDL_PremultiplyAlpha(int width, int height,
                         SDL_PixelFormat src_format, const void *src, int src_pitch,
                         SDL_PixelFormat dst_format, void *dst, int dst_pitch, bool linear)
{
    SDL_Colorspace src_colorspace = SDL_GetDefaultColorspaceForFormat(src_format);
    SDL_Colorspace dst_colorspace = SDL_GetDefaultColorspaceForFormat(dst_format);

    return SDL_PremultiplyAlphaPixelsAndColorspace(width, height, src_format, src_colorspace, 0, src, src_pitch, dst_format, dst_colorspace, 0, dst, dst_pitch, linear);
}

bool SDL_PremultiplySurfaceAlpha(SDL_Surface *surface, bool linear)
{
    SDL_Colorspace colorspace;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    colorspace = surface->colorspace;

    return SDL_PremultiplyAlphaPixelsAndColorspace(surface->w, surface->h, surface->format, colorspace, surface->props, surface->pixels, surface->pitch, surface->format, colorspace, surface->props, surface->pixels, surface->pitch, linear);
}

bool SDL_ClearSurface(SDL_Surface *surface, float r, float g, float b, float a)
{
    SDL_Rect clip_rect;
    bool result = false;

    if (!SDL_SurfaceValid(surface)) {
        return SDL_InvalidParamError("surface");
    }

    SDL_GetSurfaceClipRect(surface, &clip_rect);
    SDL_SetSurfaceClipRect(surface, NULL);

    if (!SDL_ISPIXELFORMAT_FOURCC(surface->format) &&
        SDL_BYTESPERPIXEL(surface->format) <= sizeof(Uint32)) {
        Uint32 color;

        color = SDL_MapSurfaceRGBA(surface,
                    (Uint8)SDL_roundf(SDL_clamp(r, 0.0f, 1.0f) * 255.0f),
                    (Uint8)SDL_roundf(SDL_clamp(g, 0.0f, 1.0f) * 255.0f),
                    (Uint8)SDL_roundf(SDL_clamp(b, 0.0f, 1.0f) * 255.0f),
                    (Uint8)SDL_roundf(SDL_clamp(a, 0.0f, 1.0f) * 255.0f));
        result = SDL_FillSurfaceRect(surface, NULL, color);
    } else if (SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        // We can't directly set an RGB value on a YUV surface
        SDL_Surface *tmp = SDL_CreateSurface(surface->w, surface->h, SDL_PIXELFORMAT_ARGB8888);
        if (!tmp) {
            goto done;
        }

        if (SDL_ClearSurface(tmp, r, g, b, a)) {
            result = SDL_ConvertPixelsAndColorspace(surface->w, surface->h, tmp->format, tmp->colorspace, tmp->props, tmp->pixels, tmp->pitch, surface->format, surface->colorspace, surface->props, surface->pixels, surface->pitch);
        }
        SDL_DestroySurface(tmp);
    } else {
        // Take advantage of blit color conversion
        SDL_Surface *tmp = SDL_CreateSurface(1, 1, SDL_PIXELFORMAT_RGBA128_FLOAT);
        if (!tmp) {
            goto done;
        }
        SDL_SetSurfaceColorspace(tmp, surface->colorspace);
        SDL_SetSurfaceBlendMode(tmp, SDL_BLENDMODE_NONE);

        float *pixels = (float *)tmp->pixels;
        pixels[0] = r;
        pixels[1] = g;
        pixels[2] = b;
        pixels[3] = a;

        result = SDL_BlitSurfaceScaled(tmp, NULL, surface, NULL, SDL_SCALEMODE_NEAREST);
        SDL_DestroySurface(tmp);
    }

done:
    SDL_SetSurfaceClipRect(surface, &clip_rect);

    return result;
}

Uint32 SDL_MapSurfaceRGB(SDL_Surface *surface, Uint8 r, Uint8 g, Uint8 b)
{
    return SDL_MapSurfaceRGBA(surface, r, g, b, SDL_ALPHA_OPAQUE);
}

Uint32 SDL_MapSurfaceRGBA(SDL_Surface *surface, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    if (!SDL_SurfaceValid(surface)) {
        SDL_InvalidParamError("surface");
        return true;
    }
    return SDL_MapRGBA(surface->fmt, surface->palette, r, g, b, a);
}

// This function Copyright 2023 Collabora Ltd., contributed to SDL under the ZLib license
bool SDL_ReadSurfacePixel(SDL_Surface *surface, int x, int y, Uint8 *r, Uint8 *g, Uint8 *b, Uint8 *a)
{
    Uint32 pixel = 0;
    size_t bytes_per_pixel;
    Uint8 unused;
    Uint8 *p;
    bool result = false;

    if (r) {
        *r = 0;
    } else {
        r = &unused;
    }

    if (g) {
        *g = 0;
    } else {
        g = &unused;
    }

    if (b) {
        *b = 0;
    } else {
        b = &unused;
    }

    if (a) {
        *a = 0;
    } else {
        a = &unused;
    }

    if (!SDL_SurfaceValid(surface) || !surface->format || !surface->pixels) {
        return SDL_InvalidParamError("surface");
    }

    if (x < 0 || x >= surface->w) {
        return SDL_InvalidParamError("x");
    }

    if (y < 0 || y >= surface->h) {
        return SDL_InvalidParamError("y");
    }

    bytes_per_pixel = SDL_BYTESPERPIXEL(surface->format);

    if (SDL_MUSTLOCK(surface)) {
        if (!SDL_LockSurface(surface)) {
            return false;
        }
    }

    p = (Uint8 *)surface->pixels + y * surface->pitch + x * bytes_per_pixel;

    if (bytes_per_pixel <= sizeof(pixel) && !SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        /* Fill the appropriate number of least-significant bytes of pixel,
         * leaving the most-significant bytes set to zero */
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        SDL_memcpy(((Uint8 *)&pixel) + (sizeof(pixel) - bytes_per_pixel), p, bytes_per_pixel);
#else
        SDL_memcpy(&pixel, p, bytes_per_pixel);
#endif
        SDL_GetRGBA(pixel, surface->fmt, surface->palette, r, g, b, a);
        result = true;
    } else if (SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        // FIXME: We need code to extract a single macroblock from a YUV surface
        SDL_Surface *converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ARGB8888);
        if (converted) {
            result = SDL_ReadSurfacePixel(converted, x, y, r, g, b, a);
            SDL_DestroySurface(converted);
        }
    } else {
        // This is really slow, but it gets the job done
        Uint8 rgba[4];

        if (SDL_ConvertPixelsAndColorspace(1, 1, surface->format, surface->colorspace, surface->props, p, surface->pitch, SDL_PIXELFORMAT_RGBA32, SDL_COLORSPACE_SRGB, 0, rgba, sizeof(rgba))) {
            *r = rgba[0];
            *g = rgba[1];
            *b = rgba[2];
            *a = rgba[3];
            result = true;
        }
    }

    if (SDL_MUSTLOCK(surface)) {
        SDL_UnlockSurface(surface);
    }
    return result;
}

bool SDL_ReadSurfacePixelFloat(SDL_Surface *surface, int x, int y, float *r, float *g, float *b, float *a)
{
    float unused;
    bool result = false;

    if (r) {
        *r = 0.0f;
    } else {
        r = &unused;
    }

    if (g) {
        *g = 0.0f;
    } else {
        g = &unused;
    }

    if (b) {
        *b = 0.0f;
    } else {
        b = &unused;
    }

    if (a) {
        *a = 0.0f;
    } else {
        a = &unused;
    }

    if (!SDL_SurfaceValid(surface) || !surface->format || !surface->pixels) {
        return SDL_InvalidParamError("surface");
    }

    if (x < 0 || x >= surface->w) {
        return SDL_InvalidParamError("x");
    }

    if (y < 0 || y >= surface->h) {
        return SDL_InvalidParamError("y");
    }

    if (SDL_BYTESPERPIXEL(surface->format) <= sizeof(Uint32) && !SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        Uint8 r8, g8, b8, a8;

        if (SDL_ReadSurfacePixel(surface, x, y, &r8, &g8, &b8, &a8)) {
            *r = (float)r8 / 255.0f;
            *g = (float)g8 / 255.0f;
            *b = (float)b8 / 255.0f;
            *a = (float)a8 / 255.0f;
            result = true;
        }
    } else if (SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        // FIXME: We need code to extract a single macroblock from a YUV surface
        SDL_Surface *converted = SDL_ConvertSurface(surface, SDL_PIXELFORMAT_ARGB8888);
        if (converted) {
            result = SDL_ReadSurfacePixelFloat(converted, x, y, r, g, b, a);
            SDL_DestroySurface(converted);
        }
    } else {
        // This is really slow, but it gets the job done
        float rgba[4];
        Uint8 *p;

        if (SDL_MUSTLOCK(surface)) {
            if (!SDL_LockSurface(surface)) {
                return false;
            }
        }

        p = (Uint8 *)surface->pixels + y * surface->pitch + x * SDL_BYTESPERPIXEL(surface->format);

        if (surface->format == SDL_PIXELFORMAT_RGBA128_FLOAT) {
            SDL_memcpy(rgba, p, sizeof(rgba));
            result = true;
        } else {
            SDL_Colorspace src_colorspace = surface->colorspace;
            SDL_Colorspace dst_colorspace = (src_colorspace == SDL_COLORSPACE_SRGB_LINEAR ? SDL_COLORSPACE_SRGB_LINEAR : SDL_COLORSPACE_SRGB);

            if (SDL_ConvertPixelsAndColorspace(1, 1, surface->format, src_colorspace, surface->props, p, surface->pitch, SDL_PIXELFORMAT_RGBA128_FLOAT, dst_colorspace, 0, rgba, sizeof(rgba))) {
                result = true;
            }
        }

        if (result) {
            *r = rgba[0];
            *g = rgba[1];
            *b = rgba[2];
            *a = rgba[3];
        }

        if (SDL_MUSTLOCK(surface)) {
            SDL_UnlockSurface(surface);
        }
    }
    return result;
}

bool SDL_WriteSurfacePixel(SDL_Surface *surface, int x, int y, Uint8 r, Uint8 g, Uint8 b, Uint8 a)
{
    Uint32 pixel = 0;
    size_t bytes_per_pixel;
    Uint8 *p;
    bool result = false;

    if (!SDL_SurfaceValid(surface) || !surface->format || !surface->pixels) {
        return SDL_InvalidParamError("surface");
    }

    if (x < 0 || x >= surface->w) {
        return SDL_InvalidParamError("x");
    }

    if (y < 0 || y >= surface->h) {
        return SDL_InvalidParamError("y");
    }

    bytes_per_pixel = SDL_BYTESPERPIXEL(surface->format);

    if (SDL_MUSTLOCK(surface)) {
        if (!SDL_LockSurface(surface)) {
            return false;
        }
    }

    p = (Uint8 *)surface->pixels + y * surface->pitch + x * bytes_per_pixel;

    if (bytes_per_pixel <= sizeof(pixel) && !SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        pixel = SDL_MapRGBA(surface->fmt, surface->palette, r, g, b, a);
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
        SDL_memcpy(p, ((Uint8 *)&pixel) + (sizeof(pixel) - bytes_per_pixel), bytes_per_pixel);
#else
        SDL_memcpy(p, &pixel, bytes_per_pixel);
#endif
        result = true;
    } else if (SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        result = SDL_Unsupported();
    } else {
        // This is really slow, but it gets the job done
        Uint8 rgba[4];

        rgba[0] = r;
        rgba[1] = g;
        rgba[2] = b;
        rgba[3] = a;
        result = SDL_ConvertPixelsAndColorspace(1, 1, SDL_PIXELFORMAT_RGBA32, SDL_COLORSPACE_SRGB, 0, rgba, sizeof(rgba), surface->format, surface->colorspace, surface->props, p, surface->pitch);
    }

    if (SDL_MUSTLOCK(surface)) {
        SDL_UnlockSurface(surface);
    }
    return result;
}

bool SDL_WriteSurfacePixelFloat(SDL_Surface *surface, int x, int y, float r, float g, float b, float a)
{
    bool result = false;

    if (!SDL_SurfaceValid(surface) || !surface->format || !surface->pixels) {
        return SDL_InvalidParamError("surface");
    }

    if (x < 0 || x >= surface->w) {
        return SDL_InvalidParamError("x");
    }

    if (y < 0 || y >= surface->h) {
        return SDL_InvalidParamError("y");
    }

    if (SDL_BYTESPERPIXEL(surface->format) <= sizeof(Uint32) && !SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        Uint8 r8, g8, b8, a8;

        r8 = (Uint8)SDL_round(SDL_clamp(r, 0.0f, 1.0f) * 255.0f);
        g8 = (Uint8)SDL_round(SDL_clamp(g, 0.0f, 1.0f) * 255.0f);
        b8 = (Uint8)SDL_round(SDL_clamp(b, 0.0f, 1.0f) * 255.0f);
        a8 = (Uint8)SDL_round(SDL_clamp(a, 0.0f, 1.0f) * 255.0f);
        if (SDL_WriteSurfacePixel(surface, x, y, r8, g8, b8, a8)) {
            result = true;
        }
    } else if (SDL_ISPIXELFORMAT_FOURCC(surface->format)) {
        result = SDL_Unsupported();
    } else {
        // This is really slow, but it gets the job done
        float rgba[4];
        Uint8 *p;

        if (SDL_MUSTLOCK(surface)) {
            if (!SDL_LockSurface(surface)) {
                return false;
            }
        }

        p = (Uint8 *)surface->pixels + y * surface->pitch + x * SDL_BYTESPERPIXEL(surface->format);

        rgba[0] = r;
        rgba[1] = g;
        rgba[2] = b;
        rgba[3] = a;

        if (surface->format == SDL_PIXELFORMAT_RGBA128_FLOAT) {
            SDL_memcpy(p, rgba, sizeof(rgba));
            result = true;
        } else {
            SDL_Colorspace dst_colorspace = surface->colorspace;
            SDL_Colorspace src_colorspace = (dst_colorspace == SDL_COLORSPACE_SRGB_LINEAR ? SDL_COLORSPACE_SRGB_LINEAR : SDL_COLORSPACE_SRGB);

            result = SDL_ConvertPixelsAndColorspace(1, 1, SDL_PIXELFORMAT_RGBA128_FLOAT, src_colorspace, 0, rgba, sizeof(rgba), surface->format, dst_colorspace, surface->props, p, surface->pitch);
        }

        if (SDL_MUSTLOCK(surface)) {
            SDL_UnlockSurface(surface);
        }
    }
    return result;
}

/*
 * Free a surface created by the above function.
 */
void SDL_DestroySurface(SDL_Surface *surface)
{
    if (!SDL_SurfaceValid(surface)) {
        return;
    }
    if (surface->internal_flags & SDL_INTERNAL_SURFACE_DONTFREE) {
        return;
    }
    if (--surface->refcount > 0) {
        return;
    }

    SDL_RemoveSurfaceAlternateImages(surface);

    SDL_DestroyProperties(surface->props);

    SDL_InvalidateMap(&surface->map);

    while (surface->locked > 0) {
        SDL_UnlockSurface(surface);
    }
#ifdef SDL_HAVE_RLE
    if (surface->internal_flags & SDL_INTERNAL_SURFACE_RLEACCEL) {
        SDL_UnRLESurface(surface, false);
    }
#endif
    SDL_SetSurfacePalette(surface, NULL);

    if (surface->flags & SDL_SURFACE_PREALLOCATED) {
        // Don't free
    } else if (surface->flags & SDL_SURFACE_SIMD_ALIGNED) {
        // Free aligned
        SDL_aligned_free(surface->pixels);
    } else {
        // Normal
        SDL_free(surface->pixels);
    }

    surface->reserved = NULL;

    if (!(surface->internal_flags & SDL_INTERNAL_SURFACE_STACK)) {
        SDL_free(surface);
    }
}
