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

#include "SDL_stb_c.h"


// We currently only support JPEG, but we could add other image formats if we wanted
#ifdef SDL_HAVE_STB
#define malloc SDL_malloc
#define realloc SDL_realloc
#define free SDL_free
#undef memcpy
#define memcpy SDL_memcpy
#undef memset
#define memset SDL_memset
#undef strcmp
#define strcmp SDL_strcmp
#undef strncmp
#define strncmp SDL_strncmp
#define strtol SDL_strtol

#define abs SDL_abs
#define pow SDL_pow
#define ldexp SDL_scalbn

#define STB_IMAGE_STATIC
#define STBI_NO_THREAD_LOCALS
#define STBI_FAILURE_USERMSG
#if defined(SDL_NEON_INTRINSICS)
#define STBI_NEON
#endif
#define STBI_ONLY_JPEG
#define STBI_NO_GIF
#define STBI_NO_PNG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_NO_ZLIB
#define STBI_NO_STDIO
#define STBI_ASSERT SDL_assert
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#undef memset
#endif

#ifdef SDL_HAVE_STB
static bool SDL_ConvertPixels_MJPG_to_NV12(int width, int height, const void *src, int src_pitch, void *dst, int dst_pitch)
{
    int w = 0, h = 0, format = 0;
    stbi__context s;
    stbi__start_mem(&s, src, src_pitch);

    stbi__result_info ri;
    SDL_zero(ri);
    ri.bits_per_channel = 8;
    ri.channel_order = STBI_ORDER_RGB;
    ri.num_channels = 0;

    stbi__nv12 nv12;
    nv12.w = width;
    nv12.h = height;
    nv12.pitch = dst_pitch;
    nv12.y = (stbi_uc *)dst;
    nv12.uv = nv12.y + (nv12.h * nv12.pitch);

    void *pixels = stbi__jpeg_load(&s, &w, &h, &format, 4, &nv12, &ri);
    if (!pixels) {
        return false;
    }
    return true;
}
#endif // SDL_HAVE_STB

bool SDL_ConvertPixels_STB(int width, int height,
                           SDL_PixelFormat src_format, SDL_Colorspace src_colorspace, SDL_PropertiesID src_properties, const void *src, int src_pitch,
                           SDL_PixelFormat dst_format, SDL_Colorspace dst_colorspace, SDL_PropertiesID dst_properties, void *dst, int dst_pitch)
{
#ifdef SDL_HAVE_STB
    if (src_format == SDL_PIXELFORMAT_MJPG && dst_format == SDL_PIXELFORMAT_NV12) {
        return SDL_ConvertPixels_MJPG_to_NV12(width, height, src, src_pitch, dst, dst_pitch);
    }

    bool result;
    int w = 0, h = 0, format = 0;
    int len = (src_format == SDL_PIXELFORMAT_MJPG) ? src_pitch : (height * src_pitch);
    void *pixels = stbi_load_from_memory(src, len, &w, &h, &format, 4);
    if (!pixels) {
        return false;
    }

    if (w == width && h == height) {
        result = SDL_ConvertPixelsAndColorspace(w, h, SDL_PIXELFORMAT_RGBA32, SDL_COLORSPACE_SRGB, 0, pixels, width * 4, dst_format, dst_colorspace, dst_properties, dst, dst_pitch);
    } else {
        result = SDL_SetError("Expected image size %dx%d, actual size %dx%d", width, height, w, h);
    }
    stbi_image_free(pixels);

    return result;
#else
    return SDL_SetError("SDL not built with STB image support");
#endif
}
