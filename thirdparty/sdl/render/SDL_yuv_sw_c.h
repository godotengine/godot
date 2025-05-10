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

#ifndef SDL_yuv_sw_c_h_
#define SDL_yuv_sw_c_h_

#include "SDL_internal.h"

// This is the software implementation of the YUV texture support

struct SDL_SW_YUVTexture
{
    SDL_PixelFormat format;
    SDL_Colorspace colorspace;
    SDL_PixelFormat target_format;
    int w, h;
    Uint8 *pixels;

    // These are just so we don't have to allocate them separately
    int pitches[3];
    Uint8 *planes[3];

    // This is a temporary surface in case we have to stretch copy
    SDL_Surface *stretch;
    SDL_Surface *display;
};

typedef struct SDL_SW_YUVTexture SDL_SW_YUVTexture;

extern SDL_SW_YUVTexture *SDL_SW_CreateYUVTexture(SDL_PixelFormat format, SDL_Colorspace colorspace, int w, int h);
extern bool SDL_SW_QueryYUVTexturePixels(SDL_SW_YUVTexture *swdata, void **pixels, int *pitch);
extern bool SDL_SW_UpdateYUVTexture(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect, const void *pixels, int pitch);
extern bool SDL_SW_UpdateYUVTexturePlanar(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect,
                                         const Uint8 *Yplane, int Ypitch,
                                         const Uint8 *Uplane, int Upitch,
                                         const Uint8 *Vplane, int Vpitch);
extern bool SDL_SW_UpdateNVTexturePlanar(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect,
                                         const Uint8 *Yplane, int Ypitch,
                                         const Uint8 *UVplane, int UVpitch);
extern bool SDL_SW_LockYUVTexture(SDL_SW_YUVTexture *swdata, const SDL_Rect *rect, void **pixels, int *pitch);
extern void SDL_SW_UnlockYUVTexture(SDL_SW_YUVTexture *swdata);
extern bool SDL_SW_CopyYUVToRGB(SDL_SW_YUVTexture *swdata, const SDL_Rect *srcrect, SDL_PixelFormat target_format, int w, int h, void *pixels, int pitch);
extern void SDL_SW_DestroyYUVTexture(SDL_SW_YUVTexture *swdata);

#endif // SDL_yuv_sw_c_h_
