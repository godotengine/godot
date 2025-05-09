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

#ifndef SDL_pixels_c_h_
#define SDL_pixels_c_h_

// Useful functions and variables from SDL_pixel.c

#include "SDL_blit.h"


// Pixel format functions
extern void SDL_Get8888AlphaMaskAndShift(const SDL_PixelFormatDetails *fmt, Uint32 *mask, Uint32 *shift);
extern SDL_Colorspace SDL_GetDefaultColorspaceForFormat(SDL_PixelFormat pixel_format);
extern void SDL_QuitPixelFormatDetails(void);

// Colorspace conversion functions
extern float SDL_sRGBtoLinear(float v);
extern float SDL_sRGBfromLinear(float v);
extern float SDL_PQtoNits(float v);
extern float SDL_PQfromNits(float v);
extern const float *SDL_GetYCbCRtoRGBConversionMatrix(SDL_Colorspace colorspace, int w, int h, int bits_per_pixel);
extern const float *SDL_GetColorPrimariesConversionMatrix(SDL_ColorPrimaries src, SDL_ColorPrimaries dst);
extern void SDL_ConvertColorPrimaries(float *fR, float *fG, float *fB, const float *matrix);

// Blit mapping functions
extern bool SDL_ValidateMap(SDL_Surface *src, SDL_Surface *dst);
extern void SDL_InvalidateMap(SDL_BlitMap *map);
extern bool SDL_MapSurface(SDL_Surface *src, SDL_Surface *dst);

// Miscellaneous functions
extern void SDL_DitherPalette(SDL_Palette *palette);
extern Uint8 SDL_FindColor(const SDL_Palette *pal, Uint8 r, Uint8 g, Uint8 b, Uint8 a);
extern Uint8 SDL_LookupRGBAColor(SDL_HashTable *palette_map, Uint32 pixelvalue, const SDL_Palette *pal);
extern void SDL_DetectPalette(const SDL_Palette *pal, bool *is_opaque, bool *has_alpha_channel);
extern SDL_Surface *SDL_DuplicatePixels(int width, int height, SDL_PixelFormat format, SDL_Colorspace colorspace, void *pixels, int pitch);

#endif // SDL_pixels_c_h_
