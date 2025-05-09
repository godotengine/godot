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

#ifndef SDL_triangle_h_
#define SDL_triangle_h_

#include "SDL_internal.h"

extern bool SDL_SW_FillTriangle(SDL_Surface *dst,
                                SDL_Point *d0, SDL_Point *d1, SDL_Point *d2,
                                SDL_BlendMode blend, SDL_Color c0, SDL_Color c1, SDL_Color c2);

extern bool SDL_SW_BlitTriangle(SDL_Surface *src,
                                SDL_Point *s0, SDL_Point *s1, SDL_Point *s2,
                                SDL_Surface *dst,
                                SDL_Point *d0, SDL_Point *d1, SDL_Point *d2,
                                SDL_Color c0, SDL_Color c1, SDL_Color c2,
                                SDL_TextureAddressMode texture_address_mode_u,
                                SDL_TextureAddressMode texture_address_mode_v);

extern void trianglepoint_2_fixedpoint(SDL_Point *a);

#endif // SDL_triangle_h_
