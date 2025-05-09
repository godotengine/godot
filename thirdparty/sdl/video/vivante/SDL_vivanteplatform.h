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

#ifndef SDL_vivanteplatform_h_
#define SDL_vivanteplatform_h_

#ifdef SDL_VIDEO_DRIVER_VIVANTE

#include "SDL_vivantevideo.h"

#ifdef CAVIUM
#define VIVANTE_PLATFORM_CAVIUM
#elif defined(MARVELL)
#define VIVANTE_PLATFORM_MARVELL
#else
#define VIVANTE_PLATFORM_GENERIC
#endif

extern bool VIVANTE_SetupPlatform(SDL_VideoDevice *_this);
extern char *VIVANTE_GetDisplayName(SDL_VideoDevice *_this);
extern void VIVANTE_UpdateDisplayScale(SDL_VideoDevice *_this);
extern void VIVANTE_CleanupPlatform(SDL_VideoDevice *_this);

#endif // SDL_VIDEO_DRIVER_VIVANTE

#endif // SDL_vivanteplatform_h_
