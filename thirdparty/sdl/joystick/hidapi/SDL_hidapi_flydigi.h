/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

// These are values used in the controller type byte of the controller GUID

typedef enum
{
    SDL_FLYDIGI_UNKNOWN,
    SDL_FLYDIGI_APEX2 = (1 << 0),
    SDL_FLYDIGI_APEX3,
    SDL_FLYDIGI_APEX4,
    SDL_FLYDIGI_APEX5,
    SDL_FLYDIGI_VADER2 = (1 << 4),
    SDL_FLYDIGI_VADER2_PRO,
    SDL_FLYDIGI_VADER3,
    SDL_FLYDIGI_VADER3_PRO,
    SDL_FLYDIGI_VADER4_PRO,
    SDL_FLYDIGI_VADER5_PRO,
} SDL_FlyDigiControllerType;

