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
// This is included in SDL_internal.h
//#include "SDL_internal.h"

#ifndef SDL_utils_h_
#define SDL_utils_h_

// Common utility functions that aren't in the public API

// Return the smallest power of 2 greater than or equal to 'x'
extern int SDL_powerof2(int x);

extern Uint32 SDL_CalculateGCD(Uint32 a, Uint32 b);
extern void SDL_CalculateFraction(float x, int *numerator, int *denominator);

extern bool SDL_startswith(const char *string, const char *prefix);
extern bool SDL_endswith(const char *string, const char *suffix);

/** Convert URI to a local filename, stripping the "file://"
 *  preamble and hostname if present, and writes the result
 *  to the dst buffer. Since URI-encoded characters take
 *  three times the space of normal characters, src and dst
 *  can safely point to the same buffer for in situ conversion.
 *
 *  Returns the number of decoded bytes that wound up in
 *  the destination buffer, excluding the terminating NULL byte.
 *
 *  On error, -1 is returned.
 */
extern int SDL_URIToLocal(const char *src, char *dst);

typedef enum
{
    SDL_OBJECT_TYPE_UNKNOWN,
    SDL_OBJECT_TYPE_WINDOW,
    SDL_OBJECT_TYPE_RENDERER,
    SDL_OBJECT_TYPE_TEXTURE,
    SDL_OBJECT_TYPE_JOYSTICK,
    SDL_OBJECT_TYPE_GAMEPAD,
    SDL_OBJECT_TYPE_HAPTIC,
    SDL_OBJECT_TYPE_SENSOR,
    SDL_OBJECT_TYPE_HIDAPI_DEVICE,
    SDL_OBJECT_TYPE_HIDAPI_JOYSTICK,
    SDL_OBJECT_TYPE_THREAD,
    SDL_OBJECT_TYPE_TRAY,

} SDL_ObjectType;

extern Uint32 SDL_GetNextObjectID(void);
extern void SDL_SetObjectValid(void *object, SDL_ObjectType type, bool valid);
extern bool SDL_ObjectValid(void *object, SDL_ObjectType type);
extern int SDL_GetObjects(SDL_ObjectType type, void **objects, int count);
extern void SDL_SetObjectsInvalid(void);

extern const char *SDL_GetPersistentString(const char *string);

extern char *SDL_CreateDeviceName(Uint16 vendor, Uint16 product, const char *vendor_name, const char *product_name, const char *default_name);

#endif // SDL_utils_h_
