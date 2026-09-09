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

/* Public domain CRC implementation adapted from:
   http://home.thep.lu.se/~bjorn/crc/crc32_simple.c

   This algorithm is compatible with the 16-bit CRC described here:
   https://www.lammertbies.nl/comm/info/crc-calculation
*/
/* NOTE: DO NOT CHANGE THIS ALGORITHM
   There is code that relies on this in the joystick code
*/

static Uint16 crc16_for_byte(Uint8 r)
{
    Uint16 crc = 0;
    int i;
    for (i = 0; i < 8; ++i) {
        crc = ((crc ^ r) & 1 ? 0xA001 : 0) ^ crc >> 1;
        r >>= 1;
    }
    return crc;
}

Uint16 SDL_crc16(Uint16 crc, const void *data, size_t len)
{
    // As an optimization we can precalculate a 256 entry table for each byte
    size_t i;
    for (i = 0; i < len; ++i) {
        crc = crc16_for_byte((Uint8)crc ^ ((const Uint8 *)data)[i]) ^ crc >> 8;
    }
    return crc;
}
