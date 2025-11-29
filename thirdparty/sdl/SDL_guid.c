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

// convert the guid to a printable string
void SDL_GUIDToString(SDL_GUID guid, char *pszGUID, int cbGUID)
{
    static const char k_rgchHexToASCII[] = "0123456789abcdef";
    int i;

    if ((!pszGUID) || (cbGUID <= 0)) {
        return;
    }

    for (i = 0; i < sizeof(guid.data) && i < (cbGUID - 1) / 2; i++) {
        // each input byte writes 2 ascii chars, and might write a null byte.
        // If we don't have room for next input byte, stop
        unsigned char c = guid.data[i];

        *pszGUID++ = k_rgchHexToASCII[c >> 4];
        *pszGUID++ = k_rgchHexToASCII[c & 0x0F];
    }
    *pszGUID = '\0';
}

/*-----------------------------------------------------------------------------
 * Purpose: Returns the 4 bit nibble for a hex character
 * Input  : c -
 * Output : unsigned char
 *-----------------------------------------------------------------------------*/
static unsigned char nibble(unsigned char c)
{
    if ((c >= '0') && (c <= '9')) {
        return c - '0';
    }

    if ((c >= 'A') && (c <= 'F')) {
        return c - 'A' + 0x0a;
    }

    if ((c >= 'a') && (c <= 'f')) {
        return c - 'a' + 0x0a;
    }

    // received an invalid character, and no real way to return an error
    // AssertMsg1(false, "Q_nibble invalid hex character '%c' ", c);
    return 0;
}

// convert the string version of a guid to the struct
SDL_GUID SDL_StringToGUID(const char *pchGUID)
{
    SDL_GUID guid;
    int maxoutputbytes = sizeof(guid);
    size_t len = SDL_strlen(pchGUID);
    Uint8 *p;
    size_t i;

    // Make sure it's even
    len = (len) & ~0x1;

    SDL_memset(&guid, 0x00, sizeof(guid));

    p = (Uint8 *)&guid;
    for (i = 0; (i < len) && ((p - (Uint8 *)&guid) < maxoutputbytes); i += 2, p++) {
        *p = (nibble((unsigned char)pchGUID[i]) << 4) | nibble((unsigned char)pchGUID[i + 1]);
    }

    return guid;
}
