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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Default cursor - it happens to be the Mac cursor, but could be anything   */

#define DEFAULT_CWIDTH  16
#define DEFAULT_CHEIGHT 16
#define DEFAULT_CHOTX   0
#define DEFAULT_CHOTY   0

// Added a real MacOS cursor, at the request of Luc-Olivier de Charrière
#define USE_MACOS_CURSOR

#ifdef USE_MACOS_CURSOR

static const unsigned char default_cdata[] = {
    0x00, 0x00,
    0x40, 0x00,
    0x60, 0x00,
    0x70, 0x00,
    0x78, 0x00,
    0x7C, 0x00,
    0x7E, 0x00,
    0x7F, 0x00,
    0x7F, 0x80,
    0x7C, 0x00,
    0x6C, 0x00,
    0x46, 0x00,
    0x06, 0x00,
    0x03, 0x00,
    0x03, 0x00,
    0x00, 0x00
};

static const unsigned char default_cmask[] = {
    0xC0, 0x00,
    0xE0, 0x00,
    0xF0, 0x00,
    0xF8, 0x00,
    0xFC, 0x00,
    0xFE, 0x00,
    0xFF, 0x00,
    0xFF, 0x80,
    0xFF, 0xC0,
    0xFF, 0xE0,
    0xFE, 0x00,
    0xEF, 0x00,
    0xCF, 0x00,
    0x87, 0x80,
    0x07, 0x80,
    0x03, 0x00
};

#else

static const unsigned char default_cdata[] = {
    0x00, 0x00,
    0x40, 0x00,
    0x60, 0x00,
    0x70, 0x00,
    0x78, 0x00,
    0x7C, 0x00,
    0x7E, 0x00,
    0x7F, 0x00,
    0x7F, 0x80,
    0x7C, 0x00,
    0x6C, 0x00,
    0x46, 0x00,
    0x06, 0x00,
    0x03, 0x00,
    0x03, 0x00,
    0x00, 0x00
};

static const unsigned char default_cmask[] = {
    0x40, 0x00,
    0xE0, 0x00,
    0xF0, 0x00,
    0xF8, 0x00,
    0xFC, 0x00,
    0xFE, 0x00,
    0xFF, 0x00,
    0xFF, 0x80,
    0xFF, 0xC0,
    0xFF, 0x80,
    0xFE, 0x00,
    0xEF, 0x00,
    0x4F, 0x00,
    0x07, 0x80,
    0x07, 0x80,
    0x03, 0x00
};

#endif // USE_MACOS_CURSOR
