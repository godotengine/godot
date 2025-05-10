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

#if defined(SDL_INPUT_LINUXEV) || defined(SDL_VIDEO_DRIVER_WAYLAND) || defined(SDL_VIDEO_DRIVER_X11)

#include "SDL_scancode_tables_c.h"

#include "scancodes_darwin.h"
#include "scancodes_linux.h"
#include "scancodes_xfree86.h"

static const struct
{
    SDL_ScancodeTable table;
    SDL_Scancode const *scancodes;
    int num_entries;
} SDL_scancode_tables[] = {
    { SDL_SCANCODE_TABLE_DARWIN, darwin_scancode_table, SDL_arraysize(darwin_scancode_table) },
    { SDL_SCANCODE_TABLE_LINUX, linux_scancode_table, SDL_arraysize(linux_scancode_table) },
    { SDL_SCANCODE_TABLE_XFREE86_1, xfree86_scancode_table, SDL_arraysize(xfree86_scancode_table) },
    { SDL_SCANCODE_TABLE_XFREE86_2, xfree86_scancode_table2, SDL_arraysize(xfree86_scancode_table2) },
    { SDL_SCANCODE_TABLE_XVNC, xvnc_scancode_table, SDL_arraysize(xvnc_scancode_table) },
};

const SDL_Scancode *SDL_GetScancodeTable(SDL_ScancodeTable table, int *num_entries)
{
    int i;

    for (i = 0; i < SDL_arraysize(SDL_scancode_tables); ++i) {
        if (table == SDL_scancode_tables[i].table) {
            *num_entries = SDL_scancode_tables[i].num_entries;
            return SDL_scancode_tables[i].scancodes;
        }
    }

    *num_entries = 0;
    return NULL;
}

SDL_Scancode SDL_GetScancodeFromTable(SDL_ScancodeTable table, int keycode)
{
    SDL_Scancode scancode = SDL_SCANCODE_UNKNOWN;
    int num_entries;
    const SDL_Scancode *scancodes = SDL_GetScancodeTable(table, &num_entries);

    if (keycode >= 0 && keycode < num_entries) {
        scancode = scancodes[keycode];
    }
    return scancode;
}

#endif // SDL_INPUT_LINUXEV || SDL_VIDEO_DRIVER_WAYLAND || SDL_VIDEO_DRIVER_X11
