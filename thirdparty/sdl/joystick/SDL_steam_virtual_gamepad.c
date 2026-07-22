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

#include "SDL_joystick_c.h"
#include "SDL_steam_virtual_gamepad.h"

#ifdef SDL_PLATFORM_LINUX
#include "../core/unix/SDL_appid.h"
#endif
#ifdef SDL_PLATFORM_WIN32
#include "../core/windows/SDL_windows.h"
#else
#include <sys/types.h>
#include <sys/stat.h>
#endif

#define SDL_HINT_STEAM_VIRTUAL_GAMEPAD_INFO_FILE    "SteamVirtualGamepadInfo"

static char *SDL_steam_virtual_gamepad_info_file SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static Uint64 SDL_steam_virtual_gamepad_info_file_mtime SDL_GUARDED_BY(SDL_joystick_lock) = 0;
static Uint64 SDL_steam_virtual_gamepad_info_check_time SDL_GUARDED_BY(SDL_joystick_lock) = 0;
static SDL_SteamVirtualGamepadInfo **SDL_steam_virtual_gamepad_info SDL_GUARDED_BY(SDL_joystick_lock) = NULL;
static int SDL_steam_virtual_gamepad_info_count SDL_GUARDED_BY(SDL_joystick_lock) = 0;


static Uint64 GetFileModificationTime(const char *file)
{
    Uint64 modification_time = 0;

#ifdef SDL_PLATFORM_WIN32
    WCHAR *wFile = WIN_UTF8ToStringW(file);
    if (wFile) {
        HANDLE hFile = CreateFileW(wFile, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
        if (hFile != INVALID_HANDLE_VALUE) {
            FILETIME last_write_time;
            if (GetFileTime(hFile, NULL, NULL, &last_write_time)) {
                modification_time = last_write_time.dwHighDateTime;
                modification_time <<= 32;
                modification_time |= last_write_time.dwLowDateTime;
            }
            CloseHandle(hFile);
        }
        SDL_free(wFile);
    }
#else
    struct stat sb;

    if (stat(file, &sb) == 0) {
        modification_time = (Uint64)sb.st_mtime;
    }
#endif
    return modification_time;
}

static void SDL_FreeSteamVirtualGamepadInfo(void)
{
    int i;

    SDL_AssertJoysticksLocked();

    for (i = 0; i < SDL_steam_virtual_gamepad_info_count; ++i) {
        SDL_SteamVirtualGamepadInfo *entry = SDL_steam_virtual_gamepad_info[i];
        if (entry) {
            SDL_free(entry->name);
            SDL_free(entry);
        }
    }
    SDL_free(SDL_steam_virtual_gamepad_info);
    SDL_steam_virtual_gamepad_info = NULL;
    SDL_steam_virtual_gamepad_info_count = 0;
}

static void AddVirtualGamepadInfo(int slot, SDL_SteamVirtualGamepadInfo *info)
{
    SDL_SteamVirtualGamepadInfo *new_info;

    SDL_AssertJoysticksLocked();

    if (slot < 0) {
        return;
    }

    if (slot >= SDL_steam_virtual_gamepad_info_count) {
        SDL_SteamVirtualGamepadInfo **slots = (SDL_SteamVirtualGamepadInfo **)SDL_realloc(SDL_steam_virtual_gamepad_info, (slot + 1)*sizeof(*SDL_steam_virtual_gamepad_info));
        if (!slots) {
            return;
        }
        while (SDL_steam_virtual_gamepad_info_count <= slot) {
            slots[SDL_steam_virtual_gamepad_info_count++] = NULL;
        }
        SDL_steam_virtual_gamepad_info = slots;
    }

    if (SDL_steam_virtual_gamepad_info[slot]) {
        // We already have this slot info
        return;
    }

    new_info = (SDL_SteamVirtualGamepadInfo *)SDL_malloc(sizeof(*new_info));
    if (!new_info) {
        return;
    }
    SDL_copyp(new_info, info);
    SDL_steam_virtual_gamepad_info[slot] = new_info;
    SDL_zerop(info);
}

void SDL_InitSteamVirtualGamepadInfo(void)
{
    const char *file;

    SDL_AssertJoysticksLocked();

    // The file isn't available inside the macOS sandbox
    if (SDL_GetSandbox() == SDL_SANDBOX_MACOS) {
        return;
    }

    file = SDL_GetHint(SDL_HINT_STEAM_VIRTUAL_GAMEPAD_INFO_FILE);
    if (file && *file) {
#ifdef SDL_PLATFORM_LINUX
        // Older versions of Wine will blacklist the Steam Virtual Gamepad if
        // it appears to have the real controller's VID/PID, so ignore this.
        const char *exe = SDL_GetExeName();
        if (exe && SDL_strcmp(exe, "wine64-preloader") == 0) {
            SDL_LogDebug(SDL_LOG_CATEGORY_INPUT, "Wine launched by Steam, ignoring %s", SDL_HINT_STEAM_VIRTUAL_GAMEPAD_INFO_FILE);
            return;
        }
#endif
        SDL_steam_virtual_gamepad_info_file = SDL_strdup(file);
    }
    SDL_UpdateSteamVirtualGamepadInfo();
}

bool SDL_SteamVirtualGamepadEnabled(void)
{
    SDL_AssertJoysticksLocked();

    return (SDL_steam_virtual_gamepad_info != NULL);
}

bool SDL_UpdateSteamVirtualGamepadInfo(void)
{
    const int UPDATE_CHECK_INTERVAL_MS = 3000;
    Uint64 now;
    Uint64 mtime;
    char *data, *end, *next, *line, *value;
    size_t size;
    int slot, new_slot;
    SDL_SteamVirtualGamepadInfo info;

    SDL_AssertJoysticksLocked();

    if (!SDL_steam_virtual_gamepad_info_file) {
        return false;
    }

    now = SDL_GetTicks();
    if (SDL_steam_virtual_gamepad_info_check_time &&
        now < (SDL_steam_virtual_gamepad_info_check_time + UPDATE_CHECK_INTERVAL_MS)) {
        return false;
    }
    SDL_steam_virtual_gamepad_info_check_time = now;

    mtime = GetFileModificationTime(SDL_steam_virtual_gamepad_info_file);
    if (mtime == 0 || mtime == SDL_steam_virtual_gamepad_info_file_mtime) {
        return false;
    }

    data = (char *)SDL_LoadFile(SDL_steam_virtual_gamepad_info_file, &size);
    if (!data) {
        return false;
    }

    SDL_FreeSteamVirtualGamepadInfo();

    slot = -1;
    SDL_zero(info);

    for (next = data, end = data + size; next < end; ) {
        while (next < end && (*next == '\0' || *next == '\r' || *next == '\n')) {
            ++next;
        }

        line = next;

        while (next < end && (*next != '\r' && *next != '\n')) {
            ++next;
        }
        *next = '\0';

        if (SDL_sscanf(line, "[slot %d]", &new_slot) == 1) {
            if (slot >= 0) {
                AddVirtualGamepadInfo(slot, &info);
            }
            slot = new_slot;
        } else {
            value = SDL_strchr(line, '=');
            if (value) {
                *value++ = '\0';

                if (SDL_strcmp(line, "name") == 0) {
                    SDL_free(info.name);
                    info.name = SDL_strdup(value);
                } else if (SDL_strcmp(line, "VID") == 0) {
                    info.vendor_id = (Uint16)SDL_strtoul(value, NULL, 0);
                } else if (SDL_strcmp(line, "PID") == 0) {
                    info.product_id = (Uint16)SDL_strtoul(value, NULL, 0);
                } else if (SDL_strcmp(line, "type") == 0) {
                    info.type = SDL_GetGamepadTypeFromString(value);
                } else if (SDL_strcmp(line, "handle") == 0) {
                    info.handle = (Uint64)SDL_strtoull(value, NULL, 0);
                }
            }
        }
    }
    if (slot >= 0) {
        AddVirtualGamepadInfo(slot, &info);
    }
    SDL_free(info.name);
    SDL_free(data);

    SDL_steam_virtual_gamepad_info_file_mtime = mtime;

    return true;
}

const SDL_SteamVirtualGamepadInfo *SDL_GetSteamVirtualGamepadInfo(int slot)
{
    SDL_AssertJoysticksLocked();

    if (slot < 0 || slot >= SDL_steam_virtual_gamepad_info_count) {
        return NULL;
    }
    return SDL_steam_virtual_gamepad_info[slot];
}

void SDL_QuitSteamVirtualGamepadInfo(void)
{
    SDL_AssertJoysticksLocked();

    if (SDL_steam_virtual_gamepad_info_file) {
        SDL_FreeSteamVirtualGamepadInfo();
        SDL_free(SDL_steam_virtual_gamepad_info_file);
        SDL_steam_virtual_gamepad_info_file = NULL;
    }
}
