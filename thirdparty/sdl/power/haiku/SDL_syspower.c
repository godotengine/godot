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

// uses BeOS euc.jp apm driver.
// !!! FIXME: does this thing even work on Haiku?

#ifndef SDL_POWER_DISABLED
#ifdef SDL_POWER_HAIKU

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <ctype.h>
#include <drivers/Drivers.h>

// These values are from apm.h ...
#define APM_DEVICE_PATH           "/dev/misc/apm"
#define APM_FUNC_OFFSET           0x5300
#define APM_FUNC_GET_POWER_STATUS 10
#define APM_DEVICE_ALL            1
#define APM_BIOS_CALL             (B_DEVICE_OP_CODES_END + 3)

bool SDL_GetPowerInfo_Haiku(SDL_PowerState *state, int *seconds, int *percent)
{
    const int fd = open("/dev/misc/apm", O_RDONLY | O_CLOEXEC);
    bool need_details = false;
    uint16 regs[6];
    uint8 ac_status;
    uint8 battery_status;
    uint8 battery_flags;
    uint8 battery_life;
    uint32 battery_time;
    int rc;

    if (fd == -1) {
        return false; // maybe some other method will work?
    }

    SDL_memset(regs, '\0', sizeof(regs));
    regs[0] = APM_FUNC_OFFSET + APM_FUNC_GET_POWER_STATUS;
    regs[1] = APM_DEVICE_ALL;
    rc = ioctl(fd, APM_BIOS_CALL, regs);
    close(fd);

    if (rc < 0) {
        return false;
    }

    ac_status = regs[1] >> 8;
    battery_status = regs[1] & 0xFF;
    battery_flags = regs[2] >> 8;
    battery_life = regs[2] & 0xFF;
    battery_time = (uint32)regs[3];

    // in theory, _something_ should be set in battery_flags, right?
    if (battery_flags == 0x00) { // older APM BIOS? Less fields.
        battery_time = 0xFFFF;
        if (battery_status == 0xFF) {
            battery_flags = 0xFF;
        } else {
            battery_flags = (1 << battery_status);
        }
    }

    if ((battery_time != 0xFFFF) && (battery_time & (1 << 15))) {
        // time is in minutes, not seconds
        battery_time = (battery_time & 0x7FFF) * 60;
    }

    if (battery_flags == 0xFF) { // unknown state
        *state = SDL_POWERSTATE_UNKNOWN;
    } else if (battery_flags & (1 << 7)) { // no battery
        *state = SDL_POWERSTATE_NO_BATTERY;
    } else if (battery_flags & (1 << 3)) { // charging
        *state = SDL_POWERSTATE_CHARGING;
        need_details = true;
    } else if (ac_status == 1) {
        *state = SDL_POWERSTATE_CHARGED; // on AC, not charging.
        need_details = true;
    } else {
        *state = SDL_POWERSTATE_ON_BATTERY; // not on AC.
        need_details = true;
    }

    *percent = -1;
    *seconds = -1;
    if (need_details) {
        const int pct = (int)battery_life;
        const int secs = (int)battery_time;

        if (pct != 255) {                       // 255 == unknown
            *percent = (pct > 100) ? 100 : pct; // clamp between 0%, 100%
        }
        if (secs != 0xFFFF) { // 0xFFFF == unknown
            *seconds = secs;
        }
    }

    return true; // the definitive answer if APM driver replied.
}

#endif // SDL_POWER_HAIKU
#endif // SDL_POWER_DISABLED
