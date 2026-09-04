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

#include "SDL_internal.h"

#include "SDL_poll.h"

#include <poll.h>
#include <errno.h>

#ifdef HAVE_PPOLL
#include <time.h>
#endif

int SDL_IOReady(int fd, int flags, Sint64 timeoutNS)
{
    int result;

    SDL_assert(flags & (SDL_IOR_READ | SDL_IOR_WRITE));

    // Note: We don't bother to account for elapsed time if we get EINTR
    do {
        struct pollfd info;

        info.fd = fd;
        info.events = 0;
        if (flags & SDL_IOR_READ) {
            info.events |= POLLIN | POLLPRI;
        }
        if (flags & SDL_IOR_WRITE) {
            info.events |= POLLOUT;
        }

#ifdef HAVE_PPOLL
        struct timespec *timeout = NULL;
        struct timespec ts;

        if (timeoutNS >= 0) {
            ts.tv_sec = SDL_NS_TO_SECONDS(timeoutNS);
            ts.tv_nsec = timeoutNS - SDL_SECONDS_TO_NS(ts.tv_sec);
            timeout = &ts;
        }

        result = ppoll(&info, 1, timeout, NULL);
#else
        int timeoutMS;

        if (timeoutNS > 0) {
            timeoutMS = (int)SDL_NS_TO_MS(timeoutNS + (SDL_NS_PER_MS - 1));
        } else if (timeoutNS == 0) {
            timeoutMS = 0;
        } else {
            timeoutMS = -1;
        }
        result = poll(&info, 1, timeoutMS);
#endif
    } while (result < 0 && errno == EINTR && !(flags & SDL_IOR_NO_RETRY));

    return result;
}
