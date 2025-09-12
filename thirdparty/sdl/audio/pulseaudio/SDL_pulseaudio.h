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

#ifndef SDL_pulseaudio_h_
#define SDL_pulseaudio_h_

#include <pulse/pulseaudio.h>

#include "../SDL_sysaudio.h"

struct SDL_PrivateAudioData
{
    // pulseaudio structures
    pa_stream *stream;

    // Raw mixing buffer
    Uint8 *mixbuf;

    int bytes_requested; // bytes of data the hardware wants _now_.

    const Uint8 *recordingbuf;
    int recordinglen;
};

#endif // SDL_pulseaudio_h_
