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

#ifndef SDL_directsound_h_
#define SDL_directsound_h_

#include "../../core/windows/SDL_directx.h"

#include "../SDL_sysaudio.h"

// The DirectSound objects
struct SDL_PrivateAudioData
{
    // !!! FIXME: make this a union with capture/playback sections?
    LPDIRECTSOUND sound;
    LPDIRECTSOUNDBUFFER mixbuf;
    LPDIRECTSOUNDCAPTURE capture;
    LPDIRECTSOUNDCAPTUREBUFFER capturebuf;
    int num_buffers;
    DWORD lastchunk;
    Uint8 *locked_buf;
};

#endif // SDL_directsound_h_
