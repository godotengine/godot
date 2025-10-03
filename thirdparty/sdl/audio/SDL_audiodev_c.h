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

#ifndef SDL_audiodev_c_h_
#define SDL_audiodev_c_h_

#include "SDL_internal.h"
#include "SDL_sysaudio.h"

// Open the audio device for playback, and don't block if busy
//#define USE_BLOCKING_WRITES

#ifdef USE_BLOCKING_WRITES
#define OPEN_FLAGS_OUTPUT O_WRONLY
#define OPEN_FLAGS_INPUT  O_RDONLY
#else
#define OPEN_FLAGS_OUTPUT (O_WRONLY | O_NONBLOCK)
#define OPEN_FLAGS_INPUT  (O_RDONLY | O_NONBLOCK)
#endif

extern void SDL_EnumUnixAudioDevices(const bool classic, bool (*test)(int));

#endif // SDL_audiodev_c_h_
