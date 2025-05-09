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

/**
 * Main include header for the SDL library, version 3.2.14
 *
 * It is almost always best to include just this one header instead of
 * picking out individual headers included here. There are exceptions to
 * this rule--SDL_main.h is special and not included here--but usually
 * letting SDL.h include the kitchen sink for you is the correct approach.
 */

#ifndef SDL_h_
#define SDL_h_

#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_assert.h>
#include <SDL3/SDL_asyncio.h>
#include <SDL3/SDL_atomic.h>
#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_bits.h>
#include <SDL3/SDL_blendmode.h>
#include <SDL3/SDL_camera.h>
#include <SDL3/SDL_clipboard.h>
#include <SDL3/SDL_cpuinfo.h>
#include <SDL3/SDL_dialog.h>
#include <SDL3/SDL_endian.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_filesystem.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_gpu.h>
#include <SDL3/SDL_guid.h>
#include <SDL3/SDL_haptic.h>
#include <SDL3/SDL_hidapi.h>
#include <SDL3/SDL_hints.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_iostream.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_keyboard.h>
#include <SDL3/SDL_keycode.h>
#include <SDL3/SDL_loadso.h>
#include <SDL3/SDL_locale.h>
#include <SDL3/SDL_log.h>
#include <SDL3/SDL_messagebox.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_misc.h>
#include <SDL3/SDL_mouse.h>
#include <SDL3/SDL_mutex.h>
#include <SDL3/SDL_pen.h>
#include <SDL3/SDL_pixels.h>
#include <SDL3/SDL_platform.h>
#include <SDL3/SDL_power.h>
#include <SDL3/SDL_process.h>
#include <SDL3/SDL_properties.h>
#include <SDL3/SDL_rect.h>
#include <SDL3/SDL_render.h>
#include <SDL3/SDL_scancode.h>
#include <SDL3/SDL_sensor.h>
#include <SDL3/SDL_storage.h>
#include <SDL3/SDL_surface.h>
#include <SDL3/SDL_system.h>
#include <SDL3/SDL_thread.h>
#include <SDL3/SDL_time.h>
#include <SDL3/SDL_timer.h>
#include <SDL3/SDL_tray.h>
#include <SDL3/SDL_touch.h>
#include <SDL3/SDL_version.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_oldnames.h>

#endif /* SDL_h_ */
