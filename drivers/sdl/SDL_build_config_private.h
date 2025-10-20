/**************************************************************************/
/*  SDL_build_config_private.h                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#define SDL_build_config_h_

#include <SDL3/SDL_platform_defines.h>

#define HAVE_STDARG_H 1
#define HAVE_STDDEF_H 1

// Here we disable SDL subsystems that are not going to be used
#define SDL_CPUINFO_DISABLED 1
#define SDL_AUDIO_DISABLED 1
#define SDL_PROCESS_DUMMY 1
#define SDL_LOADSO_DUMMY 1
#define SDL_VIDEO_DISABLED 1
#define SDL_CAMERA_DISABLED 1
#define SDL_DIALOG_DISABLED 1
#define SDL_FILESYSTEM_DUMMY 1
#define SDL_FSOPS_DUMMY 1
#define SDL_GPU_DISABLED 1
#define SDL_RENDER_DISABLED 1
#define SDL_POWER_DISABLED 1
#define SDL_LEAN_AND_MEAN 1

// Windows defines
#if defined(SDL_PLATFORM_WINDOWS)

#define SDL_PLATFORM_PRIVATE_NAME "Windows"
#define HAVE_LIBC 1
#define HAVE_DINPUT_H 1
#define HAVE_XINPUT_H 1
#if defined(_WIN32_MAXVER) && _WIN32_MAXVER >= 0x0A00 /* Windows 10 SDK */
#define HAVE_WINDOWS_GAMING_INPUT_H 1
#define SDL_JOYSTICK_WGI 1
#endif
#define SDL_JOYSTICK_DINPUT 1
#define SDL_JOYSTICK_HIDAPI 1
#define SDL_JOYSTICK_RAWINPUT 1
#define SDL_JOYSTICK_XINPUT 1
#define SDL_HAPTIC_DINPUT 1
#define SDL_THREAD_GENERIC_COND_SUFFIX 1
#define SDL_THREAD_GENERIC_RWLOCK_SUFFIX 1
#define SDL_THREAD_WINDOWS 1
#define SDL_TIMER_WINDOWS 1
#define SDL_SENSOR_WINDOWS 1

// Linux defines
#elif defined(SDL_PLATFORM_LINUX)

#define SDL_PLATFORM_PRIVATE_NAME "Linux"
#define SDL_PLATFORM_UNIX 1

#define HAVE_STDIO_H 1
#define HAVE_LIBC 1
#define HAVE_LINUX_INPUT_H 1
#define HAVE_POLL 1

#ifdef __linux__
#define HAVE_INOTIFY 1
#define HAVE_INOTIFY_INIT1 1
// Don't add these defines, for some reason they mess with C#'s ability
// to use environment variables (see GH-109024)
//#define HAVE_GETENV 1
//#define HAVE_SETENV 1
//#define HAVE_UNSETENV 1
#endif

#ifdef DBUS_ENABLED
#define HAVE_DBUS_DBUS_H 1
#define SDL_USE_LIBDBUS 1
// SOWRAP_ENABLED is handled in thirdparty/sdl/core/linux/SDL_dbus.c
#endif

#ifdef UDEV_ENABLED
#define HAVE_LIBUDEV_H 1
#define SDL_USE_LIBUDEV
#ifdef SOWRAP_ENABLED
#define SDL_UDEV_DYNAMIC "libudev.so.1"
#endif
#endif

#define SDL_LOADSO_DLOPEN 1
#define SDL_HAPTIC_LINUX 1
#define SDL_TIMER_UNIX 1
#define SDL_JOYSTICK_LINUX 1
#define SDL_JOYSTICK_HIDAPI 1
#define SDL_INPUT_LINUXEV 1
#define SDL_THREAD_PTHREAD 1

// MacOS defines
#elif defined(SDL_PLATFORM_MACOS)

#define SDL_PLATFORM_PRIVATE_NAME "macOS"
#define SDL_PLATFORM_UNIX 1
#define HAVE_STDIO_H 1
#define HAVE_LIBC 1
#define SDL_HAPTIC_IOKIT 1
#define SDL_JOYSTICK_IOKIT 1
#define SDL_JOYSTICK_MFI 1
#define SDL_JOYSTICK_HIDAPI 1
#define SDL_TIMER_UNIX 1
#define SDL_THREAD_PTHREAD 1
#define SDL_THREAD_PTHREAD_RECURSIVE_MUTEX 1

// Other platforms are not supported (for now)
#else
#error "No SDL build config was found for this platform. Setup one before compiling the engine."
#endif

#if !defined(HAVE_STDINT_H) && !defined(_STDINT_H_)
#define HAVE_STDINT_H 1
#endif /* !_STDINT_H_ && !HAVE_STDINT_H */

#ifdef __GNUC__
#define HAVE_GCC_SYNC_LOCK_TEST_AND_SET 1
#endif
