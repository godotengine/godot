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

#ifndef SDL_build_config_android_h_
#define SDL_build_config_android_h_
#define SDL_build_config_h_

#include <SDL3/SDL_platform_defines.h>

/**
 *  \file SDL_build_config_android.h
 *
 *  This is a configuration that can be used to build SDL for Android
 */

#include <stdarg.h>

#define HAVE_GCC_ATOMICS 1

#define HAVE_ALLOCA_H 1
#define HAVE_FLOAT_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_LIMITS_H 1
#define HAVE_MATH_H 1
#define HAVE_SIGNAL_H 1
#define HAVE_STDARG_H 1
#define HAVE_STDDEF_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_WCHAR_H 1

/* C library functions */
#define HAVE_LIBC 1
#define HAVE_DLOPEN 1
#define HAVE_MALLOC 1
#define HAVE_FDATASYNC 1
#define HAVE_GETENV 1
#define HAVE_GETHOSTNAME 1
#define HAVE_PUTENV 1
#define HAVE_SETENV 1
#define HAVE_UNSETENV 1
#define HAVE_ABS 1
#define HAVE_BCOPY 1
#define HAVE_MEMSET 1
#define HAVE_MEMCPY 1
#define HAVE_MEMMOVE 1
#define HAVE_MEMCMP 1
#define HAVE_STRLEN 1
#define HAVE_STRLCPY 1
#define HAVE_STRLCAT 1
#define HAVE_STRCHR 1
#define HAVE_STRRCHR 1
#define HAVE_STRSTR 1
#define HAVE_STRTOK_R 1
#define HAVE_STRTOL 1
#define HAVE_STRTOUL 1
#define HAVE_STRTOLL 1
#define HAVE_STRTOULL 1
#define HAVE_STRTOD 1
#define HAVE_ATOI 1
#define HAVE_ATOF 1
#define HAVE_STRCMP 1
#define HAVE_STRNCMP 1
#define HAVE_VSSCANF 1
#define HAVE_VSNPRINTF 1
#define HAVE_ACOS 1
#define HAVE_ACOSF 1
#define HAVE_ASIN 1
#define HAVE_ASINF 1
#define HAVE_ATAN 1
#define HAVE_ATANF 1
#define HAVE_ATAN2 1
#define HAVE_ATAN2F 1
#define HAVE_CEIL 1
#define HAVE_CEILF 1
#define HAVE_COPYSIGN 1
#define HAVE_COPYSIGNF 1
#define HAVE_COS 1
#define HAVE_COSF 1
#define HAVE_EXP 1
#define HAVE_EXPF 1
#define HAVE_FABS 1
#define HAVE_FABSF 1
#define HAVE_FLOOR 1
#define HAVE_FLOORF 1
#define HAVE_FMOD 1
#define HAVE_FMODF 1
#define HAVE_ISINF 1
#define HAVE_ISINF_FLOAT_MACRO 1
#define HAVE_ISNAN 1
#define HAVE_ISNAN_FLOAT_MACRO 1
#define HAVE_LOG 1
#define HAVE_LOGF 1
#define HAVE_LOG10 1
#define HAVE_LOG10F 1
#define HAVE_LROUND 1
#define HAVE_LROUNDF 1
#define HAVE_MODF 1
#define HAVE_MODFF 1
#define HAVE_POW 1
#define HAVE_POWF 1
#define HAVE_ROUND 1
#define HAVE_ROUNDF 1
#define HAVE_SCALBN 1
#define HAVE_SCALBNF 1
#define HAVE_SIN 1
#define HAVE_SINF 1
#define HAVE_SQRT 1
#define HAVE_SQRTF 1
#define HAVE_TAN 1
#define HAVE_TANF 1
#define HAVE_TRUNC 1
#define HAVE_TRUNCF 1
#define HAVE_SIGACTION 1
#define HAVE_SETJMP 1
#define HAVE_NANOSLEEP 1
#define HAVE_GMTIME_R 1
#define HAVE_LOCALTIME_R 1
#define HAVE_SYSCONF 1
#define HAVE_CLOCK_GETTIME 1

/* Enable various audio drivers */
#ifndef SDL_AUDIO_DISABLED
#define SDL_AUDIO_DRIVER_OPENSLES 1
#define SDL_AUDIO_DRIVER_AAUDIO 1
#endif /* SDL_AUDIO_DISABLED */

/* Enable various input drivers */
#ifndef SDL_JOYSTICK_DISABLED
#define SDL_JOYSTICK_ANDROID 1
#define SDL_JOYSTICK_HIDAPI 1
#define SDL_JOYSTICK_VIRTUAL 1
#endif /* SDL_JOYSTICK_DISABLED */
#ifndef SDL_HAPTIC_DISABLED
#define SDL_HAPTIC_ANDROID 1
#endif /* SDL_HAPTIC_DISABLED */

/* Enable the stub process support */
#define SDL_PROCESS_DUMMY 1

/* Enable sensor driver */
#ifndef SDL_SENSOR_DISABLED
#define SDL_SENSOR_ANDROID 1
#endif /* SDL_SENSOR_DISABLED */

/* Enable various shared object loading systems */
#define SDL_LOADSO_DLOPEN 1

/* Enable various threading systems */
#define SDL_THREAD_PTHREAD 1
#define SDL_THREAD_PTHREAD_RECURSIVE_MUTEX 1

/* Enable RTC system */
#define SDL_TIME_UNIX 1

/* Enable various timer systems */
#define SDL_TIMER_UNIX 1

/* Enable various video drivers */
#define SDL_VIDEO_DRIVER_ANDROID 1

/* Enable OpenGL ES */
#define SDL_VIDEO_OPENGL_ES 1
#define SDL_VIDEO_OPENGL_ES2 1
#define SDL_VIDEO_OPENGL_EGL 1
#define SDL_VIDEO_RENDER_OGL_ES2 1

/* Enable Vulkan support */
#if defined(__ARM_ARCH) && __ARM_ARCH < 7
/* Android does not support Vulkan in native code using the "armeabi" ABI. */
#else
#define SDL_VIDEO_VULKAN 1
#define SDL_VIDEO_RENDER_VULKAN 1
#define SDL_GPU_VULKAN 1
#define SDL_VIDEO_RENDER_GPU 1
#endif

/* Enable system power support */
#define SDL_POWER_ANDROID 1

/* Enable the filesystem driver */
#define SDL_FILESYSTEM_ANDROID 1
#define SDL_FSOPS_POSIX 1

/* Enable the camera driver */
#ifndef SDL_CAMERA_DISABLED
#define SDL_CAMERA_DRIVER_ANDROID 1
#endif /* SDL_CAMERA_DISABLED */

/* Enable nl_langinfo and high-res file times on version 26 and higher. */
#if __ANDROID_API__ >= 26
#define HAVE_NL_LANGINFO 1
#define HAVE_ST_MTIM 1
#endif

#endif /* SDL_build_config_android_h_ */
