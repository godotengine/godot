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

#ifndef SDL_build_config_wingdk_h_
#define SDL_build_config_wingdk_h_
#define SDL_build_config_h_

#include <SDL3/SDL_platform_defines.h>

/* Windows GDK does not need Windows SDK version checks because it requires
 * a recent version of the Windows 10 SDK. */

#ifdef __clang__
# define HAVE_GCC_ATOMICS 1
#endif

#define HAVE_DDRAW_H 1
#define HAVE_DINPUT_H 1
#define HAVE_DSOUND_H 1
/* No SDK version checks needed for these because the SDK has to be new. */
#define HAVE_DXGI_H 1
#define HAVE_DXGI1_6_H 1
#define HAVE_XINPUT_H 1
#define HAVE_WINDOWS_GAMING_INPUT_H 1
#define HAVE_D3D11_H 1
#define HAVE_ROAPI_H 1
#define HAVE_SHELLSCALINGAPI_H 1
#define HAVE_MMDEVICEAPI_H 1
#define HAVE_AUDIOCLIENT_H 1
#define HAVE_TPCSHRD_H 1
#define HAVE_SENSORSAPI_H 1
#define HAVE_GAMEINPUT_H 1
#if (defined(_M_IX86) || defined(_M_X64) || defined(_M_AMD64)) && (defined(_MSC_VER) && _MSC_VER >= 1600)
#elif defined(__has_include) && (defined(__i386__) || defined(__x86_64))
# if !__has_include(<immintrin.h>)
#  define SDL_DISABLE_AVX 1
# endif
#else
# define SDL_DISABLE_AVX 1
#endif

/* Useful headers */
#define HAVE_FLOAT_H 1
#define HAVE_LIMITS_H 1
#define HAVE_MATH_H 1
#define HAVE_SIGNAL_H 1
#define HAVE_STDARG_H 1
#define HAVE_STDDEF_H 1
#define HAVE_STDINT_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_WCHAR_H 1

/* C library functions */
#define HAVE_LIBC 1
#define HAVE_MALLOC 1
#define HAVE_ABS 1
#define HAVE_MEMSET 1
#define HAVE_MEMCPY 1
#define HAVE_MEMMOVE 1
#define HAVE_MEMCMP 1
#define HAVE_STRLEN 1
#define HAVE__STRREV 1
#define HAVE_STRCHR 1
#define HAVE_STRRCHR 1
#define HAVE_STRSTR 1
/* #undef HAVE_STRTOK_R */
/* These functions have security warnings, so we won't use them */
/* #undef HAVE__LTOA */
/* #undef HAVE__ULTOA */
#define HAVE_STRTOL 1
#define HAVE_STRTOUL 1
#define HAVE_STRTOD 1
#define HAVE_ATOI 1
#define HAVE_ATOF 1
#define HAVE_STRCMP 1
#define HAVE_STRNCMP 1
#define HAVE_ACOS 1
#define HAVE_ASIN 1
#define HAVE_ATAN 1
#define HAVE_ATAN2 1
#define HAVE_CEIL 1
#define HAVE_COS 1
#define HAVE_EXP 1
#define HAVE_FABS 1
#define HAVE_FLOOR 1
#define HAVE_FMOD 1
#define HAVE_ISINF 1
#define HAVE_ISINF_FLOAT_MACRO 1
#define HAVE_ISNAN 1
#define HAVE_ISNAN_FLOAT_MACRO 1
#define HAVE_LOG 1
#define HAVE_LOG10 1
#define HAVE_POW 1
#define HAVE_SIN 1
#define HAVE_SQRT 1
#define HAVE_TAN 1
#define HAVE_ACOSF 1
#define HAVE_ASINF 1
#define HAVE_ATANF 1
#define HAVE_ATAN2F 1
#define HAVE_CEILF 1
#define HAVE__COPYSIGN 1
#define HAVE_COSF 1
#define HAVE_EXPF 1
#define HAVE_FABSF 1
#define HAVE_FLOORF 1
#define HAVE_FMODF 1
#define HAVE_LOGF 1
#define HAVE_LOG10F 1
#define HAVE_POWF 1
#define HAVE_SINF 1
#define HAVE_SQRTF 1
#define HAVE_TANF 1
#ifdef _MSC_VER
/* These functions were added with the VC++ 2013 C runtime library */
#define HAVE_STRTOLL 1
#define HAVE_STRTOULL 1
#define HAVE_VSSCANF 1
#define HAVE_LROUND 1
#define HAVE_LROUNDF 1
#define HAVE_ROUND 1
#define HAVE_ROUNDF 1
#define HAVE_SCALBN 1
#define HAVE_SCALBNF 1
#define HAVE_TRUNC 1
#define HAVE_TRUNCF 1
#define HAVE__FSEEKI64 1
#endif    /* _MSC_VER */

/* Enable various audio drivers */
#if defined(HAVE_MMDEVICEAPI_H) && defined(HAVE_AUDIOCLIENT_H)
#define SDL_AUDIO_DRIVER_WASAPI 1
#endif
#define SDL_AUDIO_DRIVER_DSOUND 1
#define SDL_AUDIO_DRIVER_DISK 1
#define SDL_AUDIO_DRIVER_DUMMY 1

/* Enable various input drivers */
#define SDL_JOYSTICK_DINPUT 1
#define SDL_JOYSTICK_GAMEINPUT 1
#define SDL_JOYSTICK_HIDAPI 1
#define SDL_JOYSTICK_RAWINPUT 1
#define SDL_JOYSTICK_VIRTUAL 1
#define SDL_JOYSTICK_WGI 1
#define SDL_JOYSTICK_XINPUT 1
#define SDL_HAPTIC_DINPUT 1

/* Enable various process implementations */
#define SDL_PROCESS_WINDOWS 1

/* Enable the sensor driver */
#ifdef HAVE_SENSORSAPI_H
#define SDL_SENSOR_WINDOWS 1
#else
#define SDL_SENSOR_DUMMY 1
#endif

/* Enable various shared object loading systems */
#define SDL_LOADSO_WINDOWS 1

/* Enable various threading systems */
#define SDL_THREAD_GENERIC_COND_SUFFIX 1
#define SDL_THREAD_GENERIC_RWLOCK_SUFFIX 1
#define SDL_THREAD_WINDOWS 1

/* Enable various time systems */
#define SDL_TIME_WINDOWS 1

/* Enable various timer systems */
#define SDL_TIMER_WINDOWS 1

/* Enable various video drivers */
#define SDL_VIDEO_DRIVER_DUMMY 1
#define SDL_VIDEO_DRIVER_WINDOWS 1
#define SDL_VIDEO_RENDER_D3D 1
#ifdef HAVE_D3D11_H
#define SDL_VIDEO_RENDER_D3D11 1
#endif
#define SDL_VIDEO_RENDER_D3D12 1

/* Enable OpenGL support */
#define SDL_VIDEO_OPENGL 1
#define SDL_VIDEO_OPENGL_WGL 1
#define SDL_VIDEO_RENDER_OGL 1
#define SDL_VIDEO_RENDER_OGL_ES2 1
#define SDL_VIDEO_OPENGL_ES2 1
#define SDL_VIDEO_OPENGL_EGL 1

/* Enable Vulkan support */
#define SDL_VIDEO_VULKAN 1
#define SDL_VIDEO_RENDER_VULKAN 1

/* Enable GPU support */
#ifdef HAVE_D3D11_H
#define SDL_GPU_D3D11 1
#endif
#define SDL_GPU_D3D12 1
#define SDL_GPU_VULKAN 1
#define SDL_VIDEO_RENDER_GPU 1

/* Enable system power support */
#define SDL_POWER_WINDOWS 1

/* Enable filesystem support */
#define SDL_FILESYSTEM_WINDOWS 1
#define SDL_FSOPS_WINDOWS 1

/* Enable the camera driver (src/camera/dummy/\*.c) */  /* !!! FIXME */
#define SDL_CAMERA_DRIVER_DUMMY 1

/* Use the (inferior) GDK text input method for GDK platforms */
/*#define SDL_GDK_TEXTINPUT 1*/

#endif /* SDL_build_config_wingdk_h_ */
