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
 *  \file SDL_build_config.h
 *
 *  This is a set of defines to configure the SDL features
 */

#ifndef SDL_build_config_h_
#define SDL_build_config_h_

/* General platform specific identifiers */
#include <SDL3/SDL_platform_defines.h>

#define HAVE_GCC_ATOMICS 1
/* #undef HAVE_GCC_SYNC_LOCK_TEST_AND_SET */

/* #undef SDL_DISABLE_ALLOCA */

/* Useful headers */
#define HAVE_FLOAT_H 1
#define HAVE_STDARG_H 1
#define HAVE_STDDEF_H 1
#define HAVE_STDINT_H 1

/* Comment this if you want to build without any C library requirements */
#define HAVE_LIBC 1
#ifdef HAVE_LIBC

/* Useful headers */
#define HAVE_ALLOCA_H 1
#define HAVE_ICONV_H 1
#define HAVE_INTTYPES_H 1
#define HAVE_LIMITS_H 1
#define HAVE_MALLOC_H 1
#define HAVE_MATH_H 1
#define HAVE_MEMORY_H 1
#define HAVE_SIGNAL_H 1
#define HAVE_STDIO_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRINGS_H 1
#define HAVE_STRING_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_WCHAR_H 1
/* #undef HAVE_PTHREAD_NP_H */

/* C library functions */
#define HAVE_DLOPEN 1
#define HAVE_MALLOC 1
#define HAVE_FDATASYNC 1
#define HAVE_GETENV 1
#define HAVE_GETHOSTNAME 1
#define HAVE_SETENV 1
#define HAVE_PUTENV 1
#define HAVE_UNSETENV 1
#define HAVE_ABS 1
#define HAVE_BCOPY 1
#define HAVE_MEMSET 1
#define HAVE_MEMCPY 1
#define HAVE_MEMMOVE 1
#define HAVE_MEMCMP 1
#define HAVE_WCSLEN 1
#define HAVE_WCSNLEN 1
#define HAVE_WCSLCPY 1
#define HAVE_WCSLCAT 1
#define HAVE_WCSSTR 1
#define HAVE_WCSCMP 1
#define HAVE_WCSNCMP 1
#define HAVE_WCSTOL 1
#define HAVE_STRLEN 1
#define HAVE_STRNLEN 1
#define HAVE_STRLCPY 1
#define HAVE_STRLCAT 1
#define HAVE_STRPBRK 1
/* #undef HAVE__STRREV */
#define HAVE_INDEX 1
#define HAVE_RINDEX 1
#define HAVE_STRCHR 1
#define HAVE_STRRCHR 1
#define HAVE_STRSTR 1
/* #undef HAVE_STRNSTR */
#define HAVE_STRTOK_R 1
/* #undef HAVE_ITOA */
/* #undef HAVE__LTOA */
/* #undef HAVE__UITOA */
/* #undef HAVE__ULTOA */
#define HAVE_STRTOL 1
#define HAVE_STRTOUL 1
/* #undef HAVE__I64TOA */
/* #undef HAVE__UI64TOA */
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
/* #undef HAVE__COPYSIGN */
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
#define HAVE_ISINFF 1
#define HAVE_ISINF_FLOAT_MACRO 1
#define HAVE_ISNAN 1
#define HAVE_ISNANF 1
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
/* #undef HAVE__FSEEKI64 */
#define HAVE_FOPEN64 1
#define HAVE_FSEEKO 1
#define HAVE_FSEEKO64 1
#define HAVE_MEMFD_CREATE 1
#define HAVE_POSIX_FALLOCATE 1
#define HAVE_SIGACTION 1
#define HAVE_SIGTIMEDWAIT 1
#define HAVE_SA_SIGACTION 1
#define HAVE_ST_MTIM 1
#define HAVE_SETJMP 1
#define HAVE_NANOSLEEP 1
#define HAVE_GMTIME_R 1
#define HAVE_LOCALTIME_R 1
#define HAVE_NL_LANGINFO 1
#define HAVE_SYSCONF 1
/* #undef HAVE_SYSCTLBYNAME */
#define HAVE_CLOCK_GETTIME 1
#define HAVE_GETPAGESIZE 1
#define HAVE_ICONV 1
/* #undef SDL_USE_LIBICONV */
#define HAVE_PTHREAD_SETNAME_NP 1
/* #undef HAVE_PTHREAD_SET_NAME_NP */
#define HAVE_SEM_TIMEDWAIT 1
#define HAVE_GETAUXVAL 1
/* #undef HAVE_ELF_AUX_INFO */
#define HAVE_POLL 1
#define HAVE__EXIT 1
#define HAVE_GETRESUID 1
#define HAVE_GETRESGID 1

#endif /* HAVE_LIBC */

#define HAVE_DBUS_DBUS_H 1
#define HAVE_FCITX 1
/* #undef HAVE_IBUS_IBUS_H */
#define HAVE_INOTIFY_INIT1 1
#define HAVE_INOTIFY 1
/* #undef HAVE_LIBUSB */
#define HAVE_O_CLOEXEC 1

#define HAVE_LINUX_INPUT_H 1
#define HAVE_LIBUDEV_H 1
/* #undef HAVE_LIBDECOR_H */
/* #undef HAVE_LIBURING_H */

/* #undef HAVE_DDRAW_H */
/* #undef HAVE_DSOUND_H */
/* #undef HAVE_DINPUT_H */
/* #undef HAVE_XINPUT_H */
/* #undef HAVE_WINDOWS_GAMING_INPUT_H */
/* #undef HAVE_GAMEINPUT_H */
/* #undef HAVE_DXGI_H */
/* #undef HAVE_DXGI1_6_H */

/* #undef HAVE_MMDEVICEAPI_H */
/* #undef HAVE_TPCSHRD_H */
/* #undef HAVE_ROAPI_H */
/* #undef HAVE_SHELLSCALINGAPI_H */

/* #undef USE_POSIX_SPAWN */

/* SDL internal assertion support */
/* #undef SDL_DEFAULT_ASSERT_LEVEL_CONFIGURED */
#ifdef SDL_DEFAULT_ASSERT_LEVEL_CONFIGURED
#define SDL_DEFAULT_ASSERT_LEVEL 
#endif

/* Allow disabling of major subsystems */
#define SDL_AUDIO_DISABLED 1
/* #undef SDL_VIDEO_DISABLED */
/* #undef SDL_GPU_DISABLED */
/* #undef SDL_RENDER_DISABLED */
#define SDL_CAMERA_DISABLED 1
/* #undef SDL_JOYSTICK_DISABLED */
/* #undef SDL_HAPTIC_DISABLED */
/* #undef SDL_HIDAPI_DISABLED */
/* #undef SDL_POWER_DISABLED */
/* #undef SDL_SENSOR_DISABLED */
#define SDL_DIALOG_DISABLED 1
/* #undef SDL_THREADS_DISABLED */

/* Enable various audio drivers */
/* #undef SDL_AUDIO_DRIVER_ALSA */
/* #undef SDL_AUDIO_DRIVER_ALSA_DYNAMIC */
/* #undef SDL_AUDIO_DRIVER_OPENSLES */
/* #undef SDL_AUDIO_DRIVER_AAUDIO */
/* #undef SDL_AUDIO_DRIVER_COREAUDIO */
/* #undef SDL_AUDIO_DRIVER_DISK */
/* #undef SDL_AUDIO_DRIVER_DSOUND */
#define SDL_AUDIO_DRIVER_DUMMY 1
/* #undef SDL_AUDIO_DRIVER_EMSCRIPTEN */
/* #undef SDL_AUDIO_DRIVER_HAIKU */
/* #undef SDL_AUDIO_DRIVER_JACK */
/* #undef SDL_AUDIO_DRIVER_JACK_DYNAMIC */
/* #undef SDL_AUDIO_DRIVER_NETBSD */
/* #undef SDL_AUDIO_DRIVER_OSS */
/* #undef SDL_AUDIO_DRIVER_PIPEWIRE */
/* #undef SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC */
/* #undef SDL_AUDIO_DRIVER_PULSEAUDIO */
/* #undef SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC */
/* #undef SDL_AUDIO_DRIVER_SNDIO */
/* #undef SDL_AUDIO_DRIVER_SNDIO_DYNAMIC */
/* #undef SDL_AUDIO_DRIVER_WASAPI */
/* #undef SDL_AUDIO_DRIVER_VITA */
/* #undef SDL_AUDIO_DRIVER_PSP */
/* #undef SDL_AUDIO_DRIVER_PS2 */
/* #undef SDL_AUDIO_DRIVER_N3DS */
/* #undef SDL_AUDIO_DRIVER_QNX */

/* Enable various input drivers */
#define SDL_INPUT_LINUXEV 1
#define SDL_INPUT_LINUXKD 1
/* #undef SDL_INPUT_FBSDKBIO */
/* #undef SDL_INPUT_WSCONS */
/* #undef SDL_HAVE_MACHINE_JOYSTICK_H */
/* #undef SDL_JOYSTICK_ANDROID */
/* #undef SDL_JOYSTICK_DINPUT */
/* #undef SDL_JOYSTICK_DUMMY */
/* #undef SDL_JOYSTICK_EMSCRIPTEN */
/* #undef SDL_JOYSTICK_GAMEINPUT */
/* #undef SDL_JOYSTICK_HAIKU */
#define SDL_JOYSTICK_HIDAPI 1
/* #undef SDL_JOYSTICK_IOKIT */
#define SDL_JOYSTICK_LINUX 1
/* #undef SDL_JOYSTICK_MFI */
/* #undef SDL_JOYSTICK_N3DS */
/* #undef SDL_JOYSTICK_PS2 */
/* #undef SDL_JOYSTICK_PSP */
/* #undef SDL_JOYSTICK_RAWINPUT */
/* #undef SDL_JOYSTICK_USBHID */
#define SDL_JOYSTICK_VIRTUAL 1
/* #undef SDL_JOYSTICK_VITA */
/* #undef SDL_JOYSTICK_WGI */
/* #undef SDL_JOYSTICK_XINPUT */
/* #undef SDL_HAPTIC_DUMMY */
#define SDL_HAPTIC_LINUX 1
/* #undef SDL_HAPTIC_IOKIT */
/* #undef SDL_HAPTIC_DINPUT */
/* #undef SDL_HAPTIC_ANDROID */
/* #undef SDL_LIBUSB_DYNAMIC */
#define SDL_UDEV_DYNAMIC "libudev.so.1"

/* Enable various process implementations */
/* #undef SDL_PROCESS_DUMMY */
#define SDL_PROCESS_POSIX 1
/* #undef SDL_PROCESS_WINDOWS */

/* Enable various sensor drivers */
/* #undef SDL_SENSOR_ANDROID */
/* #undef SDL_SENSOR_COREMOTION */
/* #undef SDL_SENSOR_WINDOWS */
#define SDL_SENSOR_DUMMY 1
/* #undef SDL_SENSOR_VITA */
/* #undef SDL_SENSOR_N3DS */

/* Enable various shared object loading systems */
#define SDL_LOADSO_DLOPEN 1
/* #undef SDL_LOADSO_DUMMY */
/* #undef SDL_LOADSO_WINDOWS */

/* Enable various threading systems */
/* #undef SDL_THREAD_GENERIC_COND_SUFFIX */
/* #undef SDL_THREAD_GENERIC_RWLOCK_SUFFIX */
#define SDL_THREAD_PTHREAD 1
#define SDL_THREAD_PTHREAD_RECURSIVE_MUTEX 1
/* #undef SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP */
/* #undef SDL_THREAD_WINDOWS */
/* #undef SDL_THREAD_VITA */
/* #undef SDL_THREAD_PSP */
/* #undef SDL_THREAD_PS2 */
/* #undef SDL_THREAD_N3DS */

/* Enable various RTC systems */
#define SDL_TIME_UNIX 1
/* #undef SDL_TIME_WINDOWS */
/* #undef SDL_TIME_VITA */
/* #undef SDL_TIME_PSP */
/* #undef SDL_TIME_PS2 */
/* #undef SDL_TIME_N3DS */

/* Enable various timer systems */
/* #undef SDL_TIMER_HAIKU */
#define SDL_TIMER_UNIX 1
/* #undef SDL_TIMER_WINDOWS */
/* #undef SDL_TIMER_VITA */
/* #undef SDL_TIMER_PSP */
/* #undef SDL_TIMER_PS2 */
/* #undef SDL_TIMER_N3DS */

/* Enable various video drivers */
/* #undef SDL_VIDEO_DRIVER_ANDROID */
/* #undef SDL_VIDEO_DRIVER_COCOA */
#define SDL_VIDEO_DRIVER_DUMMY 1
/* #undef SDL_VIDEO_DRIVER_EMSCRIPTEN */
/* #undef SDL_VIDEO_DRIVER_HAIKU */
/* #undef SDL_VIDEO_DRIVER_KMSDRM */
/* #undef SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC */
/* #undef SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM */
/* #undef SDL_VIDEO_DRIVER_N3DS */
#define SDL_VIDEO_DRIVER_OFFSCREEN 1
/* #undef SDL_VIDEO_DRIVER_PS2 */
/* #undef SDL_VIDEO_DRIVER_PSP */
/* #undef SDL_VIDEO_DRIVER_RISCOS */
/* #undef SDL_VIDEO_DRIVER_ROCKCHIP */
/* #undef SDL_VIDEO_DRIVER_RPI */
/* #undef SDL_VIDEO_DRIVER_UIKIT */
/* #undef SDL_VIDEO_DRIVER_VITA */
/* #undef SDL_VIDEO_DRIVER_VIVANTE */
/* #undef SDL_VIDEO_DRIVER_VIVANTE_VDK */
/* #undef SDL_VIDEO_DRIVER_OPENVR */
/* #undef SDL_VIDEO_DRIVER_WAYLAND */
/* #undef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC */
/* #undef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR */
/* #undef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL */
/* #undef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR */
/* #undef SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON */
/* #undef SDL_VIDEO_DRIVER_WINDOWS */
#define SDL_VIDEO_DRIVER_X11 1
#define SDL_VIDEO_DRIVER_X11_DYNAMIC "libX11.so.6"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR "libXcursor.so.1"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT "libXext.so.6"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES "libXfixes.so.3"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2 "libXi.so.6"
#define SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR "libXrandr.so.2"
/* #undef SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS */
#define SDL_VIDEO_DRIVER_X11_HAS_XKBLOOKUPKEYSYM 1
#define SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS 1
#define SDL_VIDEO_DRIVER_X11_XCURSOR 1
#define SDL_VIDEO_DRIVER_X11_XDBE 1
#define SDL_VIDEO_DRIVER_X11_XFIXES 1
#define SDL_VIDEO_DRIVER_X11_XINPUT2 1
#define SDL_VIDEO_DRIVER_X11_XINPUT2_SUPPORTS_MULTITOUCH 1
#define SDL_VIDEO_DRIVER_X11_XRANDR 1
/* #undef SDL_VIDEO_DRIVER_X11_XSCRNSAVER */
#define SDL_VIDEO_DRIVER_X11_XSHAPE 1
#define SDL_VIDEO_DRIVER_X11_XSYNC 1
/* #undef SDL_VIDEO_DRIVER_QNX */

/* #undef SDL_VIDEO_RENDER_D3D */
/* #undef SDL_VIDEO_RENDER_D3D11 */
/* #undef SDL_VIDEO_RENDER_D3D12 */
#define SDL_VIDEO_RENDER_GPU 1
/* #undef SDL_VIDEO_RENDER_METAL */
#define SDL_VIDEO_RENDER_VULKAN 1
#define SDL_VIDEO_RENDER_OGL 1
/* #undef SDL_VIDEO_RENDER_OGL_ES2 */
/* #undef SDL_VIDEO_RENDER_PS2 */
/* #undef SDL_VIDEO_RENDER_PSP */
/* #undef SDL_VIDEO_RENDER_VITA_GXM */

/* Enable OpenGL support */
#define SDL_VIDEO_OPENGL 1
/* #undef SDL_VIDEO_OPENGL_ES */
/* #undef SDL_VIDEO_OPENGL_ES2 */
/* #undef SDL_VIDEO_OPENGL_CGL */
#define SDL_VIDEO_OPENGL_GLX 1
/* #undef SDL_VIDEO_OPENGL_WGL */
#define SDL_VIDEO_OPENGL_EGL 1

/* Enable Vulkan support */
#define SDL_VIDEO_VULKAN 1

/* Enable Metal support */
/* #undef SDL_VIDEO_METAL */

/* Enable GPU support */
/* #undef SDL_GPU_D3D11 */
/* #undef SDL_GPU_D3D12 */
#define SDL_GPU_VULKAN 1
/* #undef SDL_GPU_METAL */

/* Enable system power support */
/* #undef SDL_POWER_ANDROID */
#define SDL_POWER_LINUX 1
/* #undef SDL_POWER_WINDOWS */
/* #undef SDL_POWER_MACOSX */
/* #undef SDL_POWER_UIKIT */
/* #undef SDL_POWER_HAIKU */
/* #undef SDL_POWER_EMSCRIPTEN */
/* #undef SDL_POWER_HARDWIRED */
/* #undef SDL_POWER_VITA */
/* #undef SDL_POWER_PSP */
/* #undef SDL_POWER_N3DS */

/* Enable system filesystem support */
/* #undef SDL_FILESYSTEM_ANDROID */
/* #undef SDL_FILESYSTEM_HAIKU */
/* #undef SDL_FILESYSTEM_COCOA */
/* #undef SDL_FILESYSTEM_DUMMY */
/* #undef SDL_FILESYSTEM_RISCOS */
#define SDL_FILESYSTEM_UNIX 1
/* #undef SDL_FILESYSTEM_WINDOWS */
/* #undef SDL_FILESYSTEM_EMSCRIPTEN */
/* #undef SDL_FILESYSTEM_VITA */
/* #undef SDL_FILESYSTEM_PSP */
/* #undef SDL_FILESYSTEM_PS2 */
/* #undef SDL_FILESYSTEM_N3DS */

/* Enable system storage support */
#define SDL_STORAGE_STEAM 1

/* Enable system FSops support */
#define SDL_FSOPS_POSIX 1
/* #undef SDL_FSOPS_WINDOWS */
/* #undef SDL_FSOPS_DUMMY */

/* Enable camera subsystem */
#define SDL_CAMERA_DRIVER_DUMMY 1
/* !!! FIXME: for later cmakedefine SDL_CAMERA_DRIVER_DISK 1 */
/* #undef SDL_CAMERA_DRIVER_V4L2 */
/* #undef SDL_CAMERA_DRIVER_COREMEDIA */
/* #undef SDL_CAMERA_DRIVER_ANDROID */
/* #undef SDL_CAMERA_DRIVER_EMSCRIPTEN */
/* #undef SDL_CAMERA_DRIVER_MEDIAFOUNDATION */
/* #undef SDL_CAMERA_DRIVER_PIPEWIRE */
/* #undef SDL_CAMERA_DRIVER_PIPEWIRE_DYNAMIC */
/* #undef SDL_CAMERA_DRIVER_VITA */

/* Enable dialog subsystem */
#define SDL_DIALOG_DUMMY 1

/* Enable assembly routines */
/* #undef SDL_ALTIVEC_BLITTERS */

/* Whether SDL_DYNAMIC_API needs dlopen */
#define DYNAPI_NEEDS_DLOPEN 1

/* Enable ime support */
#define SDL_USE_IME 1
/* #undef SDL_DISABLE_WINDOWS_IME */
/* #undef SDL_GDK_TEXTINPUT */

/* Platform specific definitions */
/* #undef SDL_IPHONE_KEYBOARD */
/* #undef SDL_IPHONE_LAUNCHSCREEN */

/* #undef SDL_VIDEO_VITA_PIB */
/* #undef SDL_VIDEO_VITA_PVR */
/* #undef SDL_VIDEO_VITA_PVR_OGL */

/* Libdecor version info */
#define SDL_LIBDECOR_VERSION_MAJOR 
#define SDL_LIBDECOR_VERSION_MINOR 
#define SDL_LIBDECOR_VERSION_PATCH 

#if !defined(HAVE_STDINT_H) && !defined(_STDINT_H_)
/* Most everything except Visual Studio 2008 and earlier has stdint.h now */
#if defined(_MSC_VER) && (_MSC_VER < 1600)
typedef signed __int8 int8_t;
typedef unsigned __int8 uint8_t;
typedef signed __int16 int16_t;
typedef unsigned __int16 uint16_t;
typedef signed __int32 int32_t;
typedef unsigned __int32 uint32_t;
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
#ifndef _UINTPTR_T_DEFINED
#ifdef _WIN64
typedef unsigned __int64 uintptr_t;
#else
typedef unsigned int uintptr_t;
#endif
#endif
#endif /* Visual Studio 2008 */
#endif /* !_STDINT_H_ && !HAVE_STDINT_H */

/* Configure use of intrinsics */
/* #undef SDL_DISABLE_SSE */
/* #undef SDL_DISABLE_SSE2 */
/* #undef SDL_DISABLE_SSE3 */
/* #undef SDL_DISABLE_SSE4_1 */
/* #undef SDL_DISABLE_SSE4_2 */
/* #undef SDL_DISABLE_AVX */
/* #undef SDL_DISABLE_AVX2 */
/* #undef SDL_DISABLE_AVX512F */
/* #undef SDL_DISABLE_MMX */
#define SDL_DISABLE_LSX 1
#define SDL_DISABLE_LASX 1
#define SDL_DISABLE_NEON 1

#endif /* SDL_build_config_h_ */
