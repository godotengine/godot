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

#cmakedefine HAVE_GCC_ATOMICS 1
#cmakedefine HAVE_GCC_SYNC_LOCK_TEST_AND_SET 1

#cmakedefine SDL_DISABLE_ALLOCA 1

/* Useful headers */
#cmakedefine HAVE_FLOAT_H 1
#cmakedefine HAVE_STDARG_H 1
#cmakedefine HAVE_STDDEF_H 1
#cmakedefine HAVE_STDINT_H 1

/* Comment this if you want to build without any C library requirements */
#cmakedefine HAVE_LIBC 1
#ifdef HAVE_LIBC

/* Useful headers */
#cmakedefine HAVE_ALLOCA_H 1
#cmakedefine HAVE_ICONV_H 1
#cmakedefine HAVE_INTTYPES_H 1
#cmakedefine HAVE_LIMITS_H 1
#cmakedefine HAVE_MALLOC_H 1
#cmakedefine HAVE_MATH_H 1
#cmakedefine HAVE_MEMORY_H 1
#cmakedefine HAVE_SIGNAL_H 1
#cmakedefine HAVE_STDIO_H 1
#cmakedefine HAVE_STDLIB_H 1
#cmakedefine HAVE_STRINGS_H 1
#cmakedefine HAVE_STRING_H 1
#cmakedefine HAVE_SYS_TYPES_H 1
#cmakedefine HAVE_WCHAR_H 1
#cmakedefine HAVE_PTHREAD_NP_H 1

/* C library functions */
#cmakedefine HAVE_DLOPEN 1
#cmakedefine HAVE_MALLOC 1
#cmakedefine HAVE_FDATASYNC 1
#cmakedefine HAVE_GETENV 1
#cmakedefine HAVE_GETHOSTNAME 1
#cmakedefine HAVE_SETENV 1
#cmakedefine HAVE_PUTENV 1
#cmakedefine HAVE_UNSETENV 1
#cmakedefine HAVE_ABS 1
#cmakedefine HAVE_BCOPY 1
#cmakedefine HAVE_MEMSET 1
#cmakedefine HAVE_MEMCPY 1
#cmakedefine HAVE_MEMMOVE 1
#cmakedefine HAVE_MEMCMP 1
#cmakedefine HAVE_WCSLEN 1
#cmakedefine HAVE_WCSNLEN 1
#cmakedefine HAVE_WCSLCPY 1
#cmakedefine HAVE_WCSLCAT 1
#cmakedefine HAVE_WCSSTR 1
#cmakedefine HAVE_WCSCMP 1
#cmakedefine HAVE_WCSNCMP 1
#cmakedefine HAVE_WCSTOL 1
#cmakedefine HAVE_STRLEN 1
#cmakedefine HAVE_STRNLEN 1
#cmakedefine HAVE_STRLCPY 1
#cmakedefine HAVE_STRLCAT 1
#cmakedefine HAVE_STRPBRK 1
#cmakedefine HAVE__STRREV 1
#cmakedefine HAVE_INDEX 1
#cmakedefine HAVE_RINDEX 1
#cmakedefine HAVE_STRCHR 1
#cmakedefine HAVE_STRRCHR 1
#cmakedefine HAVE_STRSTR 1
#cmakedefine HAVE_STRNSTR 1
#cmakedefine HAVE_STRTOK_R 1
#cmakedefine HAVE_ITOA 1
#cmakedefine HAVE__LTOA 1
#cmakedefine HAVE__UITOA 1
#cmakedefine HAVE__ULTOA 1
#cmakedefine HAVE_STRTOL 1
#cmakedefine HAVE_STRTOUL 1
#cmakedefine HAVE__I64TOA 1
#cmakedefine HAVE__UI64TOA 1
#cmakedefine HAVE_STRTOLL 1
#cmakedefine HAVE_STRTOULL 1
#cmakedefine HAVE_STRTOD 1
#cmakedefine HAVE_ATOI 1
#cmakedefine HAVE_ATOF 1
#cmakedefine HAVE_STRCMP 1
#cmakedefine HAVE_STRNCMP 1
#cmakedefine HAVE_VSSCANF 1
#cmakedefine HAVE_VSNPRINTF 1
#cmakedefine HAVE_ACOS 1
#cmakedefine HAVE_ACOSF 1
#cmakedefine HAVE_ASIN 1
#cmakedefine HAVE_ASINF 1
#cmakedefine HAVE_ATAN 1
#cmakedefine HAVE_ATANF 1
#cmakedefine HAVE_ATAN2 1
#cmakedefine HAVE_ATAN2F 1
#cmakedefine HAVE_CEIL 1
#cmakedefine HAVE_CEILF 1
#cmakedefine HAVE_COPYSIGN 1
#cmakedefine HAVE_COPYSIGNF 1
#cmakedefine HAVE__COPYSIGN 1
#cmakedefine HAVE_COS 1
#cmakedefine HAVE_COSF 1
#cmakedefine HAVE_EXP 1
#cmakedefine HAVE_EXPF 1
#cmakedefine HAVE_FABS 1
#cmakedefine HAVE_FABSF 1
#cmakedefine HAVE_FLOOR 1
#cmakedefine HAVE_FLOORF 1
#cmakedefine HAVE_FMOD 1
#cmakedefine HAVE_FMODF 1
#cmakedefine HAVE_ISINF 1
#cmakedefine HAVE_ISINFF 1
#cmakedefine HAVE_ISINF_FLOAT_MACRO 1
#cmakedefine HAVE_ISNAN 1
#cmakedefine HAVE_ISNANF 1
#cmakedefine HAVE_ISNAN_FLOAT_MACRO 1
#cmakedefine HAVE_LOG 1
#cmakedefine HAVE_LOGF 1
#cmakedefine HAVE_LOG10 1
#cmakedefine HAVE_LOG10F 1
#cmakedefine HAVE_LROUND 1
#cmakedefine HAVE_LROUNDF 1
#cmakedefine HAVE_MODF 1
#cmakedefine HAVE_MODFF 1
#cmakedefine HAVE_POW 1
#cmakedefine HAVE_POWF 1
#cmakedefine HAVE_ROUND 1
#cmakedefine HAVE_ROUNDF 1
#cmakedefine HAVE_SCALBN 1
#cmakedefine HAVE_SCALBNF 1
#cmakedefine HAVE_SIN 1
#cmakedefine HAVE_SINF 1
#cmakedefine HAVE_SQRT 1
#cmakedefine HAVE_SQRTF 1
#cmakedefine HAVE_TAN 1
#cmakedefine HAVE_TANF 1
#cmakedefine HAVE_TRUNC 1
#cmakedefine HAVE_TRUNCF 1
#cmakedefine HAVE__FSEEKI64 1
#cmakedefine HAVE_FOPEN64 1
#cmakedefine HAVE_FSEEKO 1
#cmakedefine HAVE_FSEEKO64 1
#cmakedefine HAVE_MEMFD_CREATE 1
#cmakedefine HAVE_POSIX_FALLOCATE 1
#cmakedefine HAVE_SIGACTION 1
#cmakedefine HAVE_SA_SIGACTION 1
#cmakedefine HAVE_ST_MTIM 1
#cmakedefine HAVE_SETJMP 1
#cmakedefine HAVE_NANOSLEEP 1
#cmakedefine HAVE_GMTIME_R 1
#cmakedefine HAVE_LOCALTIME_R 1
#cmakedefine HAVE_NL_LANGINFO 1
#cmakedefine HAVE_SYSCONF 1
#cmakedefine HAVE_SYSCTLBYNAME 1
#cmakedefine HAVE_CLOCK_GETTIME 1
#cmakedefine HAVE_GETPAGESIZE 1
#cmakedefine HAVE_ICONV 1
#cmakedefine SDL_USE_LIBICONV 1
#cmakedefine HAVE_PTHREAD_SETNAME_NP 1
#cmakedefine HAVE_PTHREAD_SET_NAME_NP 1
#cmakedefine HAVE_SEM_TIMEDWAIT 1
#cmakedefine HAVE_GETAUXVAL 1
#cmakedefine HAVE_ELF_AUX_INFO 1
#cmakedefine HAVE_POLL 1
#cmakedefine HAVE__EXIT 1

#endif /* HAVE_LIBC */

#cmakedefine HAVE_DBUS_DBUS_H 1
#cmakedefine HAVE_FCITX 1
#cmakedefine HAVE_IBUS_IBUS_H 1
#cmakedefine HAVE_INOTIFY_INIT1 1
#cmakedefine HAVE_INOTIFY 1
#cmakedefine HAVE_LIBUSB 1
#cmakedefine HAVE_O_CLOEXEC 1

#cmakedefine HAVE_LINUX_INPUT_H 1
#cmakedefine HAVE_LIBUDEV_H 1
#cmakedefine HAVE_LIBDECOR_H 1
#cmakedefine HAVE_LIBURING_H 1

#cmakedefine HAVE_DDRAW_H 1
#cmakedefine HAVE_DSOUND_H 1
#cmakedefine HAVE_DINPUT_H 1
#cmakedefine HAVE_XINPUT_H 1
#cmakedefine HAVE_WINDOWS_GAMING_INPUT_H 1
#cmakedefine HAVE_GAMEINPUT_H 1
#cmakedefine HAVE_DXGI_H 1
#cmakedefine HAVE_DXGI1_6_H 1

#cmakedefine HAVE_MMDEVICEAPI_H 1
#cmakedefine HAVE_TPCSHRD_H 1
#cmakedefine HAVE_ROAPI_H 1
#cmakedefine HAVE_SHELLSCALINGAPI_H 1

#cmakedefine USE_POSIX_SPAWN 1

/* SDL internal assertion support */
#cmakedefine SDL_DEFAULT_ASSERT_LEVEL_CONFIGURED 1
#ifdef SDL_DEFAULT_ASSERT_LEVEL_CONFIGURED
#define SDL_DEFAULT_ASSERT_LEVEL @SDL_DEFAULT_ASSERT_LEVEL@
#endif

/* Allow disabling of major subsystems */
#cmakedefine SDL_AUDIO_DISABLED 1
#cmakedefine SDL_VIDEO_DISABLED 1
#cmakedefine SDL_GPU_DISABLED 1
#cmakedefine SDL_RENDER_DISABLED 1
#cmakedefine SDL_CAMERA_DISABLED 1
#cmakedefine SDL_JOYSTICK_DISABLED 1
#cmakedefine SDL_HAPTIC_DISABLED 1
#cmakedefine SDL_HIDAPI_DISABLED 1
#cmakedefine SDL_POWER_DISABLED 1
#cmakedefine SDL_SENSOR_DISABLED 1
#cmakedefine SDL_DIALOG_DISABLED 1
#cmakedefine SDL_THREADS_DISABLED 1

/* Enable various audio drivers */
#cmakedefine SDL_AUDIO_DRIVER_ALSA 1
#cmakedefine SDL_AUDIO_DRIVER_ALSA_DYNAMIC @SDL_AUDIO_DRIVER_ALSA_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_OPENSLES 1
#cmakedefine SDL_AUDIO_DRIVER_AAUDIO 1
#cmakedefine SDL_AUDIO_DRIVER_COREAUDIO 1
#cmakedefine SDL_AUDIO_DRIVER_DISK 1
#cmakedefine SDL_AUDIO_DRIVER_DSOUND 1
#cmakedefine SDL_AUDIO_DRIVER_DUMMY 1
#cmakedefine SDL_AUDIO_DRIVER_EMSCRIPTEN 1
#cmakedefine SDL_AUDIO_DRIVER_HAIKU 1
#cmakedefine SDL_AUDIO_DRIVER_JACK 1
#cmakedefine SDL_AUDIO_DRIVER_JACK_DYNAMIC @SDL_AUDIO_DRIVER_JACK_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_NETBSD 1
#cmakedefine SDL_AUDIO_DRIVER_OSS 1
#cmakedefine SDL_AUDIO_DRIVER_PIPEWIRE 1
#cmakedefine SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC @SDL_AUDIO_DRIVER_PIPEWIRE_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_PULSEAUDIO 1
#cmakedefine SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC @SDL_AUDIO_DRIVER_PULSEAUDIO_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_SNDIO 1
#cmakedefine SDL_AUDIO_DRIVER_SNDIO_DYNAMIC @SDL_AUDIO_DRIVER_SNDIO_DYNAMIC@
#cmakedefine SDL_AUDIO_DRIVER_WASAPI 1
#cmakedefine SDL_AUDIO_DRIVER_VITA 1
#cmakedefine SDL_AUDIO_DRIVER_PSP 1
#cmakedefine SDL_AUDIO_DRIVER_PS2 1
#cmakedefine SDL_AUDIO_DRIVER_N3DS 1
#cmakedefine SDL_AUDIO_DRIVER_QNX 1

/* Enable various input drivers */
#cmakedefine SDL_INPUT_LINUXEV 1
#cmakedefine SDL_INPUT_LINUXKD 1
#cmakedefine SDL_INPUT_FBSDKBIO 1
#cmakedefine SDL_INPUT_WSCONS 1
#cmakedefine SDL_HAVE_MACHINE_JOYSTICK_H 1
#cmakedefine SDL_JOYSTICK_ANDROID 1
#cmakedefine SDL_JOYSTICK_DINPUT 1
#cmakedefine SDL_JOYSTICK_DUMMY 1
#cmakedefine SDL_JOYSTICK_EMSCRIPTEN 1
#cmakedefine SDL_JOYSTICK_GAMEINPUT 1
#cmakedefine SDL_JOYSTICK_HAIKU 1
#cmakedefine SDL_JOYSTICK_HIDAPI 1
#cmakedefine SDL_JOYSTICK_IOKIT 1
#cmakedefine SDL_JOYSTICK_LINUX 1
#cmakedefine SDL_JOYSTICK_MFI 1
#cmakedefine SDL_JOYSTICK_N3DS 1
#cmakedefine SDL_JOYSTICK_PS2 1
#cmakedefine SDL_JOYSTICK_PSP 1
#cmakedefine SDL_JOYSTICK_RAWINPUT 1
#cmakedefine SDL_JOYSTICK_USBHID 1
#cmakedefine SDL_JOYSTICK_VIRTUAL 1
#cmakedefine SDL_JOYSTICK_VITA 1
#cmakedefine SDL_JOYSTICK_WGI 1
#cmakedefine SDL_JOYSTICK_XINPUT 1
#cmakedefine SDL_HAPTIC_DUMMY 1
#cmakedefine SDL_HAPTIC_LINUX 1
#cmakedefine SDL_HAPTIC_IOKIT 1
#cmakedefine SDL_HAPTIC_DINPUT 1
#cmakedefine SDL_HAPTIC_ANDROID 1
#cmakedefine SDL_LIBUSB_DYNAMIC @SDL_LIBUSB_DYNAMIC@
#cmakedefine SDL_UDEV_DYNAMIC @SDL_UDEV_DYNAMIC@

/* Enable various process implementations */
#cmakedefine SDL_PROCESS_DUMMY 1
#cmakedefine SDL_PROCESS_POSIX 1
#cmakedefine SDL_PROCESS_WINDOWS 1

/* Enable various sensor drivers */
#cmakedefine SDL_SENSOR_ANDROID 1
#cmakedefine SDL_SENSOR_COREMOTION 1
#cmakedefine SDL_SENSOR_WINDOWS 1
#cmakedefine SDL_SENSOR_DUMMY 1
#cmakedefine SDL_SENSOR_VITA 1
#cmakedefine SDL_SENSOR_N3DS 1

/* Enable various shared object loading systems */
#cmakedefine SDL_LOADSO_DLOPEN 1
#cmakedefine SDL_LOADSO_DUMMY 1
#cmakedefine SDL_LOADSO_WINDOWS 1

/* Enable various threading systems */
#cmakedefine SDL_THREAD_GENERIC_COND_SUFFIX 1
#cmakedefine SDL_THREAD_GENERIC_RWLOCK_SUFFIX 1
#cmakedefine SDL_THREAD_PTHREAD 1
#cmakedefine SDL_THREAD_PTHREAD_RECURSIVE_MUTEX 1
#cmakedefine SDL_THREAD_PTHREAD_RECURSIVE_MUTEX_NP 1
#cmakedefine SDL_THREAD_WINDOWS 1
#cmakedefine SDL_THREAD_VITA 1
#cmakedefine SDL_THREAD_PSP 1
#cmakedefine SDL_THREAD_PS2 1
#cmakedefine SDL_THREAD_N3DS 1

/* Enable various RTC systems */
#cmakedefine SDL_TIME_UNIX 1
#cmakedefine SDL_TIME_WINDOWS 1
#cmakedefine SDL_TIME_VITA 1
#cmakedefine SDL_TIME_PSP 1
#cmakedefine SDL_TIME_PS2 1
#cmakedefine SDL_TIME_N3DS 1

/* Enable various timer systems */
#cmakedefine SDL_TIMER_HAIKU 1
#cmakedefine SDL_TIMER_UNIX 1
#cmakedefine SDL_TIMER_WINDOWS 1
#cmakedefine SDL_TIMER_VITA 1
#cmakedefine SDL_TIMER_PSP 1
#cmakedefine SDL_TIMER_PS2 1
#cmakedefine SDL_TIMER_N3DS 1

/* Enable various video drivers */
#cmakedefine SDL_VIDEO_DRIVER_ANDROID 1
#cmakedefine SDL_VIDEO_DRIVER_COCOA 1
#cmakedefine SDL_VIDEO_DRIVER_DUMMY 1
#cmakedefine SDL_VIDEO_DRIVER_EMSCRIPTEN 1
#cmakedefine SDL_VIDEO_DRIVER_HAIKU 1
#cmakedefine SDL_VIDEO_DRIVER_KMSDRM 1
#cmakedefine SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC @SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM @SDL_VIDEO_DRIVER_KMSDRM_DYNAMIC_GBM@
#cmakedefine SDL_VIDEO_DRIVER_N3DS 1
#cmakedefine SDL_VIDEO_DRIVER_OFFSCREEN 1
#cmakedefine SDL_VIDEO_DRIVER_PS2 1
#cmakedefine SDL_VIDEO_DRIVER_PSP 1
#cmakedefine SDL_VIDEO_DRIVER_RISCOS 1
#cmakedefine SDL_VIDEO_DRIVER_ROCKCHIP 1
#cmakedefine SDL_VIDEO_DRIVER_RPI 1
#cmakedefine SDL_VIDEO_DRIVER_UIKIT 1
#cmakedefine SDL_VIDEO_DRIVER_VITA 1
#cmakedefine SDL_VIDEO_DRIVER_VIVANTE 1
#cmakedefine SDL_VIDEO_DRIVER_VIVANTE_VDK 1
#cmakedefine SDL_VIDEO_DRIVER_OPENVR 1
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND 1
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_CURSOR@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_EGL@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_LIBDECOR@
#cmakedefine SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON @SDL_VIDEO_DRIVER_WAYLAND_DYNAMIC_XKBCOMMON@
#cmakedefine SDL_VIDEO_DRIVER_WINDOWS 1
#cmakedefine SDL_VIDEO_DRIVER_X11 1
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC @SDL_VIDEO_DRIVER_X11_DYNAMIC@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR @SDL_VIDEO_DRIVER_X11_DYNAMIC_XCURSOR@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT @SDL_VIDEO_DRIVER_X11_DYNAMIC_XEXT@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES @SDL_VIDEO_DRIVER_X11_DYNAMIC_XFIXES@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2 @SDL_VIDEO_DRIVER_X11_DYNAMIC_XINPUT2@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR @SDL_VIDEO_DRIVER_X11_DYNAMIC_XRANDR@
#cmakedefine SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS @SDL_VIDEO_DRIVER_X11_DYNAMIC_XSS@
#cmakedefine SDL_VIDEO_DRIVER_X11_HAS_XKBLOOKUPKEYSYM 1
#cmakedefine SDL_VIDEO_DRIVER_X11_SUPPORTS_GENERIC_EVENTS 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XCURSOR 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XDBE 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XFIXES 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XINPUT2 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XINPUT2_SUPPORTS_MULTITOUCH 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XRANDR 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XSCRNSAVER 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XSHAPE 1
#cmakedefine SDL_VIDEO_DRIVER_X11_XSYNC 1
#cmakedefine SDL_VIDEO_DRIVER_QNX 1

#cmakedefine SDL_VIDEO_RENDER_D3D 1
#cmakedefine SDL_VIDEO_RENDER_D3D11 1
#cmakedefine SDL_VIDEO_RENDER_D3D12 1
#cmakedefine SDL_VIDEO_RENDER_GPU 1
#cmakedefine SDL_VIDEO_RENDER_METAL 1
#cmakedefine SDL_VIDEO_RENDER_VULKAN 1
#cmakedefine SDL_VIDEO_RENDER_OGL 1
#cmakedefine SDL_VIDEO_RENDER_OGL_ES2 1
#cmakedefine SDL_VIDEO_RENDER_PS2 1
#cmakedefine SDL_VIDEO_RENDER_PSP 1
#cmakedefine SDL_VIDEO_RENDER_VITA_GXM 1

/* Enable OpenGL support */
#cmakedefine SDL_VIDEO_OPENGL 1
#cmakedefine SDL_VIDEO_OPENGL_ES 1
#cmakedefine SDL_VIDEO_OPENGL_ES2 1
#cmakedefine SDL_VIDEO_OPENGL_CGL 1
#cmakedefine SDL_VIDEO_OPENGL_GLX 1
#cmakedefine SDL_VIDEO_OPENGL_WGL 1
#cmakedefine SDL_VIDEO_OPENGL_EGL 1

/* Enable Vulkan support */
#cmakedefine SDL_VIDEO_VULKAN 1

/* Enable Metal support */
#cmakedefine SDL_VIDEO_METAL 1

/* Enable GPU support */
#cmakedefine SDL_GPU_D3D11 1
#cmakedefine SDL_GPU_D3D12 1
#cmakedefine SDL_GPU_VULKAN 1
#cmakedefine SDL_GPU_METAL 1

/* Enable system power support */
#cmakedefine SDL_POWER_ANDROID 1
#cmakedefine SDL_POWER_LINUX 1
#cmakedefine SDL_POWER_WINDOWS 1
#cmakedefine SDL_POWER_MACOSX 1
#cmakedefine SDL_POWER_UIKIT 1
#cmakedefine SDL_POWER_HAIKU 1
#cmakedefine SDL_POWER_EMSCRIPTEN 1
#cmakedefine SDL_POWER_HARDWIRED 1
#cmakedefine SDL_POWER_VITA 1
#cmakedefine SDL_POWER_PSP 1
#cmakedefine SDL_POWER_N3DS 1

/* Enable system filesystem support */
#cmakedefine SDL_FILESYSTEM_ANDROID 1
#cmakedefine SDL_FILESYSTEM_HAIKU 1
#cmakedefine SDL_FILESYSTEM_COCOA 1
#cmakedefine SDL_FILESYSTEM_DUMMY 1
#cmakedefine SDL_FILESYSTEM_RISCOS 1
#cmakedefine SDL_FILESYSTEM_UNIX 1
#cmakedefine SDL_FILESYSTEM_WINDOWS 1
#cmakedefine SDL_FILESYSTEM_EMSCRIPTEN 1
#cmakedefine SDL_FILESYSTEM_VITA 1
#cmakedefine SDL_FILESYSTEM_PSP 1
#cmakedefine SDL_FILESYSTEM_PS2 1
#cmakedefine SDL_FILESYSTEM_N3DS 1

/* Enable system storage support */
#cmakedefine SDL_STORAGE_STEAM @SDL_STORAGE_STEAM@

/* Enable system FSops support */
#cmakedefine SDL_FSOPS_POSIX 1
#cmakedefine SDL_FSOPS_WINDOWS 1
#cmakedefine SDL_FSOPS_DUMMY 1

/* Enable camera subsystem */
#cmakedefine SDL_CAMERA_DRIVER_DUMMY 1
/* !!! FIXME: for later cmakedefine SDL_CAMERA_DRIVER_DISK 1 */
#cmakedefine SDL_CAMERA_DRIVER_V4L2 1
#cmakedefine SDL_CAMERA_DRIVER_COREMEDIA 1
#cmakedefine SDL_CAMERA_DRIVER_ANDROID 1
#cmakedefine SDL_CAMERA_DRIVER_EMSCRIPTEN 1
#cmakedefine SDL_CAMERA_DRIVER_MEDIAFOUNDATION 1
#cmakedefine SDL_CAMERA_DRIVER_PIPEWIRE 1
#cmakedefine SDL_CAMERA_DRIVER_PIPEWIRE_DYNAMIC @SDL_CAMERA_DRIVER_PIPEWIRE_DYNAMIC@
#cmakedefine SDL_CAMERA_DRIVER_VITA 1

/* Enable dialog subsystem */
#cmakedefine SDL_DIALOG_DUMMY 1

/* Enable assembly routines */
#cmakedefine SDL_ALTIVEC_BLITTERS 1

/* Whether SDL_DYNAMIC_API needs dlopen */
#cmakedefine DYNAPI_NEEDS_DLOPEN 1

/* Enable ime support */
#cmakedefine SDL_USE_IME 1
#cmakedefine SDL_DISABLE_WINDOWS_IME 1
#cmakedefine SDL_GDK_TEXTINPUT 1

/* Platform specific definitions */
#cmakedefine SDL_IPHONE_KEYBOARD 1
#cmakedefine SDL_IPHONE_LAUNCHSCREEN 1

#cmakedefine SDL_VIDEO_VITA_PIB 1
#cmakedefine SDL_VIDEO_VITA_PVR 1
#cmakedefine SDL_VIDEO_VITA_PVR_OGL 1

/* Libdecor version info */
#define SDL_LIBDECOR_VERSION_MAJOR @SDL_LIBDECOR_VERSION_MAJOR@
#define SDL_LIBDECOR_VERSION_MINOR @SDL_LIBDECOR_VERSION_MINOR@
#define SDL_LIBDECOR_VERSION_PATCH @SDL_LIBDECOR_VERSION_PATCH@

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
#cmakedefine SDL_DISABLE_SSE 1
#cmakedefine SDL_DISABLE_SSE2 1
#cmakedefine SDL_DISABLE_SSE3 1
#cmakedefine SDL_DISABLE_SSE4_1 1
#cmakedefine SDL_DISABLE_SSE4_2 1
#cmakedefine SDL_DISABLE_AVX 1
#cmakedefine SDL_DISABLE_AVX2 1
#cmakedefine SDL_DISABLE_AVX512F 1
#cmakedefine SDL_DISABLE_MMX 1
#cmakedefine SDL_DISABLE_LSX 1
#cmakedefine SDL_DISABLE_LASX 1
#cmakedefine SDL_DISABLE_NEON 1

#endif /* SDL_build_config_h_ */
