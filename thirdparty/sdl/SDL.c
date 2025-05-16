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
#include "SDL3/SDL_revision.h"

#if defined(SDL_PLATFORM_WINDOWS)
#include "core/windows/SDL_windows.h"
#else
#include <unistd.h> // _exit(), etc.
#endif

// this checks for HAVE_DBUS_DBUS_H internally.
#include "core/linux/SDL_dbus.h"

#ifdef SDL_PLATFORM_EMSCRIPTEN
#include <emscripten.h>
#endif

// Initialization code for SDL

#include "SDL_assert_c.h"
#include "SDL_hints_c.h"
#include "SDL_log_c.h"
#include "SDL_properties_c.h"
#include "audio/SDL_sysaudio.h"
#include "camera/SDL_camera_c.h"
#include "cpuinfo/SDL_cpuinfo_c.h"
#include "events/SDL_events_c.h"
#include "haptic/SDL_haptic_c.h"
#include "joystick/SDL_gamepad_c.h"
#include "joystick/SDL_joystick_c.h"
#include "render/SDL_sysrender.h"
#include "sensor/SDL_sensor_c.h"
#include "stdlib/SDL_getenv_c.h"
#include "thread/SDL_thread_c.h"
#include "tray/SDL_tray_utils.h"
#include "video/SDL_pixels_c.h"
#include "video/SDL_surface_c.h"
#include "video/SDL_video_c.h"
#include "filesystem/SDL_filesystem_c.h"
#include "io/SDL_asyncio_c.h"
#ifdef SDL_PLATFORM_ANDROID
#include "core/android/SDL_android.h"
#endif

#define SDL_INIT_EVERYTHING ~0U

// Initialization/Cleanup routines
#include "timer/SDL_timer_c.h"
#ifdef SDL_VIDEO_DRIVER_WINDOWS
extern bool SDL_HelperWindowCreate(void);
extern void SDL_HelperWindowDestroy(void);
#endif

#ifdef SDL_BUILD_MAJOR_VERSION
SDL_COMPILE_TIME_ASSERT(SDL_BUILD_MAJOR_VERSION,
                        SDL_MAJOR_VERSION == SDL_BUILD_MAJOR_VERSION);
SDL_COMPILE_TIME_ASSERT(SDL_BUILD_MINOR_VERSION,
                        SDL_MINOR_VERSION == SDL_BUILD_MINOR_VERSION);
SDL_COMPILE_TIME_ASSERT(SDL_BUILD_MICRO_VERSION,
                        SDL_MICRO_VERSION == SDL_BUILD_MICRO_VERSION);
#endif

// Limited by its encoding in SDL_VERSIONNUM
SDL_COMPILE_TIME_ASSERT(SDL_MAJOR_VERSION_min, SDL_MAJOR_VERSION >= 0);
SDL_COMPILE_TIME_ASSERT(SDL_MAJOR_VERSION_max, SDL_MAJOR_VERSION <= 10);
SDL_COMPILE_TIME_ASSERT(SDL_MINOR_VERSION_min, SDL_MINOR_VERSION >= 0);
SDL_COMPILE_TIME_ASSERT(SDL_MINOR_VERSION_max, SDL_MINOR_VERSION <= 999);
SDL_COMPILE_TIME_ASSERT(SDL_MICRO_VERSION_min, SDL_MICRO_VERSION >= 0);
SDL_COMPILE_TIME_ASSERT(SDL_MICRO_VERSION_max, SDL_MICRO_VERSION <= 999);

/* This is not declared in any header, although it is shared between some
    parts of SDL, because we don't want anything calling it without an
    extremely good reason. */
extern SDL_NORETURN void SDL_ExitProcess(int exitcode);
SDL_NORETURN void SDL_ExitProcess(int exitcode)
{
#if defined(SDL_PLATFORM_WINDOWS)
    /* "if you do not know the state of all threads in your process, it is
       better to call TerminateProcess than ExitProcess"
       https://msdn.microsoft.com/en-us/library/windows/desktop/ms682658(v=vs.85).aspx */
    TerminateProcess(GetCurrentProcess(), exitcode);
    /* MingW doesn't have TerminateProcess marked as noreturn, so add an
       ExitProcess here that will never be reached but make MingW happy. */
    ExitProcess(exitcode);
#elif defined(SDL_PLATFORM_EMSCRIPTEN)
    emscripten_cancel_main_loop();   // this should "kill" the app.
    emscripten_force_exit(exitcode); // this should "kill" the app.
    exit(exitcode);
#elif defined(SDL_PLATFORM_HAIKU)  // Haiku has _Exit, but it's not marked noreturn.
    _exit(exitcode);
#elif defined(HAVE__EXIT) // Upper case _Exit()
    _Exit(exitcode);
#else
    _exit(exitcode);
#endif
}

// App metadata

bool SDL_SetAppMetadata(const char *appname, const char *appversion, const char *appidentifier)
{
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING, appname);
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_VERSION_STRING, appversion);
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_IDENTIFIER_STRING, appidentifier);
    return true;
}

static bool SDL_ValidMetadataProperty(const char *name)
{
    if (!name || !*name) {
        return false;
    }

    if (SDL_strcmp(name, SDL_PROP_APP_METADATA_NAME_STRING) == 0 ||
        SDL_strcmp(name, SDL_PROP_APP_METADATA_VERSION_STRING) == 0 ||
        SDL_strcmp(name, SDL_PROP_APP_METADATA_IDENTIFIER_STRING) == 0 ||
        SDL_strcmp(name, SDL_PROP_APP_METADATA_CREATOR_STRING) == 0 ||
        SDL_strcmp(name, SDL_PROP_APP_METADATA_COPYRIGHT_STRING) == 0 ||
        SDL_strcmp(name, SDL_PROP_APP_METADATA_URL_STRING) == 0 ||
        SDL_strcmp(name, SDL_PROP_APP_METADATA_TYPE_STRING) == 0) {
        return true;
    }
    return false;
}

bool SDL_SetAppMetadataProperty(const char *name, const char *value)
{
    if (!SDL_ValidMetadataProperty(name)) {
        return SDL_InvalidParamError("name");
    }

    return SDL_SetStringProperty(SDL_GetGlobalProperties(), name, value);
}

const char *SDL_GetAppMetadataProperty(const char *name)
{
    if (!SDL_ValidMetadataProperty(name)) {
        SDL_InvalidParamError("name");
        return NULL;
    }

    const char *value = NULL;
    if (SDL_strcmp(name, SDL_PROP_APP_METADATA_NAME_STRING) == 0) {
        value = SDL_GetHint(SDL_HINT_APP_NAME);
    } else if (SDL_strcmp(name, SDL_PROP_APP_METADATA_IDENTIFIER_STRING) == 0) {
        value = SDL_GetHint(SDL_HINT_APP_ID);
    }
    if (!value || !*value) {
        value = SDL_GetStringProperty(SDL_GetGlobalProperties(), name, NULL);
    }
    if (!value || !*value) {
        if (SDL_strcmp(name, SDL_PROP_APP_METADATA_NAME_STRING) == 0) {
            value = "SDL Application";
        } else if (SDL_strcmp(name, SDL_PROP_APP_METADATA_TYPE_STRING) == 0) {
            value = "application";
        }
    }
    return value;
}


// The initialized subsystems
#ifdef SDL_MAIN_NEEDED
static bool SDL_MainIsReady = false;
#else
static bool SDL_MainIsReady = true;
#endif
static SDL_ThreadID SDL_MainThreadID = 0;
static bool SDL_bInMainQuit = false;
static Uint8 SDL_SubsystemRefCount[32];

// Private helper to increment a subsystem's ref counter.
static void SDL_IncrementSubsystemRefCount(Uint32 subsystem)
{
    const int subsystem_index = SDL_MostSignificantBitIndex32(subsystem);
    SDL_assert((subsystem_index < 0) || (SDL_SubsystemRefCount[subsystem_index] < 255));
    if (subsystem_index >= 0) {
        ++SDL_SubsystemRefCount[subsystem_index];
    }
}

// Private helper to decrement a subsystem's ref counter.
static void SDL_DecrementSubsystemRefCount(Uint32 subsystem)
{
    const int subsystem_index = SDL_MostSignificantBitIndex32(subsystem);
    if ((subsystem_index >= 0) && (SDL_SubsystemRefCount[subsystem_index] > 0)) {
        if (SDL_bInMainQuit) {
            SDL_SubsystemRefCount[subsystem_index] = 0;
        } else {
            --SDL_SubsystemRefCount[subsystem_index];
        }
    }
}

// Private helper to check if a system needs init.
static bool SDL_ShouldInitSubsystem(Uint32 subsystem)
{
    const int subsystem_index = SDL_MostSignificantBitIndex32(subsystem);
    SDL_assert((subsystem_index < 0) || (SDL_SubsystemRefCount[subsystem_index] < 255));
    return ((subsystem_index >= 0) && (SDL_SubsystemRefCount[subsystem_index] == 0));
}

// Private helper to check if a system needs to be quit.
static bool SDL_ShouldQuitSubsystem(Uint32 subsystem)
{
    const int subsystem_index = SDL_MostSignificantBitIndex32(subsystem);
    if ((subsystem_index >= 0) && (SDL_SubsystemRefCount[subsystem_index] == 0)) {
        return false;
    }

    /* If we're in SDL_Quit, we shut down every subsystem, even if refcount
     * isn't zero.
     */
    return (((subsystem_index >= 0) && (SDL_SubsystemRefCount[subsystem_index] == 1)) || SDL_bInMainQuit);
}

/* Private helper to either increment's existing ref counter,
 * or fully init a new subsystem. */
static bool SDL_InitOrIncrementSubsystem(Uint32 subsystem)
{
    int subsystem_index = SDL_MostSignificantBitIndex32(subsystem);
    SDL_assert((subsystem_index < 0) || (SDL_SubsystemRefCount[subsystem_index] < 255));
    if (subsystem_index < 0) {
        return false;
    }
    if (SDL_SubsystemRefCount[subsystem_index] > 0) {
        ++SDL_SubsystemRefCount[subsystem_index];
        return true;
    }
    return SDL_InitSubSystem(subsystem);
}

void SDL_SetMainReady(void)
{
    SDL_MainIsReady = true;

    if (SDL_MainThreadID == 0) {
        SDL_MainThreadID = SDL_GetCurrentThreadID();
    }
}

bool SDL_IsMainThread(void)
{
    if (SDL_MainThreadID == 0) {
        // Not initialized yet?
        return true;
    }
    if (SDL_MainThreadID == SDL_GetCurrentThreadID()) {
        return true;
    }
    return false;
}

// Initialize all the subsystems that require initialization before threads start
void SDL_InitMainThread(void)
{
    static bool done_info = false;

    SDL_InitTLSData();
    SDL_InitEnvironment();
    SDL_InitTicks();
    SDL_InitFilesystem();

    if (!done_info) {
        const char *value;

        value = SDL_GetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING);
        SDL_LogInfo(SDL_LOG_CATEGORY_SYSTEM, "App name: %s", value ? value : "<unspecified>");
        value = SDL_GetAppMetadataProperty(SDL_PROP_APP_METADATA_VERSION_STRING);
        SDL_LogInfo(SDL_LOG_CATEGORY_SYSTEM, "App version: %s", value ? value : "<unspecified>");
        value = SDL_GetAppMetadataProperty(SDL_PROP_APP_METADATA_IDENTIFIER_STRING);
        SDL_LogInfo(SDL_LOG_CATEGORY_SYSTEM, "App ID: %s", value ? value : "<unspecified>");
        SDL_LogInfo(SDL_LOG_CATEGORY_SYSTEM, "SDL revision: %s", SDL_REVISION);

        done_info = true;
    }
}

static void SDL_QuitMainThread(void)
{
    SDL_QuitFilesystem();
    SDL_QuitTicks();
    SDL_QuitEnvironment();
    SDL_QuitTLSData();
}

bool SDL_InitSubSystem(SDL_InitFlags flags)
{
    Uint32 flags_initialized = 0;

    if (!SDL_MainIsReady) {
        return SDL_SetError("Application didn't initialize properly, did you include SDL_main.h in the file containing your main() function?");
    }

    SDL_InitMainThread();

#ifdef SDL_USE_LIBDBUS
    SDL_DBus_Init();
#endif

#ifdef SDL_VIDEO_DRIVER_WINDOWS
    if (flags & (SDL_INIT_HAPTIC | SDL_INIT_JOYSTICK)) {
        if (!SDL_HelperWindowCreate()) {
            goto quit_and_error;
        }
    }
#endif

    // Initialize the event subsystem
    if (flags & SDL_INIT_EVENTS) {
        if (SDL_ShouldInitSubsystem(SDL_INIT_EVENTS)) {
            SDL_IncrementSubsystemRefCount(SDL_INIT_EVENTS);
            if (!SDL_InitEvents()) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_EVENTS);
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_EVENTS);
        }
        flags_initialized |= SDL_INIT_EVENTS;
    }

    // Initialize the video subsystem
    if (flags & SDL_INIT_VIDEO) {
#ifndef SDL_VIDEO_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_VIDEO)) {
            // video implies events
            if (!SDL_InitOrIncrementSubsystem(SDL_INIT_EVENTS)) {
                goto quit_and_error;
            }

            // We initialize video on the main thread
            // On Apple platforms this is a requirement.
            // On other platforms, this is the definition.
            SDL_MainThreadID = SDL_GetCurrentThreadID();

            SDL_IncrementSubsystemRefCount(SDL_INIT_VIDEO);
            if (!SDL_VideoInit(NULL)) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_VIDEO);
                SDL_PushError();
                SDL_QuitSubSystem(SDL_INIT_EVENTS);
                SDL_PopError();
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_VIDEO);
        }
        flags_initialized |= SDL_INIT_VIDEO;
#else
        SDL_SetError("SDL not built with video support");
        goto quit_and_error;
#endif
    }

    // Initialize the audio subsystem
    if (flags & SDL_INIT_AUDIO) {
#ifndef SDL_AUDIO_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_AUDIO)) {
            // audio implies events
            if (!SDL_InitOrIncrementSubsystem(SDL_INIT_EVENTS)) {
                goto quit_and_error;
            }

            SDL_IncrementSubsystemRefCount(SDL_INIT_AUDIO);
            if (!SDL_InitAudio(NULL)) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_AUDIO);
                SDL_PushError();
                SDL_QuitSubSystem(SDL_INIT_EVENTS);
                SDL_PopError();
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_AUDIO);
        }
        flags_initialized |= SDL_INIT_AUDIO;
#else
        SDL_SetError("SDL not built with audio support");
        goto quit_and_error;
#endif
    }

    // Initialize the joystick subsystem
    if (flags & SDL_INIT_JOYSTICK) {
#ifndef SDL_JOYSTICK_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_JOYSTICK)) {
            // joystick implies events
            if (!SDL_InitOrIncrementSubsystem(SDL_INIT_EVENTS)) {
                goto quit_and_error;
            }

            SDL_IncrementSubsystemRefCount(SDL_INIT_JOYSTICK);
            if (!SDL_InitJoysticks()) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_JOYSTICK);
                SDL_PushError();
                SDL_QuitSubSystem(SDL_INIT_EVENTS);
                SDL_PopError();
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_JOYSTICK);
        }
        flags_initialized |= SDL_INIT_JOYSTICK;
#else
        SDL_SetError("SDL not built with joystick support");
        goto quit_and_error;
#endif
    }

    if (flags & SDL_INIT_GAMEPAD) {
#ifndef SDL_JOYSTICK_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_GAMEPAD)) {
            // game controller implies joystick
            if (!SDL_InitOrIncrementSubsystem(SDL_INIT_JOYSTICK)) {
                goto quit_and_error;
            }

            SDL_IncrementSubsystemRefCount(SDL_INIT_GAMEPAD);
            if (!SDL_InitGamepads()) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_GAMEPAD);
                SDL_PushError();
                SDL_QuitSubSystem(SDL_INIT_JOYSTICK);
                SDL_PopError();
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_GAMEPAD);
        }
        flags_initialized |= SDL_INIT_GAMEPAD;
#else
        SDL_SetError("SDL not built with joystick support");
        goto quit_and_error;
#endif
    }

    // Initialize the haptic subsystem
    if (flags & SDL_INIT_HAPTIC) {
#ifndef SDL_HAPTIC_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_HAPTIC)) {
            SDL_IncrementSubsystemRefCount(SDL_INIT_HAPTIC);
            if (!SDL_InitHaptics()) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_HAPTIC);
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_HAPTIC);
        }
        flags_initialized |= SDL_INIT_HAPTIC;
#else
        SDL_SetError("SDL not built with haptic (force feedback) support");
        goto quit_and_error;
#endif
    }

    // Initialize the sensor subsystem
    if (flags & SDL_INIT_SENSOR) {
#ifndef SDL_SENSOR_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_SENSOR)) {
            SDL_IncrementSubsystemRefCount(SDL_INIT_SENSOR);
            if (!SDL_InitSensors()) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_SENSOR);
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_SENSOR);
        }
        flags_initialized |= SDL_INIT_SENSOR;
#else
        SDL_SetError("SDL not built with sensor support");
        goto quit_and_error;
#endif
    }

    // Initialize the camera subsystem
    if (flags & SDL_INIT_CAMERA) {
#ifndef SDL_CAMERA_DISABLED
        if (SDL_ShouldInitSubsystem(SDL_INIT_CAMERA)) {
            // camera implies events
            if (!SDL_InitOrIncrementSubsystem(SDL_INIT_EVENTS)) {
                goto quit_and_error;
            }

            SDL_IncrementSubsystemRefCount(SDL_INIT_CAMERA);
            if (!SDL_CameraInit(NULL)) {
                SDL_DecrementSubsystemRefCount(SDL_INIT_CAMERA);
                SDL_PushError();
                SDL_QuitSubSystem(SDL_INIT_EVENTS);
                SDL_PopError();
                goto quit_and_error;
            }
        } else {
            SDL_IncrementSubsystemRefCount(SDL_INIT_CAMERA);
        }
        flags_initialized |= SDL_INIT_CAMERA;
#else
        SDL_SetError("SDL not built with camera support");
        goto quit_and_error;
#endif
    }

    (void)flags_initialized; // make static analysis happy, since this only gets used in error cases.

    return SDL_ClearError();

quit_and_error:
    {
        SDL_PushError();
        SDL_QuitSubSystem(flags_initialized);
        SDL_PopError();
    }
    return false;
}

bool SDL_Init(SDL_InitFlags flags)
{
    return SDL_InitSubSystem(flags);
}

void SDL_QuitSubSystem(SDL_InitFlags flags)
{
    // Shut down requested initialized subsystems

#ifndef SDL_CAMERA_DISABLED
    if (flags & SDL_INIT_CAMERA) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_CAMERA)) {
            SDL_QuitCamera();
            // camera implies events
            SDL_QuitSubSystem(SDL_INIT_EVENTS);
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_CAMERA);
    }
#endif

#ifndef SDL_SENSOR_DISABLED
    if (flags & SDL_INIT_SENSOR) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_SENSOR)) {
            SDL_QuitSensors();
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_SENSOR);
    }
#endif

#ifndef SDL_JOYSTICK_DISABLED
    if (flags & SDL_INIT_GAMEPAD) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_GAMEPAD)) {
            SDL_QuitGamepads();
            // game controller implies joystick
            SDL_QuitSubSystem(SDL_INIT_JOYSTICK);
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_GAMEPAD);
    }

    if (flags & SDL_INIT_JOYSTICK) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_JOYSTICK)) {
            SDL_QuitJoysticks();
            // joystick implies events
            SDL_QuitSubSystem(SDL_INIT_EVENTS);
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_JOYSTICK);
    }
#endif

#ifndef SDL_HAPTIC_DISABLED
    if (flags & SDL_INIT_HAPTIC) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_HAPTIC)) {
            SDL_QuitHaptics();
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_HAPTIC);
    }
#endif

#ifndef SDL_AUDIO_DISABLED
    if (flags & SDL_INIT_AUDIO) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_AUDIO)) {
            SDL_QuitAudio();
            // audio implies events
            SDL_QuitSubSystem(SDL_INIT_EVENTS);
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_AUDIO);
    }
#endif

#ifndef SDL_VIDEO_DISABLED
    if (flags & SDL_INIT_VIDEO) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_VIDEO)) {
            SDL_QuitRender();
            SDL_VideoQuit();
            // video implies events
            SDL_QuitSubSystem(SDL_INIT_EVENTS);
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_VIDEO);
    }
#endif

    if (flags & SDL_INIT_EVENTS) {
        if (SDL_ShouldQuitSubsystem(SDL_INIT_EVENTS)) {
            SDL_QuitEvents();
        }
        SDL_DecrementSubsystemRefCount(SDL_INIT_EVENTS);
    }
}

Uint32 SDL_WasInit(SDL_InitFlags flags)
{
    int i;
    int num_subsystems = SDL_arraysize(SDL_SubsystemRefCount);
    Uint32 initialized = 0;

    // Fast path for checking one flag
    if (SDL_HasExactlyOneBitSet32(flags)) {
        int subsystem_index = SDL_MostSignificantBitIndex32(flags);
        return SDL_SubsystemRefCount[subsystem_index] ? flags : 0;
    }

    if (!flags) {
        flags = SDL_INIT_EVERYTHING;
    }

    num_subsystems = SDL_min(num_subsystems, SDL_MostSignificantBitIndex32(flags) + 1);

    // Iterate over each bit in flags, and check the matching subsystem.
    for (i = 0; i < num_subsystems; ++i) {
        if ((flags & 1) && SDL_SubsystemRefCount[i] > 0) {
            initialized |= (1 << i);
        }

        flags >>= 1;
    }

    return initialized;
}

void SDL_Quit(void)
{
    SDL_bInMainQuit = true;

    // Quit all subsystems
#ifdef SDL_VIDEO_DRIVER_WINDOWS
    SDL_HelperWindowDestroy();
#endif
    SDL_QuitSubSystem(SDL_INIT_EVERYTHING);
    SDL_CleanupTrays();

#ifdef SDL_USE_LIBDBUS
    SDL_DBus_Quit();
#endif

    SDL_QuitTimers();
    SDL_QuitAsyncIO();

    SDL_SetObjectsInvalid();
    SDL_AssertionsQuit();

    SDL_QuitPixelFormatDetails();

    SDL_QuitCPUInfo();

    /* Now that every subsystem has been quit, we reset the subsystem refcount
     * and the list of initialized subsystems.
     */
    SDL_memset(SDL_SubsystemRefCount, 0x0, sizeof(SDL_SubsystemRefCount));

    SDL_QuitLog();
    SDL_QuitHints();
    SDL_QuitProperties();

    SDL_QuitMainThread();

    SDL_bInMainQuit = false;
}

// Get the library version number
int SDL_GetVersion(void)
{
    return SDL_VERSION;
}

// Get the library source revision
const char *SDL_GetRevision(void)
{
    return SDL_REVISION;
}

// Get the name of the platform
const char *SDL_GetPlatform(void)
{
#if defined(SDL_PLATFORM_PRIVATE)
    return SDL_PLATFORM_PRIVATE_NAME;
#elif defined(SDL_PLATFORM_AIX)
    return "AIX";
#elif defined(SDL_PLATFORM_ANDROID)
    return "Android";
#elif defined(SDL_PLATFORM_BSDI)
    return "BSDI";
#elif defined(SDL_PLATFORM_EMSCRIPTEN)
    return "Emscripten";
#elif defined(SDL_PLATFORM_FREEBSD)
    return "FreeBSD";
#elif defined(SDL_PLATFORM_HAIKU)
    return "Haiku";
#elif defined(SDL_PLATFORM_HPUX)
    return "HP-UX";
#elif defined(SDL_PLATFORM_IRIX)
    return "Irix";
#elif defined(SDL_PLATFORM_LINUX)
    return "Linux";
#elif defined(__MINT__)
    return "Atari MiNT";
#elif defined(SDL_PLATFORM_MACOS)
    return "macOS";
#elif defined(SDL_PLATFORM_NETBSD)
    return "NetBSD";
#elif defined(SDL_PLATFORM_OPENBSD)
    return "OpenBSD";
#elif defined(SDL_PLATFORM_OS2)
    return "OS/2";
#elif defined(SDL_PLATFORM_OSF)
    return "OSF/1";
#elif defined(SDL_PLATFORM_QNXNTO)
    return "QNX Neutrino";
#elif defined(SDL_PLATFORM_RISCOS)
    return "RISC OS";
#elif defined(SDL_PLATFORM_SOLARIS)
    return "Solaris";
#elif defined(SDL_PLATFORM_WIN32)
    return "Windows";
#elif defined(SDL_PLATFORM_WINGDK)
    return "WinGDK";
#elif defined(SDL_PLATFORM_XBOXONE)
    return "Xbox One";
#elif defined(SDL_PLATFORM_XBOXSERIES)
    return "Xbox Series X|S";
#elif defined(SDL_PLATFORM_IOS)
    return "iOS";
#elif defined(SDL_PLATFORM_TVOS)
    return "tvOS";
#elif defined(SDL_PLATFORM_PS2)
    return "PlayStation 2";
#elif defined(SDL_PLATFORM_PSP)
    return "PlayStation Portable";
#elif defined(SDL_PLATFORM_VITA)
    return "PlayStation Vita";
#elif defined(SDL_PLATFORM_3DS)
    return "Nintendo 3DS";
#elif defined(__managarm__)
    return "Managarm";
#else
    return "Unknown (see SDL_platform.h)";
#endif
}

bool SDL_IsTablet(void)
{
#ifdef SDL_PLATFORM_ANDROID
    return SDL_IsAndroidTablet();
#elif defined(SDL_PLATFORM_IOS)
    extern bool SDL_IsIPad(void);
    return SDL_IsIPad();
#else
    return false;
#endif
}

bool SDL_IsTV(void)
{
#ifdef SDL_PLATFORM_ANDROID
    return SDL_IsAndroidTV();
#elif defined(SDL_PLATFORM_IOS)
    extern bool SDL_IsAppleTV(void);
    return SDL_IsAppleTV();
#else
    return false;
#endif
}

static SDL_Sandbox SDL_DetectSandbox(void)
{
#if defined(SDL_PLATFORM_LINUX)
    if (access("/.flatpak-info", F_OK) == 0) {
        return SDL_SANDBOX_FLATPAK;
    }

    /* For Snap, we check multiple variables because they might be set for
     * unrelated reasons. This is the same thing WebKitGTK does. */
    if (SDL_getenv("SNAP") && SDL_getenv("SNAP_NAME") && SDL_getenv("SNAP_REVISION")) {
        return SDL_SANDBOX_SNAP;
    }

    if (access("/run/host/container-manager", F_OK) == 0) {
        return SDL_SANDBOX_UNKNOWN_CONTAINER;
    }

#elif defined(SDL_PLATFORM_MACOS)
    if (SDL_getenv("APP_SANDBOX_CONTAINER_ID")) {
        return SDL_SANDBOX_MACOS;
    }
#endif

    return SDL_SANDBOX_NONE;
}

SDL_Sandbox SDL_GetSandbox(void)
{
    static SDL_Sandbox sandbox;
    static bool sandbox_initialized;

    if (!sandbox_initialized) {
        sandbox = SDL_DetectSandbox();
        sandbox_initialized = true;
    }
    return sandbox;
}

#ifdef SDL_PLATFORM_WIN32

#if (!defined(HAVE_LIBC) || defined(__WATCOMC__)) && !defined(SDL_STATIC_LIB)
// FIXME: Still need to include DllMain() on Watcom C ?

BOOL APIENTRY MINGW32_FORCEALIGN _DllMainCRTStartup(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}
#endif // Building DLL

#endif // defined(SDL_PLATFORM_WIN32)
