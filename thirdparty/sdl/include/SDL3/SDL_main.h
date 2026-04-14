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
 * # CategoryMain
 *
 * Redefine main() if necessary so that it is called by SDL.
 *
 * In order to make this consistent on all platforms, the application's main()
 * should look like this:
 *
 * ```c
 * #include <SDL3/SDL.h>
 * #include <SDL3/SDL_main.h>
 *
 * int main(int argc, char *argv[])
 * {
 * }
 * ```
 *
 * SDL will take care of platform specific details on how it gets called.
 *
 * This is also where an app can be configured to use the main callbacks, via
 * the SDL_MAIN_USE_CALLBACKS macro.
 *
 * SDL_main.h is a "single-header library," which is to say that including
 * this header inserts code into your program, and you should only include it
 * once in most cases. SDL.h does not include this header automatically.
 *
 * For more information, see:
 *
 * https://wiki.libsdl.org/SDL3/README/main-functions
 */

#ifndef SDL_main_h_
#define SDL_main_h_

#include <SDL3/SDL_platform_defines.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_error.h>
#include <SDL3/SDL_events.h>

#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * Inform SDL that the app is providing an entry point instead of SDL.
 *
 * SDL does not define this macro, but will check if it is defined when
 * including `SDL_main.h`. If defined, SDL will expect the app to provide the
 * proper entry point for the platform, and all the other magic details
 * needed, like manually calling SDL_SetMainReady.
 *
 * Please see [README/main-functions](README/main-functions), (or
 * docs/README-main-functions.md in the source tree) for a more detailed
 * explanation.
 *
 * \since This macro is used by the headers since SDL 3.2.0.
 */
#define SDL_MAIN_HANDLED 1

/**
 * Inform SDL to use the main callbacks instead of main.
 *
 * SDL does not define this macro, but will check if it is defined when
 * including `SDL_main.h`. If defined, SDL will expect the app to provide
 * several functions: SDL_AppInit, SDL_AppEvent, SDL_AppIterate, and
 * SDL_AppQuit. The app should not provide a `main` function in this case, and
 * doing so will likely cause the build to fail.
 *
 * Please see [README/main-functions](README/main-functions), (or
 * docs/README-main-functions.md in the source tree) for a more detailed
 * explanation.
 *
 * \since This macro is used by the headers since SDL 3.2.0.
 *
 * \sa SDL_AppInit
 * \sa SDL_AppEvent
 * \sa SDL_AppIterate
 * \sa SDL_AppQuit
 */
#define SDL_MAIN_USE_CALLBACKS 1

/**
 * Defined if the target platform offers a special mainline through SDL.
 *
 * This won't be defined otherwise. If defined, SDL's headers will redefine
 * `main` to `SDL_main`.
 *
 * This macro is defined by `SDL_main.h`, which is not automatically included
 * by `SDL.h`.
 *
 * Even if available, an app can define SDL_MAIN_HANDLED and provide their
 * own, if they know what they're doing.
 *
 * This macro is used internally by SDL, and apps probably shouldn't rely on it.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_MAIN_AVAILABLE

/**
 * Defined if the target platform _requires_ a special mainline through SDL.
 *
 * This won't be defined otherwise. If defined, SDL's headers will redefine
 * `main` to `SDL_main`.
 *
 * This macro is defined by `SDL_main.h`, which is not automatically included
 * by `SDL.h`.
 *
 * Even if required, an app can define SDL_MAIN_HANDLED and provide their
 * own, if they know what they're doing.
 *
 * This macro is used internally by SDL, and apps probably shouldn't rely on it.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_MAIN_NEEDED

#endif

#if defined(__has_include)
    #if __has_include("SDL_main_private.h") && __has_include("SDL_main_impl_private.h")
        #define SDL_PLATFORM_PRIVATE_MAIN
    #endif
#endif

#ifndef SDL_MAIN_HANDLED
    #if defined(SDL_PLATFORM_PRIVATE_MAIN)
        /* Private platforms may have their own ideas about entry points. */
        #include "SDL_main_private.h"

    #elif defined(SDL_PLATFORM_WIN32)
        /* On Windows SDL provides WinMain(), which parses the command line and passes
           the arguments to your main function.

           If you provide your own WinMain(), you may define SDL_MAIN_HANDLED
         */
        #define SDL_MAIN_AVAILABLE

    #elif defined(SDL_PLATFORM_GDK)
        /* On GDK, SDL provides a main function that initializes the game runtime.

           If you prefer to write your own WinMain-function instead of having SDL
           provide one that calls your main() function,
           #define SDL_MAIN_HANDLED before #include'ing SDL_main.h
           and call the SDL_RunApp function from your entry point.
        */
        #define SDL_MAIN_NEEDED

    #elif defined(SDL_PLATFORM_IOS)
        /* On iOS SDL provides a main function that creates an application delegate
           and starts the iOS application run loop.

           To use it, just #include SDL_main.h in the source file that contains your
           main() function.

           See src/video/uikit/SDL_uikitappdelegate.m for more details.
         */
        #define SDL_MAIN_NEEDED

    #elif defined(SDL_PLATFORM_ANDROID)
        /* On Android SDL provides a Java class in SDLActivity.java that is the
           main activity entry point.

           See docs/README-android.md for more details on extending that class.
         */
        #define SDL_MAIN_NEEDED

        /* As this is launched from Java, the real entry point (main() function)
           is outside of the the binary built from this code.
           This define makes sure that, unlike on other platforms, SDL_main.h
           and SDL_main_impl.h export an `SDL_main()` function (to be called
           from Java), but don't implement a native `int main(int argc, char* argv[])`
           or similar.
         */
        #define SDL_MAIN_EXPORTED

    #elif defined(SDL_PLATFORM_EMSCRIPTEN)
        /* On Emscripten, SDL provides a main function that converts URL
           parameters that start with "SDL_" to environment variables, so
           they can be used as SDL hints, etc.

           This is 100% optional, so if you don't want this to happen, you may
           define SDL_MAIN_HANDLED
         */
        #define SDL_MAIN_AVAILABLE

    #elif defined(SDL_PLATFORM_PSP)
        /* On PSP SDL provides a main function that sets the module info,
           activates the GPU and starts the thread required to be able to exit
           the software.

           If you provide this yourself, you may define SDL_MAIN_HANDLED
         */
        #define SDL_MAIN_AVAILABLE

    #elif defined(SDL_PLATFORM_PS2)
        #define SDL_MAIN_AVAILABLE

        #define SDL_PS2_SKIP_IOP_RESET() \
           void reset_IOP(); \
           void reset_IOP() {}

    #elif defined(SDL_PLATFORM_3DS)
        /*
          On N3DS, SDL provides a main function that sets up the screens
          and storage.

          If you provide this yourself, you may define SDL_MAIN_HANDLED
        */
        #define SDL_MAIN_AVAILABLE

    #endif
#endif /* SDL_MAIN_HANDLED */


#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * A macro to tag a main entry point function as exported.
 *
 * Most platforms don't need this, and the macro will be defined to nothing.
 * Some, like Android, keep the entry points in a shared library and need to
 * explicitly export the symbols.
 *
 * External code rarely needs this, and if it needs something, it's almost
 * always SDL_DECLSPEC instead.
 *
 * \since This macro is available since SDL 3.2.0.
 *
 * \sa SDL_DECLSPEC
 */
#define SDLMAIN_DECLSPEC

#elif defined(SDL_MAIN_EXPORTED)
/* We need to export SDL_main so it can be launched from external code,
   like SDLActivity.java on Android */
#define SDLMAIN_DECLSPEC    SDL_DECLSPEC
#else
/* usually this is empty */
#define SDLMAIN_DECLSPEC
#endif /* SDL_MAIN_EXPORTED */

#if defined(SDL_MAIN_NEEDED) || defined(SDL_MAIN_AVAILABLE) || defined(SDL_MAIN_USE_CALLBACKS)
#define main SDL_main
#endif

#include <SDL3/SDL_init.h>
#include <SDL3/SDL_begin_code.h>
#ifdef __cplusplus
extern "C" {
#endif

/*
 * You can (optionally!) define SDL_MAIN_USE_CALLBACKS before including
 * SDL_main.h, and then your application will _not_ have a standard
 * "main" entry point. Instead, it will operate as a collection of
 * functions that are called as necessary by the system. On some
 * platforms, this is just a layer where SDL drives your program
 * instead of your program driving SDL, on other platforms this might
 * hook into the OS to manage the lifecycle. Programs on most platforms
 * can use whichever approach they prefer, but the decision boils down
 * to:
 *
 * - Using a standard "main" function: this works like it always has for
 *   the past 50+ years in C programming, and your app is in control.
 * - Using the callback functions: this might clean up some code,
 *   avoid some #ifdef blocks in your program for some platforms, be more
 *   resource-friendly to the system, and possibly be the primary way to
 *   access some future platforms (but none require this at the moment).
 *
 * This is up to the app; both approaches are considered valid and supported
 * ways to write SDL apps.
 *
 * If using the callbacks, don't define a "main" function. Instead, implement
 * the functions listed below in your program.
 */
#ifdef SDL_MAIN_USE_CALLBACKS

/**
 * App-implemented initial entry point for SDL_MAIN_USE_CALLBACKS apps.
 *
 * Apps implement this function when using SDL_MAIN_USE_CALLBACKS. If using a
 * standard "main" function, you should not supply this.
 *
 * This function is called by SDL once, at startup. The function should
 * initialize whatever is necessary, possibly create windows and open audio
 * devices, etc. The `argc` and `argv` parameters work like they would with a
 * standard "main" function.
 *
 * This function should not go into an infinite mainloop; it should do any
 * one-time setup it requires and then return.
 *
 * The app may optionally assign a pointer to `*appstate`. This pointer will
 * be provided on every future call to the other entry points, to allow
 * application state to be preserved between functions without the app needing
 * to use a global variable. If this isn't set, the pointer will be NULL in
 * future entry points.
 *
 * If this function returns SDL_APP_CONTINUE, the app will proceed to normal
 * operation, and will begin receiving repeated calls to SDL_AppIterate and
 * SDL_AppEvent for the life of the program. If this function returns
 * SDL_APP_FAILURE, SDL will call SDL_AppQuit and terminate the process with
 * an exit code that reports an error to the platform. If it returns
 * SDL_APP_SUCCESS, SDL calls SDL_AppQuit and terminates with an exit code
 * that reports success to the platform.
 *
 * This function is called by SDL on the main thread.
 *
 * \param appstate a place where the app can optionally store a pointer for
 *                 future use.
 * \param argc the standard ANSI C main's argc; number of elements in `argv`.
 * \param argv the standard ANSI C main's argv; array of command line
 *             arguments.
 * \returns SDL_APP_FAILURE to terminate with an error, SDL_APP_SUCCESS to
 *          terminate with success, SDL_APP_CONTINUE to continue.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_AppIterate
 * \sa SDL_AppEvent
 * \sa SDL_AppQuit
 */
extern SDLMAIN_DECLSPEC SDL_AppResult SDLCALL SDL_AppInit(void **appstate, int argc, char *argv[]);

/**
 * App-implemented iteration entry point for SDL_MAIN_USE_CALLBACKS apps.
 *
 * Apps implement this function when using SDL_MAIN_USE_CALLBACKS. If using a
 * standard "main" function, you should not supply this.
 *
 * This function is called repeatedly by SDL after SDL_AppInit returns 0. The
 * function should operate as a single iteration the program's primary loop;
 * it should update whatever state it needs and draw a new frame of video,
 * usually.
 *
 * On some platforms, this function will be called at the refresh rate of the
 * display (which might change during the life of your app!). There are no
 * promises made about what frequency this function might run at. You should
 * use SDL's timer functions if you need to see how much time has passed since
 * the last iteration.
 *
 * There is no need to process the SDL event queue during this function; SDL
 * will send events as they arrive in SDL_AppEvent, and in most cases the
 * event queue will be empty when this function runs anyhow.
 *
 * This function should not go into an infinite mainloop; it should do one
 * iteration of whatever the program does and return.
 *
 * The `appstate` parameter is an optional pointer provided by the app during
 * SDL_AppInit(). If the app never provided a pointer, this will be NULL.
 *
 * If this function returns SDL_APP_CONTINUE, the app will continue normal
 * operation, receiving repeated calls to SDL_AppIterate and SDL_AppEvent for
 * the life of the program. If this function returns SDL_APP_FAILURE, SDL will
 * call SDL_AppQuit and terminate the process with an exit code that reports
 * an error to the platform. If it returns SDL_APP_SUCCESS, SDL calls
 * SDL_AppQuit and terminates with an exit code that reports success to the
 * platform.
 *
 * This function is called by SDL on the main thread.
 *
 * \param appstate an optional pointer, provided by the app in SDL_AppInit.
 * \returns SDL_APP_FAILURE to terminate with an error, SDL_APP_SUCCESS to
 *          terminate with success, SDL_APP_CONTINUE to continue.
 *
 * \threadsafety This function may get called concurrently with SDL_AppEvent()
 *               for events not pushed on the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_AppInit
 * \sa SDL_AppEvent
 */
extern SDLMAIN_DECLSPEC SDL_AppResult SDLCALL SDL_AppIterate(void *appstate);

/**
 * App-implemented event entry point for SDL_MAIN_USE_CALLBACKS apps.
 *
 * Apps implement this function when using SDL_MAIN_USE_CALLBACKS. If using a
 * standard "main" function, you should not supply this.
 *
 * This function is called as needed by SDL after SDL_AppInit returns
 * SDL_APP_CONTINUE. It is called once for each new event.
 *
 * There is (currently) no guarantee about what thread this will be called
 * from; whatever thread pushes an event onto SDL's queue will trigger this
 * function. SDL is responsible for pumping the event queue between each call
 * to SDL_AppIterate, so in normal operation one should only get events in a
 * serial fashion, but be careful if you have a thread that explicitly calls
 * SDL_PushEvent. SDL itself will push events to the queue on the main thread.
 *
 * Events sent to this function are not owned by the app; if you need to save
 * the data, you should copy it.
 *
 * This function should not go into an infinite mainloop; it should handle the
 * provided event appropriately and return.
 *
 * The `appstate` parameter is an optional pointer provided by the app during
 * SDL_AppInit(). If the app never provided a pointer, this will be NULL.
 *
 * If this function returns SDL_APP_CONTINUE, the app will continue normal
 * operation, receiving repeated calls to SDL_AppIterate and SDL_AppEvent for
 * the life of the program. If this function returns SDL_APP_FAILURE, SDL will
 * call SDL_AppQuit and terminate the process with an exit code that reports
 * an error to the platform. If it returns SDL_APP_SUCCESS, SDL calls
 * SDL_AppQuit and terminates with an exit code that reports success to the
 * platform.
 *
 * \param appstate an optional pointer, provided by the app in SDL_AppInit.
 * \param event the new event for the app to examine.
 * \returns SDL_APP_FAILURE to terminate with an error, SDL_APP_SUCCESS to
 *          terminate with success, SDL_APP_CONTINUE to continue.
 *
 * \threadsafety This function may get called concurrently with
 *               SDL_AppIterate() or SDL_AppQuit() for events not pushed from
 *               the main thread.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_AppInit
 * \sa SDL_AppIterate
 */
extern SDLMAIN_DECLSPEC SDL_AppResult SDLCALL SDL_AppEvent(void *appstate, SDL_Event *event);

/**
 * App-implemented deinit entry point for SDL_MAIN_USE_CALLBACKS apps.
 *
 * Apps implement this function when using SDL_MAIN_USE_CALLBACKS. If using a
 * standard "main" function, you should not supply this.
 *
 * This function is called once by SDL before terminating the program.
 *
 * This function will be called no matter what, even if SDL_AppInit requests
 * termination.
 *
 * This function should not go into an infinite mainloop; it should
 * deinitialize any resources necessary, perform whatever shutdown activities,
 * and return.
 *
 * You do not need to call SDL_Quit() in this function, as SDL will call it
 * after this function returns and before the process terminates, but it is
 * safe to do so.
 *
 * The `appstate` parameter is an optional pointer provided by the app during
 * SDL_AppInit(). If the app never provided a pointer, this will be NULL. This
 * function call is the last time this pointer will be provided, so any
 * resources to it should be cleaned up here.
 *
 * This function is called by SDL on the main thread.
 *
 * \param appstate an optional pointer, provided by the app in SDL_AppInit.
 * \param result the result code that terminated the app (success or failure).
 *
 * \threadsafety SDL_AppEvent() may get called concurrently with this function
 *               if other threads that push events are still active.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_AppInit
 */
extern SDLMAIN_DECLSPEC void SDLCALL SDL_AppQuit(void *appstate, SDL_AppResult result);

#endif  /* SDL_MAIN_USE_CALLBACKS */


/**
 * The prototype for the application's main() function
 *
 * \param argc an ANSI-C style main function's argc.
 * \param argv an ANSI-C style main function's argv.
 * \returns an ANSI-C main return code; generally 0 is considered successful
 *          program completion, and small non-zero values are considered
 *          errors.
 *
 * \since This datatype is available since SDL 3.2.0.
 */
typedef int (SDLCALL *SDL_main_func)(int argc, char *argv[]);

/**
 * An app-supplied function for program entry.
 *
 * Apps do not directly create this function; they should create a standard
 * ANSI-C `main` function instead. If SDL needs to insert some startup code
 * before `main` runs, or the platform doesn't actually _use_ a function
 * called "main", SDL will do some macro magic to redefine `main` to
 * `SDL_main` and provide its own `main`.
 *
 * Apps should include `SDL_main.h` in the same file as their `main` function,
 * and they should not use that symbol for anything else in that file, as it
 * might get redefined.
 *
 * This function is only provided by the app if it isn't using
 * SDL_MAIN_USE_CALLBACKS.
 *
 * Program startup is a surprisingly complex topic. Please see
 * [README/main-functions](README/main-functions), (or
 * docs/README-main-functions.md in the source tree) for a more detailed
 * explanation.
 *
 * \param argc an ANSI-C style main function's argc.
 * \param argv an ANSI-C style main function's argv.
 * \returns an ANSI-C main return code; generally 0 is considered successful
 *          program completion, and small non-zero values are considered
 *          errors.
 *
 * \threadsafety This is the program entry point.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDLMAIN_DECLSPEC int SDLCALL SDL_main(int argc, char *argv[]);

/**
 * Circumvent failure of SDL_Init() when not using SDL_main() as an entry
 * point.
 *
 * This function is defined in SDL_main.h, along with the preprocessor rule to
 * redefine main() as SDL_main(). Thus to ensure that your main() function
 * will not be changed it is necessary to define SDL_MAIN_HANDLED before
 * including SDL.h.
 *
 * \since This function is available since SDL 3.2.0.
 *
 * \sa SDL_Init
 */
extern SDL_DECLSPEC void SDLCALL SDL_SetMainReady(void);

/**
 * Initializes and launches an SDL application, by doing platform-specific
 * initialization before calling your mainFunction and cleanups after it
 * returns, if that is needed for a specific platform, otherwise it just calls
 * mainFunction.
 *
 * You can use this if you want to use your own main() implementation without
 * using SDL_main (like when using SDL_MAIN_HANDLED). When using this, you do
 * *not* need SDL_SetMainReady().
 *
 * \param argc the argc parameter from the application's main() function, or 0
 *             if the platform's main-equivalent has no argc.
 * \param argv the argv parameter from the application's main() function, or
 *             NULL if the platform's main-equivalent has no argv.
 * \param mainFunction your SDL app's C-style main(). NOT the function you're
 *                     calling this from! Its name doesn't matter; it doesn't
 *                     literally have to be `main`.
 * \param reserved should be NULL (reserved for future use, will probably be
 *                 platform-specific then).
 * \returns the return value from mainFunction: 0 on success, otherwise
 *          failure; SDL_GetError() might have more information on the
 *          failure.
 *
 * \threadsafety Generally this is called once, near startup, from the
 *               process's initial thread.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_RunApp(int argc, char *argv[], SDL_main_func mainFunction, void *reserved);

/**
 * An entry point for SDL's use in SDL_MAIN_USE_CALLBACKS.
 *
 * Generally, you should not call this function directly. This only exists to
 * hand off work into SDL as soon as possible, where it has a lot more control
 * and functionality available, and make the inline code in SDL_main.h as
 * small as possible.
 *
 * Not all platforms use this, it's actual use is hidden in a magic
 * header-only library, and you should not call this directly unless you
 * _really_ know what you're doing.
 *
 * \param argc standard Unix main argc.
 * \param argv standard Unix main argv.
 * \param appinit the application's SDL_AppInit function.
 * \param appiter the application's SDL_AppIterate function.
 * \param appevent the application's SDL_AppEvent function.
 * \param appquit the application's SDL_AppQuit function.
 * \returns standard Unix main return value.
 *
 * \threadsafety It is not safe to call this anywhere except as the only
 *               function call in SDL_main.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC int SDLCALL SDL_EnterAppMainCallbacks(int argc, char *argv[], SDL_AppInit_func appinit, SDL_AppIterate_func appiter, SDL_AppEvent_func appevent, SDL_AppQuit_func appquit);


#if defined(SDL_PLATFORM_WINDOWS)

/**
 * Register a win32 window class for SDL's use.
 *
 * This can be called to set the application window class at startup. It is
 * safe to call this multiple times, as long as every call is eventually
 * paired with a call to SDL_UnregisterApp, but a second registration attempt
 * while a previous registration is still active will be ignored, other than
 * to increment a counter.
 *
 * Most applications do not need to, and should not, call this directly; SDL
 * will call it when initializing the video subsystem.
 *
 * \param name the window class name, in UTF-8 encoding. If NULL, SDL
 *             currently uses "SDL_app" but this isn't guaranteed.
 * \param style the value to use in WNDCLASSEX::style. If `name` is NULL, SDL
 *              currently uses `(CS_BYTEALIGNCLIENT | CS_OWNDC)` regardless of
 *              what is specified here.
 * \param hInst the HINSTANCE to use in WNDCLASSEX::hInstance. If zero, SDL
 *              will use `GetModuleHandle(NULL)` instead.
 * \returns true on success or false on failure; call SDL_GetError() for more
 *          information.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC bool SDLCALL SDL_RegisterApp(const char *name, Uint32 style, void *hInst);

/**
 * Deregister the win32 window class from an SDL_RegisterApp call.
 *
 * This can be called to undo the effects of SDL_RegisterApp.
 *
 * Most applications do not need to, and should not, call this directly; SDL
 * will call it when deinitializing the video subsystem.
 *
 * It is safe to call this multiple times, as long as every call is eventually
 * paired with a prior call to SDL_RegisterApp. The window class will only be
 * deregistered when the registration counter in SDL_RegisterApp decrements to
 * zero through calls to this function.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC void SDLCALL SDL_UnregisterApp(void);

#endif /* defined(SDL_PLATFORM_WINDOWS) */

/**
 * Callback from the application to let the suspend continue.
 *
 * This function is only needed for Xbox GDK support; all other platforms will
 * do nothing and set an "unsupported" error message.
 *
 * \since This function is available since SDL 3.2.0.
 */
extern SDL_DECLSPEC void SDLCALL SDL_GDKSuspendComplete(void);

#ifdef __cplusplus
}
#endif

#include <SDL3/SDL_close_code.h>

#if !defined(SDL_MAIN_HANDLED) && !defined(SDL_MAIN_NOIMPL)
    /* include header-only SDL_main implementations */
    #if defined(SDL_MAIN_USE_CALLBACKS) || defined(SDL_MAIN_NEEDED) || defined(SDL_MAIN_AVAILABLE)
        /* platforms which main (-equivalent) can be implemented in plain C */
        #include <SDL3/SDL_main_impl.h>
    #endif
#endif

#endif /* SDL_main_h_ */
