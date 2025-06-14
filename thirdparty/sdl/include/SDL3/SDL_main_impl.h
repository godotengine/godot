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

/* WIKI CATEGORY: Main */

#ifndef SDL_main_impl_h_
#define SDL_main_impl_h_

#ifndef SDL_main_h_
#error "This header should not be included directly, but only via SDL_main.h!"
#endif

/* if someone wants to include SDL_main.h but doesn't want the main handing magic,
   (maybe to call SDL_RegisterApp()) they can #define SDL_MAIN_HANDLED first.
   SDL_MAIN_NOIMPL is for SDL-internal usage (only affects implementation,
   not definition of SDL_MAIN_AVAILABLE etc in SDL_main.h) and if the user wants
   to have the SDL_main implementation (from this header) in another source file
   than their main() function, for example if SDL_main requires C++
   and main() is implemented in plain C */
#if !defined(SDL_MAIN_HANDLED) && !defined(SDL_MAIN_NOIMPL)

    /* the implementations below must be able to use the implement real main(), nothing renamed
       (the user's main() will be renamed to SDL_main so it can be called from here) */
    #ifdef main
        #undef main
    #endif

    #ifdef SDL_MAIN_USE_CALLBACKS

        #if 0
            /* currently there are no platforms that _need_ a magic entry point here
               for callbacks, but if one shows up, implement it here. */

        #else /* use a standard SDL_main, which the app SHOULD NOT ALSO SUPPLY. */

            /* this define makes the normal SDL_main entry point stuff work...we just provide SDL_main() instead of the app. */
            #define SDL_MAIN_CALLBACK_STANDARD 1

            int SDL_main(int argc, char **argv)
            {
                return SDL_EnterAppMainCallbacks(argc, argv, SDL_AppInit, SDL_AppIterate, SDL_AppEvent, SDL_AppQuit);
            }

        #endif  /* platform-specific tests */

    #endif  /* SDL_MAIN_USE_CALLBACKS */


    /* set up the usual SDL_main stuff if we're not using callbacks or if we are but need the normal entry point,
       unless the real entry point needs to be somewhere else entirely, like Android where it's in Java code */
    #if (!defined(SDL_MAIN_USE_CALLBACKS) || defined(SDL_MAIN_CALLBACK_STANDARD)) && !defined(SDL_MAIN_EXPORTED)

        #if defined(SDL_PLATFORM_PRIVATE_MAIN)
            /* Private platforms may have their own ideas about entry points. */
            #include "SDL_main_impl_private.h"

        #elif defined(SDL_PLATFORM_WINDOWS)

            /* these defines/typedefs are needed for the WinMain() definition */
            #ifndef WINAPI
                #define WINAPI __stdcall
            #endif

            typedef struct HINSTANCE__ * HINSTANCE;
            typedef char *LPSTR;
            typedef wchar_t *PWSTR;

            /* The VC++ compiler needs main/wmain defined, but not for GDK */
            #if defined(_MSC_VER) && !defined(SDL_PLATFORM_GDK)

                /* This is where execution begins [console apps] */
                #if defined(UNICODE) && UNICODE
                    int wmain(int argc, wchar_t *wargv[], wchar_t *wenvp)
                    {
                        (void)argc;
                        (void)wargv;
                        (void)wenvp;
                        return SDL_RunApp(0, NULL, SDL_main, NULL);
                    }
                #else /* ANSI */
                    int main(int argc, char *argv[])
                    {
                        (void)argc;
                        (void)argv;
                        return SDL_RunApp(0, NULL, SDL_main, NULL);
                    }
                #endif /* UNICODE */

            #endif /* _MSC_VER && ! SDL_PLATFORM_GDK */

            /* This is where execution begins [windowed apps and GDK] */

            #ifdef __cplusplus
            extern "C" {
            #endif

            #if defined(UNICODE) && UNICODE
            int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE hPrev, PWSTR szCmdLine, int sw)
            #else /* ANSI */
            int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR szCmdLine, int sw)
            #endif
            {
                (void)hInst;
                (void)hPrev;
                (void)szCmdLine;
                (void)sw;
                return SDL_RunApp(0, NULL, SDL_main, NULL);
            }

            #ifdef __cplusplus
            } /* extern "C" */
            #endif

            /* end of SDL_PLATFORM_WINDOWS impls */

        #else /* platforms that use a standard main() and just call SDL_RunApp(), like iOS and 3DS */
            int main(int argc, char *argv[])
            {
                return SDL_RunApp(argc, argv, SDL_main, NULL);
            }

            /* end of impls for standard-conforming platforms */

        #endif /* SDL_PLATFORM_WIN32 etc */

    #endif /* !defined(SDL_MAIN_USE_CALLBACKS) || defined(SDL_MAIN_CALLBACK_STANDARD) */

    /* rename users main() function to SDL_main() so it can be called from the wrappers above */
    #define main    SDL_main

#endif /* SDL_MAIN_HANDLED */

#endif /* SDL_main_impl_h_ */
