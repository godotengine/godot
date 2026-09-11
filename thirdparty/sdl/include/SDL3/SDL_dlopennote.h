/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2026 Sam Lantinga <slouken@libsdl.org>

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

/* WIKI CATEGORY: DlopenNotes */

/**
 * # CategoryDlopenNotes
 *
 * This header allows you to annotate your code so external tools know about
 * dynamic shared library dependencies.
 *
 * If you determine that your toolchain doesn't support dlopen notes, you can
 * disable this feature by defining `SDL_DISABLE_DLOPEN_NOTES`. You can use
 * this CMake snippet to check for support:
 *
 * ```cmake
 * include(CheckCSourceCompiles)
 * find_package(SDL3 REQUIRED CONFIG COMPONENTS Headers)
 * list(APPEND CMAKE_REQUIRED_LIBRARIES SDL3::Headers)
 * check_c_source_compiles([==[
 *   #include <SDL3/SDL_dlopennote.h>
 *   SDL_ELF_NOTE_DLOPEN("sdl-video",
 *     "Support for video through SDL",
 *     SDL_ELF_NOTE_DLOPEN_PRIORITY_SUGGESTED,
 *     "libSDL-1.2.so.0", "libSDL-2.0.so.0", "libSDL3.so.0"
 *   )
 *   int main(int argc, char *argv[]) {
 *     return argc + argv[0][1];
 *   }
 * ]==] COMPILER_SUPPORTS_SDL_ELF_NOTE_DLOPEN)
 * if(NOT COMPILER_SUPPORTS_SDL_ELF_NOTE_DLOPEN)
 *   add_compile_definitions(-DSDL_DISABLE_DLOPEN_NOTE)
 * endif()
 * ```
 */

#ifndef SDL_dlopennote_h
#define SDL_dlopennote_h

/**
 * Use this macro with SDL_ELF_NOTE_DLOPEN() to note that a dynamic shared
 * library dependency is optional.
 *
 * Optional functionality uses the dependency, the binary will work and the
 * dependency is only needed for full-featured installations.
 *
 * \since This macro is available since SDL 3.4.0.
 *
 * \sa SDL_ELF_NOTE_DLOPEN
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_RECOMMENDED
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_REQUIRED
 */
#define SDL_ELF_NOTE_DLOPEN_PRIORITY_SUGGESTED   "suggested"

/**
 * Use this macro with SDL_ELF_NOTE_DLOPEN() to note that a dynamic shared
 * library dependency is recommended.
 *
 * Important functionality needs the dependency, the binary will work but in
 * most cases the dependency should be provided.
 *
 * \since This macro is available since SDL 3.4.0.
 *
 * \sa SDL_ELF_NOTE_DLOPEN
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_SUGGESTED
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_REQUIRED
 */
#define SDL_ELF_NOTE_DLOPEN_PRIORITY_RECOMMENDED "recommended"

/**
 * Use this macro with SDL_ELF_NOTE_DLOPEN() to note that a dynamic shared
 * library dependency is required.
 *
 * Core functionality needs the dependency, the binary will not work if it
 * cannot be found.
 *
 * \since This macro is available since SDL 3.4.0.
 *
 * \sa SDL_ELF_NOTE_DLOPEN
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_SUGGESTED
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_RECOMMENDED
 */
#define SDL_ELF_NOTE_DLOPEN_PRIORITY_REQUIRED    "required"


#if !defined(SDL_PLATFORM_UNIX) || defined(SDL_PLATFORM_ANDROID)
/* The dlopen note functionality isn't used on this platform */
#ifndef SDL_DISABLE_DLOPEN_NOTES
#define SDL_DISABLE_DLOPEN_NOTES
#endif
#elif defined(__GNUC__) && (__GNUC__ < 3 || (__GNUC__ == 3 && __GNUC_MINOR__ < 1))
/* gcc < 3.1 too old */
#ifndef SDL_DISABLE_DLOPEN_NOTES
#define SDL_DISABLE_DLOPEN_NOTES
#endif
#endif /* SDL_PLATFORM_UNIX || SDL_PLATFORM_ANDROID */

#if defined(__ELF__) && !defined(SDL_DISABLE_DLOPEN_NOTES)

#include <SDL3/SDL_stdinc.h>

#define SDL_ELF_NOTE_DLOPEN_VENDOR "FDO"
#define SDL_ELF_NOTE_DLOPEN_TYPE 0x407c0c0aU

#define SDL_ELF_NOTE_INTERNAL2(json, variable_name)                 \
    __attribute__((aligned(4), used, section(".note.dlopen")))      \
    static const struct {                                           \
        struct {                                                    \
            Uint32 n_namesz;                                        \
            Uint32 n_descsz;                                        \
            Uint32 n_type;                                          \
        } nhdr;                                                     \
        char name[4];                                               \
        __attribute__((aligned(4))) char dlopen_json[sizeof(json)]; \
    } variable_name = {                                             \
        {                                                           \
             sizeof(SDL_ELF_NOTE_DLOPEN_VENDOR),                    \
             sizeof(json),                                          \
             SDL_ELF_NOTE_DLOPEN_TYPE                               \
        },                                                          \
        SDL_ELF_NOTE_DLOPEN_VENDOR,                                 \
        json                                                        \
    }

#define SDL_ELF_NOTE_INTERNAL(json, variable_name) \
    SDL_ELF_NOTE_INTERNAL2(json, variable_name)

#define SDL_DLNOTE_JSON_ARRAY1(N1) "[\"" N1 "\"]"
#define SDL_DLNOTE_JSON_ARRAY2(N1,N2) "[\"" N1 "\",\"" N2 "\"]"
#define SDL_DLNOTE_JSON_ARRAY3(N1,N2,N3) "[\"" N1 "\",\"" N2 "\",\"" N3 "\"]"
#define SDL_DLNOTE_JSON_ARRAY4(N1,N2,N3,N4) "[\"" N1 "\",\"" N2 "\",\"" N3 "\",\"" N4 "\"]"
#define SDL_DLNOTE_JSON_ARRAY5(N1,N2,N3,N4,N5) "[\"" N1 "\",\"" N2 "\",\"" N3 "\",\"" N4 "\",\"" N5 "\"]"
#define SDL_DLNOTE_JSON_ARRAY6(N1,N2,N3,N4,N5,N6) "[\"" N1 "\",\"" N2 "\",\"" N3 "\",\"" N4 "\",\"" N5 "\",\"" N6 "\"]"
#define SDL_DLNOTE_JSON_ARRAY7(N1,N2,N3,N4,N5,N6,N7) "[\"" N1 "\",\"" N2 "\",\"" N3 "\",\"" N4 "\",\"" N5 "\",\"" N6 "\",\"" N7 "\"]"
#define SDL_DLNOTE_JSON_ARRAY8(N1,N2,N3,N4,N5,N6,N7,N8) "[\"" N1 "\",\"" N2 "\",\"" N3 "\",\"" N4 "\",\"" N5 "\",\"" N6 "\",\"" N7 "\",\"" N8 "\"]"
#define SDL_DLNOTE_JSON_ARRAY_GET(N1,N2,N3,N4,N5,N6,N7,N8,NAME,...) NAME
#define SDL_DLNOTE_JSON_ARRAY(...) \
    SDL_DLNOTE_JSON_ARRAY_GET(     \
         __VA_ARGS__,           \
         SDL_DLNOTE_JSON_ARRAY8,   \
         SDL_DLNOTE_JSON_ARRAY7,   \
         SDL_DLNOTE_JSON_ARRAY6,   \
         SDL_DLNOTE_JSON_ARRAY5,   \
         SDL_DLNOTE_JSON_ARRAY4,   \
         SDL_DLNOTE_JSON_ARRAY3,   \
         SDL_DLNOTE_JSON_ARRAY2,   \
         SDL_DLNOTE_JSON_ARRAY1    \
    )(__VA_ARGS__)

/* Create "unique" variable name using __LINE__,
 * so creating multiple elf notes on the same line is not supported
 */
#define SDL_DLNOTE_JOIN2(A,B) A##B
#define SDL_DLNOTE_JOIN(A,B) SDL_DLNOTE_JOIN2(A,B)
#define SDL_DLNOTE_UNIQUE_NAME SDL_DLNOTE_JOIN(s_SDL_dlopen_note_, __LINE__)

/**
 * Add a note that your application has dynamic shared library dependencies.
 *
 * You can do this by adding the following to the global scope:
 *
 * ```c
 * SDL_ELF_NOTE_DLOPEN(
 *     "png",
 *     "Support for loading PNG images using libpng (required for APNG)",
 *     SDL_ELF_NOTE_DLOPEN_PRIORITY_RECOMMENDED,
 *     "libpng12.so.0"
 * )
 * ```
 *
 * A trailing semicolon is not needed.
 *
 * Or if you support multiple versions of a library, you can list them:
 *
 * ```c
 * // Our app supports SDL1, SDL2, and SDL3 by dynamically loading them
 * SDL_ELF_NOTE_DLOPEN(
 *     "SDL",
 *     "Create windows through SDL video backend",
 *     SDL_ELF_NOTE_DLOPEN_PRIORITY_REQUIRED
 *     "libSDL-1.2.so.0", "libSDL2-2.0.so.0", "libSDL3.so.0"
 * )
 * ```
 *
 * This macro is not available for compilers that do not support variadic
 * macro's.
 *
 * \since This macro is available since SDL 3.4.0.
 *
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_SUGGESTED
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_RECOMMENDED
 * \sa SDL_ELF_NOTE_DLOPEN_PRIORITY_REQUIRED
 */
#define SDL_ELF_NOTE_DLOPEN(feature, description, priority, ...) \
    SDL_ELF_NOTE_INTERNAL(                                       \
        "[{\"feature\":\"" feature                               \
        "\",\"description\":\"" description                      \
        "\",\"priority\":\"" priority                            \
        "\",\"soname\":" SDL_DLNOTE_JSON_ARRAY(__VA_ARGS__) "}]",   \
        SDL_DLNOTE_UNIQUE_NAME);

#elif defined(__GNUC__) && __GNUC__ < 3

#define SDL_ELF_NOTE_DLOPEN(args...)

#elif defined(_MSC_VER) && _MSC_VER < 1400

/* Variadic macros are not supported */

#else

#define SDL_ELF_NOTE_DLOPEN(...)

#endif

#endif /* SDL_dlopennote_h */
