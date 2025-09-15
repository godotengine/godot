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

/* WIKI CATEGORY: BeginCode */

/**
 * # CategoryBeginCode
 *
 * `SDL_begin_code.h` sets things up for C dynamic library function
 * definitions, static inlined functions, and structures aligned at 4-byte
 * alignment. If you don't like ugly C preprocessor code, don't look at this
 * file. :)
 *
 * SDL's headers use this; applications generally should not include this
 * header directly.
 */

/* This shouldn't be nested -- included it around code only. */
#ifdef SDL_begin_code_h
#error Nested inclusion of SDL_begin_code.h
#endif
#define SDL_begin_code_h

#ifdef SDL_WIKI_DOCUMENTATION_SECTION

/**
 * A macro to tag a symbol as deprecated.
 *
 * A function is marked deprecated by adding this macro to its declaration:
 *
 * ```c
 * extern SDL_DEPRECATED int ThisFunctionWasABadIdea(void);
 * ```
 *
 * Compilers with deprecation support can give a warning when a deprecated
 * function is used. This symbol may be used in SDL's headers, but apps are
 * welcome to use it for their own interfaces as well.
 *
 * SDL, on occasion, might deprecate a function for various reasons. However,
 * SDL never removes symbols before major versions, so deprecated interfaces
 * in SDL3 will remain available until SDL4, where it would be expected an app
 * would have to take steps to migrate anyhow.
 *
 * On compilers without a deprecation mechanism, this is defined to nothing,
 * and using a deprecated function will not generate a warning.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_DEPRECATED __attribute__((deprecated))

/**
 * A macro to tag a symbol as a public API.
 *
 * SDL uses this macro for all its public functions. On some targets, it is
 * used to signal to the compiler that this function needs to be exported from
 * a shared library, but it might have other side effects.
 *
 * This symbol is used in SDL's headers, but apps and other libraries are
 * welcome to use it for their own interfaces as well.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_DECLSPEC __attribute__ ((visibility("default")))

/**
 * A macro to set a function's calling conventions.
 *
 * SDL uses this macro for all its public functions, and any callbacks it
 * defines. This macro guarantees that calling conventions match between SDL
 * and the app, even if the two were built with different compilers or
 * optimization settings.
 *
 * When writing a callback function, it is very important for it to be
 * correctly tagged with SDLCALL, as mismatched calling conventions can cause
 * strange behaviors and can be difficult to diagnose. Plus, on many
 * platforms, SDLCALL is defined to nothing, so compilers won't be able to
 * warn that the tag is missing.
 *
 * This symbol is used in SDL's headers, but apps and other libraries are
 * welcome to use it for their own interfaces as well.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDLCALL __cdecl

/**
 * A macro to request a function be inlined.
 *
 * This is a hint to the compiler to inline a function. The compiler is free
 * to ignore this request. On compilers without inline support, this is
 * defined to nothing.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_INLINE __inline

/**
 * A macro to demand a function be inlined.
 *
 * This is a command to the compiler to inline a function. SDL uses this macro
 * in its public headers for a handful of simple functions. On compilers
 * without forceinline support, this is defined to `static SDL_INLINE`, which
 * is often good enough.
 *
 * This symbol is used in SDL's headers, but apps and other libraries are
 * welcome to use it for their own interfaces as well.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_FORCE_INLINE __forceinline

/**
 * A macro to tag a function as never-returning.
 *
 * This is a hint to the compiler that a function does not return. An example
 * of a function like this is the C runtime's exit() function.
 *
 * This hint can lead to code optimizations, and help analyzers understand
 * code flow better. On compilers without noreturn support, this is defined to
 * nothing.
 *
 * This symbol is used in SDL's headers, but apps and other libraries are
 * welcome to use it for their own interfaces as well.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_NORETURN __attribute__((noreturn))

/**
 * A macro to tag a function as never-returning (for analysis purposes).
 *
 * This is almost identical to SDL_NORETURN, except functions marked with this
 * _can_ actually return. The difference is that this isn't used for code
 * generation, but rather static analyzers use this information to assume
 * truths about program state and available code paths. Specifically, this tag
 * is useful for writing an assertion mechanism. Indeed, SDL_assert uses this
 * tag behind the scenes. Generally, apps that don't understand the specific
 * use-case for this tag should avoid using it directly.
 *
 * On compilers without analyzer_noreturn support, this is defined to nothing.
 *
 * This symbol is used in SDL's headers, but apps and other libraries are
 * welcome to use it for their own interfaces as well.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ANALYZER_NORETURN __attribute__((analyzer_noreturn))


/**
 * A macro to signal that a case statement without a `break` is intentional.
 *
 * C compilers have gotten more aggressive about warning when a switch's
 * `case` block does not end with a `break` or other flow control statement,
 * flowing into the next case's code, as this is a common accident that leads
 * to strange bugs. But sometimes falling through to the next case is the
 * correct and desired behavior. This symbol lets an app communicate this
 * intention to the compiler, so it doesn't generate a warning.
 *
 * It is used like this:
 *
 * ```c
 * switch (x) {
 *     case 1:
 *         DoSomethingOnlyForOne();
 *         SDL_FALLTHROUGH;  // tell the compiler this was intentional.
 *     case 2:
 *         DoSomethingForOneAndTwo();
 *         break;
 * }
 * ```
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_FALLTHROUGH [[fallthrough]]

/**
 * A macro to tag a function's return value as critical.
 *
 * This is a hint to the compiler that a function's return value should not be
 * ignored.
 *
 * If an NODISCARD function's return value is thrown away (the function is
 * called as if it returns `void`), the compiler will issue a warning.
 *
 * While it's generally good practice to check return values for errors, often
 * times legitimate programs do not for good reasons. Be careful about what
 * functions are tagged as NODISCARD. It operates best when used on a function
 * that's failure is surprising and catastrophic; a good example would be a
 * program that checks the return values of all its file write function calls
 * but not the call to close the file, which it assumes incorrectly never
 * fails.
 *
 * Function callers that want to throw away a NODISCARD return value can call
 * the function with a `(void)` cast, which informs the compiler the act is
 * intentional.
 *
 * On compilers without nodiscard support, this is defined to nothing.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_NODISCARD [[nodiscard]]

/**
 * A macro to tag a function as an allocator.
 *
 * This is a hint to the compiler that a function is an allocator, like
 * malloc(), with certain rules. A description of how GCC treats this hint is
 * here:
 *
 * https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-malloc-function-attribute
 *
 * On compilers without allocator tag support, this is defined to nothing.
 *
 * Most apps don't need to, and should not, use this directly.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_MALLOC __declspec(allocator) __desclspec(restrict)

/**
 * A macro to tag a function as returning a certain allocation.
 *
 * This is a hint to the compiler that a function allocates and returns a
 * specific amount of memory based on one of its arguments. For example, the C
 * runtime's malloc() function could use this macro with an argument of 1
 * (first argument to malloc is the size of the allocation).
 *
 * On compilers without alloc_size support, this is defined to nothing.
 *
 * Most apps don't need to, and should not, use this directly.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_ALLOC_SIZE(p) __attribute__((alloc_size(p)))

/**
 * A macro to tag a pointer variable, to help with pointer aliasing.
 *
 * A good explanation of the restrict keyword is here:
 *
 * https://en.wikipedia.org/wiki/Restrict
 *
 * On compilers without restrict support, this is defined to nothing.
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_RESTRICT __restrict__

/**
 * Check if the compiler supports a given builtin functionality.
 *
 * This allows preprocessor checks for things that otherwise might fail to
 * compile.
 *
 * Supported by virtually all clang versions and more-recent GCCs. Use this
 * instead of checking the clang version if possible.
 *
 * On compilers without has_builtin support, this is defined to 0 (always
 * false).
 *
 * \since This macro is available since SDL 3.2.0.
 */
#define SDL_HAS_BUILTIN(x) __has_builtin(x)

/* end of wiki documentation section. */
#endif

#ifndef SDL_HAS_BUILTIN
#ifdef __has_builtin
#define SDL_HAS_BUILTIN(x) __has_builtin(x)
#else
#define SDL_HAS_BUILTIN(x) 0
#endif
#endif

#ifndef SDL_DEPRECATED
#  if defined(__GNUC__) && (__GNUC__ >= 4)  /* technically, this arrived in gcc 3.1, but oh well. */
#    define SDL_DEPRECATED __attribute__((deprecated))
#  elif defined(_MSC_VER)
#    define SDL_DEPRECATED __declspec(deprecated)
#  else
#    define SDL_DEPRECATED
#  endif
#endif

#ifndef SDL_UNUSED
#  ifdef __GNUC__
#    define SDL_UNUSED __attribute__((unused))
#  else
#    define SDL_UNUSED
#  endif
#endif

/* Some compilers use a special export keyword */
#ifndef SDL_DECLSPEC
# if defined(SDL_PLATFORM_WINDOWS)
#  ifdef DLL_EXPORT
#   define SDL_DECLSPEC __declspec(dllexport)
#  else
#   define SDL_DECLSPEC
#  endif
# else
#  if defined(__GNUC__) && __GNUC__ >= 4
#   define SDL_DECLSPEC __attribute__ ((visibility("default")))
#  else
#   define SDL_DECLSPEC
#  endif
# endif
#endif

/* By default SDL uses the C calling convention */
#ifndef SDLCALL
#if defined(SDL_PLATFORM_WINDOWS) && !defined(__GNUC__)
#define SDLCALL __cdecl
#else
#define SDLCALL
#endif
#endif /* SDLCALL */

/* Force structure packing at 4 byte alignment.
   This is necessary if the header is included in code which has structure
   packing set to an alternate value, say for loading structures from disk.
   The packing is reset to the previous value in SDL_close_code.h
 */
#if defined(_MSC_VER) || defined(__MWERKS__) || defined(__BORLANDC__)
#ifdef _MSC_VER
#pragma warning(disable: 4103)
#endif
#ifdef __clang__
#pragma clang diagnostic ignored "-Wpragma-pack"
#endif
#ifdef __BORLANDC__
#pragma nopackwarning
#endif
#ifdef _WIN64
/* Use 8-byte alignment on 64-bit architectures, so pointers are aligned */
#pragma pack(push,8)
#else
#pragma pack(push,4)
#endif
#endif /* Compiler needs structure packing set */

#ifndef SDL_INLINE
#ifdef __GNUC__
#define SDL_INLINE __inline__
#elif defined(_MSC_VER) || defined(__BORLANDC__) || \
      defined(__DMC__) || defined(__SC__) || \
      defined(__WATCOMC__) || defined(__LCC__) || \
      defined(__DECC) || defined(__CC_ARM)
#define SDL_INLINE __inline
#ifndef __inline__
#define __inline__ __inline
#endif
#else
#define SDL_INLINE inline
#ifndef __inline__
#define __inline__ inline
#endif
#endif
#endif /* SDL_INLINE not defined */

#ifndef SDL_FORCE_INLINE
#ifdef _MSC_VER
#define SDL_FORCE_INLINE __forceinline
#elif ( (defined(__GNUC__) && (__GNUC__ >= 4)) || defined(__clang__) )
#define SDL_FORCE_INLINE __attribute__((always_inline)) static __inline__
#else
#define SDL_FORCE_INLINE static SDL_INLINE
#endif
#endif /* SDL_FORCE_INLINE not defined */

#ifndef SDL_NORETURN
#ifdef __GNUC__
#define SDL_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define SDL_NORETURN __declspec(noreturn)
#else
#define SDL_NORETURN
#endif
#endif /* SDL_NORETURN not defined */

#ifdef __clang__
#if __has_feature(attribute_analyzer_noreturn)
#define SDL_ANALYZER_NORETURN __attribute__((analyzer_noreturn))
#endif
#endif

#ifndef SDL_ANALYZER_NORETURN
#define SDL_ANALYZER_NORETURN
#endif

/* Apparently this is needed by several Windows compilers */
#ifndef __MACH__
#ifndef NULL
#ifdef __cplusplus
#define NULL 0
#else
#define NULL ((void *)0)
#endif
#endif /* NULL */
#endif /* ! macOS - breaks precompiled headers */

#ifndef SDL_FALLTHROUGH
#if (defined(__cplusplus) && __cplusplus >= 201703L) || \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202000L)
#define SDL_FALLTHROUGH [[fallthrough]]
#else
#if defined(__has_attribute) && !defined(__SUNPRO_C) && !defined(__SUNPRO_CC)
#define SDL_HAS_FALLTHROUGH __has_attribute(__fallthrough__)
#else
#define SDL_HAS_FALLTHROUGH 0
#endif /* __has_attribute */
#if SDL_HAS_FALLTHROUGH && \
   ((defined(__GNUC__) && __GNUC__ >= 7) || \
    (defined(__clang_major__) && __clang_major__ >= 10))
#define SDL_FALLTHROUGH __attribute__((__fallthrough__))
#else
#define SDL_FALLTHROUGH do {} while (0) /* fallthrough */
#endif /* SDL_HAS_FALLTHROUGH */
#undef SDL_HAS_FALLTHROUGH
#endif /* C++17 or C2x */
#endif /* SDL_FALLTHROUGH not defined */

#ifndef SDL_NODISCARD
#if (defined(__cplusplus) && __cplusplus >= 201703L) || \
    (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#define SDL_NODISCARD [[nodiscard]]
#elif ( (defined(__GNUC__) && (__GNUC__ >= 4)) || defined(__clang__) )
#define SDL_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER) && (_MSC_VER >= 1700)
#define SDL_NODISCARD _Check_return_
#else
#define SDL_NODISCARD
#endif /* C++17 or C23 */
#endif /* SDL_NODISCARD not defined */

#ifndef SDL_MALLOC
#if defined(__GNUC__) && (__GNUC__ >= 3)
#define SDL_MALLOC __attribute__((malloc))
/** FIXME
#elif defined(_MSC_VER)
#define SDL_MALLOC __declspec(allocator) __desclspec(restrict)
**/
#else
#define SDL_MALLOC
#endif
#endif /* SDL_MALLOC not defined */

#ifndef SDL_ALLOC_SIZE
#if (defined(__clang__) && __clang_major__ >= 4) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)))
#define SDL_ALLOC_SIZE(p) __attribute__((alloc_size(p)))
#elif defined(_MSC_VER)
#define SDL_ALLOC_SIZE(p)
#else
#define SDL_ALLOC_SIZE(p)
#endif
#endif /* SDL_ALLOC_SIZE not defined */

#ifndef SDL_ALLOC_SIZE2
#if (defined(__clang__) && __clang_major__ >= 4) || (defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)))
#define SDL_ALLOC_SIZE2(p1, p2) __attribute__((alloc_size(p1, p2)))
#elif defined(_MSC_VER)
#define SDL_ALLOC_SIZE2(p1, p2)
#else
#define SDL_ALLOC_SIZE2(p1, p2)
#endif
#endif /* SDL_ALLOC_SIZE2 not defined */
