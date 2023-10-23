/**
 * \file platform_util.h
 *
 * \brief Common and shared functions used by multiple modules in the Mbed TLS
 *        library.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef MBEDTLS_PLATFORM_UTIL_H
#define MBEDTLS_PLATFORM_UTIL_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>
#if defined(MBEDTLS_HAVE_TIME_DATE)
#include "mbedtls/platform_time.h"
#include <time.h>
#endif /* MBEDTLS_HAVE_TIME_DATE */

#ifdef __cplusplus
extern "C" {
#endif

#if defined(MBEDTLS_CHECK_PARAMS)

#if defined(MBEDTLS_CHECK_PARAMS_ASSERT)
/* Allow the user to define MBEDTLS_PARAM_FAILED to something like assert
 * (which is what our config.h suggests). */
#include <assert.h>
#endif /* MBEDTLS_CHECK_PARAMS_ASSERT */

#if defined(MBEDTLS_PARAM_FAILED)
/** An alternative definition of MBEDTLS_PARAM_FAILED has been set in config.h.
 *
 * This flag can be used to check whether it is safe to assume that
 * MBEDTLS_PARAM_FAILED() will expand to a call to mbedtls_param_failed().
 */
#define MBEDTLS_PARAM_FAILED_ALT

#elif defined(MBEDTLS_CHECK_PARAMS_ASSERT)
#define MBEDTLS_PARAM_FAILED(cond) assert(cond)
#define MBEDTLS_PARAM_FAILED_ALT

#else /* MBEDTLS_PARAM_FAILED */
#define MBEDTLS_PARAM_FAILED(cond) \
    mbedtls_param_failed( #cond, __FILE__, __LINE__)

/**
 * \brief       User supplied callback function for parameter validation failure.
 *              See #MBEDTLS_CHECK_PARAMS for context.
 *
 *              This function will be called unless an alternative treatment
 *              is defined through the #MBEDTLS_PARAM_FAILED macro.
 *
 *              This function can return, and the operation will be aborted, or
 *              alternatively, through use of setjmp()/longjmp() can resume
 *              execution in the application code.
 *
 * \param failure_condition The assertion that didn't hold.
 * \param file  The file where the assertion failed.
 * \param line  The line in the file where the assertion failed.
 */
void mbedtls_param_failed(const char *failure_condition,
                          const char *file,
                          int line);
#endif /* MBEDTLS_PARAM_FAILED */

/* Internal macro meant to be called only from within the library. */
#define MBEDTLS_INTERNAL_VALIDATE_RET(cond, ret)  \
    do {                                            \
        if (!(cond))                               \
        {                                           \
            MBEDTLS_PARAM_FAILED(cond);           \
            return ret;                          \
        }                                           \
    } while (0)

/* Internal macro meant to be called only from within the library. */
#define MBEDTLS_INTERNAL_VALIDATE(cond)           \
    do {                                            \
        if (!(cond))                               \
        {                                           \
            MBEDTLS_PARAM_FAILED(cond);           \
            return;                                 \
        }                                           \
    } while (0)

#else /* MBEDTLS_CHECK_PARAMS */

/* Internal macros meant to be called only from within the library. */
#define MBEDTLS_INTERNAL_VALIDATE_RET(cond, ret)  do { } while (0)
#define MBEDTLS_INTERNAL_VALIDATE(cond)           do { } while (0)

#endif /* MBEDTLS_CHECK_PARAMS */

/* Internal helper macros for deprecating API constants. */
#if !defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
/* Deliberately don't (yet) export MBEDTLS_DEPRECATED here
 * to avoid conflict with other headers which define and use
 * it, too. We might want to move all these definitions here at
 * some point for uniformity. */
#define MBEDTLS_DEPRECATED __attribute__((deprecated))
MBEDTLS_DEPRECATED typedef char const *mbedtls_deprecated_string_constant_t;
#define MBEDTLS_DEPRECATED_STRING_CONSTANT(VAL)       \
    ((mbedtls_deprecated_string_constant_t) (VAL))
MBEDTLS_DEPRECATED typedef int mbedtls_deprecated_numeric_constant_t;
#define MBEDTLS_DEPRECATED_NUMERIC_CONSTANT(VAL)       \
    ((mbedtls_deprecated_numeric_constant_t) (VAL))
#undef MBEDTLS_DEPRECATED
#else /* MBEDTLS_DEPRECATED_WARNING */
#define MBEDTLS_DEPRECATED_STRING_CONSTANT(VAL) VAL
#define MBEDTLS_DEPRECATED_NUMERIC_CONSTANT(VAL) VAL
#endif /* MBEDTLS_DEPRECATED_WARNING */
#endif /* MBEDTLS_DEPRECATED_REMOVED */

/* Implementation of the check-return facility.
 * See the user documentation in config.h.
 *
 * Do not use this macro directly to annotate function: instead,
 * use one of MBEDTLS_CHECK_RETURN_CRITICAL or MBEDTLS_CHECK_RETURN_TYPICAL
 * depending on how important it is to check the return value.
 */
#if !defined(MBEDTLS_CHECK_RETURN)
#if defined(__GNUC__)
#define MBEDTLS_CHECK_RETURN __attribute__((__warn_unused_result__))
#elif defined(_MSC_VER) && _MSC_VER >= 1700
#include <sal.h>
#define MBEDTLS_CHECK_RETURN _Check_return_
#else
#define MBEDTLS_CHECK_RETURN
#endif
#endif

/** Critical-failure function
 *
 * This macro appearing at the beginning of the declaration of a function
 * indicates that its return value should be checked in all applications.
 * Omitting the check is very likely to indicate a bug in the application
 * and will result in a compile-time warning if #MBEDTLS_CHECK_RETURN
 * is implemented for the compiler in use.
 *
 * \note  The use of this macro is a work in progress.
 *        This macro may be added to more functions in the future.
 *        Such an extension is not considered an API break, provided that
 *        there are near-unavoidable circumstances under which the function
 *        can fail. For example, signature/MAC/AEAD verification functions,
 *        and functions that require a random generator, are considered
 *        return-check-critical.
 */
#define MBEDTLS_CHECK_RETURN_CRITICAL MBEDTLS_CHECK_RETURN

/** Ordinary-failure function
 *
 * This macro appearing at the beginning of the declaration of a function
 * indicates that its return value should be generally be checked in portable
 * applications. Omitting the check will result in a compile-time warning if
 * #MBEDTLS_CHECK_RETURN is implemented for the compiler in use and
 * #MBEDTLS_CHECK_RETURN_WARNING is enabled in the compile-time configuration.
 *
 * You can use #MBEDTLS_IGNORE_RETURN to explicitly ignore the return value
 * of a function that is annotated with #MBEDTLS_CHECK_RETURN.
 *
 * \note  The use of this macro is a work in progress.
 *        This macro will be added to more functions in the future.
 *        Eventually this should appear before most functions returning
 *        an error code (as \c int in the \c mbedtls_xxx API or
 *        as ::psa_status_t in the \c psa_xxx API).
 */
#if defined(MBEDTLS_CHECK_RETURN_WARNING)
#define MBEDTLS_CHECK_RETURN_TYPICAL MBEDTLS_CHECK_RETURN
#else
#define MBEDTLS_CHECK_RETURN_TYPICAL
#endif

/** Benign-failure function
 *
 * This macro appearing at the beginning of the declaration of a function
 * indicates that it is rarely useful to check its return value.
 *
 * This macro has an empty expansion. It exists for documentation purposes:
 * a #MBEDTLS_CHECK_RETURN_OPTIONAL annotation indicates that the function
 * has been analyzed for return-check usefulness, whereas the lack of
 * an annotation indicates that the function has not been analyzed and its
 * return-check usefulness is unknown.
 */
#define MBEDTLS_CHECK_RETURN_OPTIONAL

/** \def MBEDTLS_IGNORE_RETURN
 *
 * Call this macro with one argument, a function call, to suppress a warning
 * from #MBEDTLS_CHECK_RETURN due to that function call.
 */
#if !defined(MBEDTLS_IGNORE_RETURN)
/* GCC doesn't silence the warning with just (void)(result).
 * (void)!(result) is known to work up at least up to GCC 10, as well
 * as with Clang and MSVC.
 *
 * https://gcc.gnu.org/onlinedocs/gcc-3.4.6/gcc/Non_002dbugs.html
 * https://stackoverflow.com/questions/40576003/ignoring-warning-wunused-result
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=66425#c34
 */
#define MBEDTLS_IGNORE_RETURN(result) ((void) !(result))
#endif

/* If the following macro is defined, the library is being built by the test
 * framework, and the framework is going to provide a replacement
 * mbedtls_platform_zeroize() using a preprocessor macro, so the function
 * declaration should be omitted.  */
#if !defined(MBEDTLS_TEST_DEFINES_ZEROIZE) //no-check-names
/**
 * \brief       Securely zeroize a buffer
 *
 *              The function is meant to wipe the data contained in a buffer so
 *              that it can no longer be recovered even if the program memory
 *              is later compromised. Call this function on sensitive data
 *              stored on the stack before returning from a function, and on
 *              sensitive data stored on the heap before freeing the heap
 *              object.
 *
 *              It is extremely difficult to guarantee that calls to
 *              mbedtls_platform_zeroize() are not removed by aggressive
 *              compiler optimizations in a portable way. For this reason, Mbed
 *              TLS provides the configuration option
 *              MBEDTLS_PLATFORM_ZEROIZE_ALT, which allows users to configure
 *              mbedtls_platform_zeroize() to use a suitable implementation for
 *              their platform and needs
 *
 * \param buf   Buffer to be zeroized
 * \param len   Length of the buffer in bytes
 *
 */
void mbedtls_platform_zeroize(void *buf, size_t len);
#endif

#if defined(MBEDTLS_HAVE_TIME_DATE)
/**
 * \brief      Platform-specific implementation of gmtime_r()
 *
 *             The function is a thread-safe abstraction that behaves
 *             similarly to the gmtime_r() function from Unix/POSIX.
 *
 *             Mbed TLS will try to identify the underlying platform and
 *             make use of an appropriate underlying implementation (e.g.
 *             gmtime_r() for POSIX and gmtime_s() for Windows). If this is
 *             not possible, then gmtime() will be used. In this case, calls
 *             from the library to gmtime() will be guarded by the mutex
 *             mbedtls_threading_gmtime_mutex if MBEDTLS_THREADING_C is
 *             enabled. It is recommended that calls from outside the library
 *             are also guarded by this mutex.
 *
 *             If MBEDTLS_PLATFORM_GMTIME_R_ALT is defined, then Mbed TLS will
 *             unconditionally use the alternative implementation for
 *             mbedtls_platform_gmtime_r() supplied by the user at compile time.
 *
 * \param tt     Pointer to an object containing time (in seconds) since the
 *               epoch to be converted
 * \param tm_buf Pointer to an object where the results will be stored
 *
 * \return      Pointer to an object of type struct tm on success, otherwise
 *              NULL
 */
struct tm *mbedtls_platform_gmtime_r(const mbedtls_time_t *tt,
                                     struct tm *tm_buf);
#endif /* MBEDTLS_HAVE_TIME_DATE */

#ifdef __cplusplus
}
#endif

#endif /* MBEDTLS_PLATFORM_UTIL_H */
