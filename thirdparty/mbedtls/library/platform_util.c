/*
 * Common and shared functions used by multiple modules in the Mbed TLS
 * library.
 *
 *  Copyright (C) 2018, Arm Limited, All Rights Reserved
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
 *
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */

/*
 * Ensure gmtime_r is available even with -std=c99; must be defined before
 * config.h, which pulls in glibc's features.h. Harmless on other platforms.
 */
#if !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200112L
#endif

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "mbedtls/platform_util.h"
#include "mbedtls/platform.h"
#include "mbedtls/threading.h"

#include <stddef.h>
#include <string.h>

#if !defined(MBEDTLS_PLATFORM_ZEROIZE_ALT)
/*
 * This implementation should never be optimized out by the compiler
 *
 * This implementation for mbedtls_platform_zeroize() was inspired from Colin
 * Percival's blog article at:
 *
 * http://www.daemonology.net/blog/2014-09-04-how-to-zero-a-buffer.html
 *
 * It uses a volatile function pointer to the standard memset(). Because the
 * pointer is volatile the compiler expects it to change at
 * any time and will not optimize out the call that could potentially perform
 * other operations on the input buffer instead of just setting it to 0.
 * Nevertheless, as pointed out by davidtgoldblatt on Hacker News
 * (refer to http://www.daemonology.net/blog/2014-09-05-erratum.html for
 * details), optimizations of the following form are still possible:
 *
 * if( memset_func != memset )
 *     memset_func( buf, 0, len );
 *
 * Note that it is extremely difficult to guarantee that
 * mbedtls_platform_zeroize() will not be optimized out by aggressive compilers
 * in a portable way. For this reason, Mbed TLS also provides the configuration
 * option MBEDTLS_PLATFORM_ZEROIZE_ALT, which allows users to configure
 * mbedtls_platform_zeroize() to use a suitable implementation for their
 * platform and needs.
 */
static void * (* const volatile memset_func)( void *, int, size_t ) = memset;

void mbedtls_platform_zeroize( void *buf, size_t len )
{
    memset_func( buf, 0, len );
}
#endif /* MBEDTLS_PLATFORM_ZEROIZE_ALT */

#if defined(MBEDTLS_HAVE_TIME_DATE) && !defined(MBEDTLS_PLATFORM_GMTIME_R_ALT)
#include <time.h>
#if !defined(_WIN32) && (defined(unix) || \
    defined(__unix) || defined(__unix__) || (defined(__APPLE__) && \
    defined(__MACH__)))
#include <unistd.h>
#endif /* !_WIN32 && (unix || __unix || __unix__ ||
        * (__APPLE__ && __MACH__)) */

#if !( ( defined(_POSIX_VERSION) && _POSIX_VERSION >= 200809L ) ||     \
       ( defined(_POSIX_THREAD_SAFE_FUNCTIONS ) &&                     \
         _POSIX_THREAD_SAFE_FUNCTIONS >= 20112L ) )
/*
 * This is a convenience shorthand macro to avoid checking the long
 * preprocessor conditions above. Ideally, we could expose this macro in
 * platform_util.h and simply use it in platform_util.c, threading.c and
 * threading.h. However, this macro is not part of the Mbed TLS public API, so
 * we keep it private by only defining it in this file
 */
#if ! ( defined(_WIN32) && !defined(EFIX64) && !defined(EFI32) )
#define PLATFORM_UTIL_USE_GMTIME
#endif /* ! ( defined(_WIN32) && !defined(EFIX64) && !defined(EFI32) ) */

#endif /* !( ( defined(_POSIX_VERSION) && _POSIX_VERSION >= 200809L ) ||     \
             ( defined(_POSIX_THREAD_SAFE_FUNCTIONS ) &&                     \
                _POSIX_THREAD_SAFE_FUNCTIONS >= 20112L ) ) */

struct tm *mbedtls_platform_gmtime_r( const mbedtls_time_t *tt,
                                      struct tm *tm_buf )
{
#if defined(_WIN32) && !defined(EFIX64) && !defined(EFI32)
    return( ( gmtime_s( tm_buf, tt ) == 0 ) ? tm_buf : NULL );
#elif !defined(PLATFORM_UTIL_USE_GMTIME)
    return( gmtime_r( tt, tm_buf ) );
#else
    struct tm *lt;

#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_lock( &mbedtls_threading_gmtime_mutex ) != 0 )
        return( NULL );
#endif /* MBEDTLS_THREADING_C */

    lt = gmtime( tt );

    if( lt != NULL )
    {
        memcpy( tm_buf, lt, sizeof( struct tm ) );
    }

#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_unlock( &mbedtls_threading_gmtime_mutex ) != 0 )
        return( NULL );
#endif /* MBEDTLS_THREADING_C */

    return( ( lt == NULL ) ? NULL : tm_buf );
#endif /* _WIN32 && !EFIX64 && !EFI32 */
}
#endif /* MBEDTLS_HAVE_TIME_DATE && MBEDTLS_PLATFORM_GMTIME_R_ALT */
