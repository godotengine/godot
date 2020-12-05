/*
 *  Platform-specific and custom entropy polling functions
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 *  This file is provided under the Apache License 2.0, or the
 *  GNU General Public License v2.0 or later.
 *
 *  **********
 *  Apache License 2.0:
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
 *  **********
 *
 *  **********
 *  GNU General Public License v2.0 or later:
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  **********
 */

#if defined(__linux__)
/* Ensure that syscall() is available even when compiling with -std=c99 */
#define _GNU_SOURCE
#endif

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <string.h>

#if defined(MBEDTLS_ENTROPY_C)

#include "mbedtls/entropy.h"
#include "mbedtls/entropy_poll.h"

#if defined(MBEDTLS_TIMING_C)
#include "mbedtls/timing.h"
#endif
#if defined(MBEDTLS_HAVEGE_C)
#include "mbedtls/havege.h"
#endif
#if defined(MBEDTLS_ENTROPY_NV_SEED)
#include "mbedtls/platform.h"
#endif

#if !defined(MBEDTLS_NO_PLATFORM_ENTROPY)

#if !defined(unix) && !defined(__unix__) && !defined(__unix) && \
    !defined(__APPLE__) && !defined(_WIN32) && !defined(__QNXNTO__) && \
    !defined(__HAIKU__)
#error "Platform entropy sources only work on Unix and Windows, see MBEDTLS_NO_PLATFORM_ENTROPY in config.h"
#endif

#if defined(_WIN32) && !defined(EFIX64) && !defined(EFI32)

#if !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0400
#endif
#include <windows.h>
#include <bcrypt.h>
#if defined(_MSC_VER) && _MSC_VER <= 1600
/* Visual Studio 2010 and earlier issue a warning when both <stdint.h> and
 * <intsafe.h> are included, as they redefine a number of <TYPE>_MAX constants.
 * These constants are guaranteed to be the same, though, so we suppress the
 * warning when including intsafe.h.
 */
#pragma warning( push )
#pragma warning( disable : 4005 )
#endif
#include <intsafe.h>
#if defined(_MSC_VER) && _MSC_VER <= 1600
#pragma warning( pop )
#endif

int mbedtls_platform_entropy_poll( void *data, unsigned char *output, size_t len,
                           size_t *olen )
{
    ULONG len_as_ulong = 0;
    ((void) data);
    *olen = 0;

    /*
     * BCryptGenRandom takes ULONG for size, which is smaller than size_t on
     * 64-bit Windows platforms. Ensure len's value can be safely converted into
     * a ULONG.
     */
    if ( FAILED( SizeTToULong( len, &len_as_ulong ) ) )
    {
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );
    }

    if ( !BCRYPT_SUCCESS( BCryptGenRandom( NULL, output, len_as_ulong, BCRYPT_USE_SYSTEM_PREFERRED_RNG ) ) )
    {
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );
    }

    *olen = len;

    return( 0 );
}
#else /* _WIN32 && !EFIX64 && !EFI32 */

/*
 * Test for Linux getrandom() support.
 * Since there is no wrapper in the libc yet, use the generic syscall wrapper
 * available in GNU libc and compatible libc's (eg uClibc).
 */
#if defined(__linux__) && defined(__GLIBC__)
#include <unistd.h>
#include <sys/syscall.h>
#if defined(SYS_getrandom)
#define HAVE_GETRANDOM
#include <errno.h>

static int getrandom_wrapper( void *buf, size_t buflen, unsigned int flags )
{
    /* MemSan cannot understand that the syscall writes to the buffer */
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
    memset( buf, 0, buflen );
#endif
#endif
    return( syscall( SYS_getrandom, buf, buflen, flags ) );
}
#endif /* SYS_getrandom */
#endif /* __linux__ */

#include <stdio.h>

int mbedtls_platform_entropy_poll( void *data,
                           unsigned char *output, size_t len, size_t *olen )
{
    FILE *file;
    size_t read_len;
    int ret;
    ((void) data);

#if defined(HAVE_GETRANDOM)
    ret = getrandom_wrapper( output, len, 0 );
    if( ret >= 0 )
    {
        *olen = ret;
        return( 0 );
    }
    else if( errno != ENOSYS )
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );
    /* Fall through if the system call isn't known. */
#else
    ((void) ret);
#endif /* HAVE_GETRANDOM */

    *olen = 0;

    file = fopen( "/dev/urandom", "rb" );
    if( file == NULL )
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );

    read_len = fread( output, 1, len, file );
    if( read_len != len )
    {
        fclose( file );
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );
    }

    fclose( file );
    *olen = len;

    return( 0 );
}
#endif /* _WIN32 && !EFIX64 && !EFI32 */
#endif /* !MBEDTLS_NO_PLATFORM_ENTROPY */

#if defined(MBEDTLS_TEST_NULL_ENTROPY)
int mbedtls_null_entropy_poll( void *data,
                    unsigned char *output, size_t len, size_t *olen )
{
    ((void) data);
    ((void) output);
    *olen = 0;

    if( len < sizeof(unsigned char) )
        return( 0 );

    *olen = sizeof(unsigned char);

    return( 0 );
}
#endif

#if defined(MBEDTLS_TIMING_C)
int mbedtls_hardclock_poll( void *data,
                    unsigned char *output, size_t len, size_t *olen )
{
    unsigned long timer = mbedtls_timing_hardclock();
    ((void) data);
    *olen = 0;

    if( len < sizeof(unsigned long) )
        return( 0 );

    memcpy( output, &timer, sizeof(unsigned long) );
    *olen = sizeof(unsigned long);

    return( 0 );
}
#endif /* MBEDTLS_TIMING_C */

#if defined(MBEDTLS_HAVEGE_C)
int mbedtls_havege_poll( void *data,
                 unsigned char *output, size_t len, size_t *olen )
{
    mbedtls_havege_state *hs = (mbedtls_havege_state *) data;
    *olen = 0;

    if( mbedtls_havege_random( hs, output, len ) != 0 )
        return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );

    *olen = len;

    return( 0 );
}
#endif /* MBEDTLS_HAVEGE_C */

#if defined(MBEDTLS_ENTROPY_NV_SEED)
int mbedtls_nv_seed_poll( void *data,
                          unsigned char *output, size_t len, size_t *olen )
{
    unsigned char buf[MBEDTLS_ENTROPY_BLOCK_SIZE];
    size_t use_len = MBEDTLS_ENTROPY_BLOCK_SIZE;
    ((void) data);

    memset( buf, 0, MBEDTLS_ENTROPY_BLOCK_SIZE );

    if( mbedtls_nv_seed_read( buf, MBEDTLS_ENTROPY_BLOCK_SIZE ) < 0 )
      return( MBEDTLS_ERR_ENTROPY_SOURCE_FAILED );

    if( len < use_len )
      use_len = len;

    memcpy( output, buf, use_len );
    *olen = use_len;

    return( 0 );
}
#endif /* MBEDTLS_ENTROPY_NV_SEED */

#endif /* MBEDTLS_ENTROPY_C */
