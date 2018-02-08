/*
 *  Debugging routines
 *
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
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
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_DEBUG_C)

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc      calloc
#define mbedtls_free        free
#define mbedtls_time_t      time_t
#define mbedtls_snprintf    snprintf
#endif

#include "mbedtls/debug.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#if ( defined(__ARMCC_VERSION) || defined(_MSC_VER) ) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

#define DEBUG_BUF_SIZE      512

static int debug_threshold = 0;

void mbedtls_debug_set_threshold( int threshold )
{
    debug_threshold = threshold;
}

/*
 * All calls to f_dbg must be made via this function
 */
static inline void debug_send_line( const mbedtls_ssl_context *ssl, int level,
                                    const char *file, int line,
                                    const char *str )
{
    /*
     * If in a threaded environment, we need a thread identifier.
     * Since there is no portable way to get one, use the address of the ssl
     * context instead, as it shouldn't be shared between threads.
     */
#if defined(MBEDTLS_THREADING_C)
    char idstr[20 + DEBUG_BUF_SIZE]; /* 0x + 16 nibbles + ': ' */
    mbedtls_snprintf( idstr, sizeof( idstr ), "%p: %s", (void*)ssl, str );
    ssl->conf->f_dbg( ssl->conf->p_dbg, level, file, line, idstr );
#else
    ssl->conf->f_dbg( ssl->conf->p_dbg, level, file, line, str );
#endif
}

void mbedtls_debug_print_msg( const mbedtls_ssl_context *ssl, int level,
                              const char *file, int line,
                              const char *format, ... )
{
    va_list argp;
    char str[DEBUG_BUF_SIZE];
    int ret;

    if( NULL == ssl || NULL == ssl->conf || NULL == ssl->conf->f_dbg || level > debug_threshold )
        return;

    va_start( argp, format );
#if defined(_WIN32)
#if defined(_TRUNCATE)
    ret = _vsnprintf_s( str, DEBUG_BUF_SIZE, _TRUNCATE, format, argp );
#else
    ret = _vsnprintf( str, DEBUG_BUF_SIZE, format, argp );
    if( ret < 0 || (size_t) ret == DEBUG_BUF_SIZE )
    {
        str[DEBUG_BUF_SIZE-1] = '\0';
        ret = -1;
    }
#endif
#else
    ret = vsnprintf( str, DEBUG_BUF_SIZE, format, argp );
#endif
    va_end( argp );

    if( ret >= 0 && ret < DEBUG_BUF_SIZE - 1 )
    {
        str[ret]     = '\n';
        str[ret + 1] = '\0';
    }

    debug_send_line( ssl, level, file, line, str );
}

void mbedtls_debug_print_ret( const mbedtls_ssl_context *ssl, int level,
                      const char *file, int line,
                      const char *text, int ret )
{
    char str[DEBUG_BUF_SIZE];

    if( ssl->conf == NULL || ssl->conf->f_dbg == NULL || level > debug_threshold )
        return;

    /*
     * With non-blocking I/O and examples that just retry immediately,
     * the logs would be quickly flooded with WANT_READ, so ignore that.
     * Don't ignore WANT_WRITE however, since is is usually rare.
     */
    if( ret == MBEDTLS_ERR_SSL_WANT_READ )
        return;

    mbedtls_snprintf( str, sizeof( str ), "%s() returned %d (-0x%04x)\n",
              text, ret, -ret );

    debug_send_line( ssl, level, file, line, str );
}

void mbedtls_debug_print_buf( const mbedtls_ssl_context *ssl, int level,
                      const char *file, int line, const char *text,
                      const unsigned char *buf, size_t len )
{
    char str[DEBUG_BUF_SIZE];
    char txt[17];
    size_t i, idx = 0;

    if( ssl->conf == NULL || ssl->conf->f_dbg == NULL || level > debug_threshold )
        return;

    mbedtls_snprintf( str + idx, sizeof( str ) - idx, "dumping '%s' (%u bytes)\n",
              text, (unsigned int) len );

    debug_send_line( ssl, level, file, line, str );

    idx = 0;
    memset( txt, 0, sizeof( txt ) );
    for( i = 0; i < len; i++ )
    {
        if( i >= 4096 )
            break;

        if( i % 16 == 0 )
        {
            if( i > 0 )
            {
                mbedtls_snprintf( str + idx, sizeof( str ) - idx, "  %s\n", txt );
                debug_send_line( ssl, level, file, line, str );

                idx = 0;
                memset( txt, 0, sizeof( txt ) );
            }

            idx += mbedtls_snprintf( str + idx, sizeof( str ) - idx, "%04x: ",
                             (unsigned int) i );

        }

        idx += mbedtls_snprintf( str + idx, sizeof( str ) - idx, " %02x",
                         (unsigned int) buf[i] );
        txt[i % 16] = ( buf[i] > 31 && buf[i] < 127 ) ? buf[i] : '.' ;
    }

    if( len > 0 )
    {
        for( /* i = i */; i % 16 != 0; i++ )
            idx += mbedtls_snprintf( str + idx, sizeof( str ) - idx, "   " );

        mbedtls_snprintf( str + idx, sizeof( str ) - idx, "  %s\n", txt );
        debug_send_line( ssl, level, file, line, str );
    }
}

#if defined(MBEDTLS_ECP_C)
void mbedtls_debug_print_ecp( const mbedtls_ssl_context *ssl, int level,
                      const char *file, int line,
                      const char *text, const mbedtls_ecp_point *X )
{
    char str[DEBUG_BUF_SIZE];

    if( ssl->conf == NULL || ssl->conf->f_dbg == NULL || level > debug_threshold )
        return;

    mbedtls_snprintf( str, sizeof( str ), "%s(X)", text );
    mbedtls_debug_print_mpi( ssl, level, file, line, str, &X->X );

    mbedtls_snprintf( str, sizeof( str ), "%s(Y)", text );
    mbedtls_debug_print_mpi( ssl, level, file, line, str, &X->Y );
}
#endif /* MBEDTLS_ECP_C */

#if defined(MBEDTLS_BIGNUM_C)
void mbedtls_debug_print_mpi( const mbedtls_ssl_context *ssl, int level,
                      const char *file, int line,
                      const char *text, const mbedtls_mpi *X )
{
    char str[DEBUG_BUF_SIZE];
    int j, k, zeros = 1;
    size_t i, n, idx = 0;

    if( ssl->conf == NULL || ssl->conf->f_dbg == NULL || X == NULL || level > debug_threshold )
        return;

    for( n = X->n - 1; n > 0; n-- )
        if( X->p[n] != 0 )
            break;

    for( j = ( sizeof(mbedtls_mpi_uint) << 3 ) - 1; j >= 0; j-- )
        if( ( ( X->p[n] >> j ) & 1 ) != 0 )
            break;

    mbedtls_snprintf( str + idx, sizeof( str ) - idx, "value of '%s' (%d bits) is:\n",
              text, (int) ( ( n * ( sizeof(mbedtls_mpi_uint) << 3 ) ) + j + 1 ) );

    debug_send_line( ssl, level, file, line, str );

    idx = 0;
    for( i = n + 1, j = 0; i > 0; i-- )
    {
        if( zeros && X->p[i - 1] == 0 )
            continue;

        for( k = sizeof( mbedtls_mpi_uint ) - 1; k >= 0; k-- )
        {
            if( zeros && ( ( X->p[i - 1] >> ( k << 3 ) ) & 0xFF ) == 0 )
                continue;
            else
                zeros = 0;

            if( j % 16 == 0 )
            {
                if( j > 0 )
                {
                    mbedtls_snprintf( str + idx, sizeof( str ) - idx, "\n" );
                    debug_send_line( ssl, level, file, line, str );
                    idx = 0;
                }
            }

            idx += mbedtls_snprintf( str + idx, sizeof( str ) - idx, " %02x", (unsigned int)
                             ( X->p[i - 1] >> ( k << 3 ) ) & 0xFF );

            j++;
        }

    }

    if( zeros == 1 )
        idx += mbedtls_snprintf( str + idx, sizeof( str ) - idx, " 00" );

    mbedtls_snprintf( str + idx, sizeof( str ) - idx, "\n" );
    debug_send_line( ssl, level, file, line, str );
}
#endif /* MBEDTLS_BIGNUM_C */

#if defined(MBEDTLS_X509_CRT_PARSE_C)
static void debug_print_pk( const mbedtls_ssl_context *ssl, int level,
                            const char *file, int line,
                            const char *text, const mbedtls_pk_context *pk )
{
    size_t i;
    mbedtls_pk_debug_item items[MBEDTLS_PK_DEBUG_MAX_ITEMS];
    char name[16];

    memset( items, 0, sizeof( items ) );

    if( mbedtls_pk_debug( pk, items ) != 0 )
    {
        debug_send_line( ssl, level, file, line,
                          "invalid PK context\n" );
        return;
    }

    for( i = 0; i < MBEDTLS_PK_DEBUG_MAX_ITEMS; i++ )
    {
        if( items[i].type == MBEDTLS_PK_DEBUG_NONE )
            return;

        mbedtls_snprintf( name, sizeof( name ), "%s%s", text, items[i].name );
        name[sizeof( name ) - 1] = '\0';

        if( items[i].type == MBEDTLS_PK_DEBUG_MPI )
            mbedtls_debug_print_mpi( ssl, level, file, line, name, items[i].value );
        else
#if defined(MBEDTLS_ECP_C)
        if( items[i].type == MBEDTLS_PK_DEBUG_ECP )
            mbedtls_debug_print_ecp( ssl, level, file, line, name, items[i].value );
        else
#endif
            debug_send_line( ssl, level, file, line,
                              "should not happen\n" );
    }
}

static void debug_print_line_by_line( const mbedtls_ssl_context *ssl, int level,
                                      const char *file, int line, const char *text )
{
    char str[DEBUG_BUF_SIZE];
    const char *start, *cur;

    start = text;
    for( cur = text; *cur != '\0'; cur++ )
    {
        if( *cur == '\n' )
        {
            size_t len = cur - start + 1;
            if( len > DEBUG_BUF_SIZE - 1 )
                len = DEBUG_BUF_SIZE - 1;

            memcpy( str, start, len );
            str[len] = '\0';

            debug_send_line( ssl, level, file, line, str );

            start = cur + 1;
        }
    }
}

void mbedtls_debug_print_crt( const mbedtls_ssl_context *ssl, int level,
                      const char *file, int line,
                      const char *text, const mbedtls_x509_crt *crt )
{
    char str[DEBUG_BUF_SIZE];
    int i = 0;

    if( ssl->conf == NULL || ssl->conf->f_dbg == NULL || crt == NULL || level > debug_threshold )
        return;

    while( crt != NULL )
    {
        char buf[1024];

        mbedtls_snprintf( str, sizeof( str ), "%s #%d:\n", text, ++i );
        debug_send_line( ssl, level, file, line, str );

        mbedtls_x509_crt_info( buf, sizeof( buf ) - 1, "", crt );
        debug_print_line_by_line( ssl, level, file, line, buf );

        debug_print_pk( ssl, level, file, line, "crt->", &crt->pk );

        crt = crt->next;
    }
}
#endif /* MBEDTLS_X509_CRT_PARSE_C */

#endif /* MBEDTLS_DEBUG_C */
