/*
 *  SSL session cache implementation
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
/*
 * These session callbacks use a simple chained list
 * to store and retrieve the session information.
 */

#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#if defined(MBEDTLS_SSL_CACHE_C)

#if defined(MBEDTLS_PLATFORM_C)
#include "mbedtls/platform.h"
#else
#include <stdlib.h>
#define mbedtls_calloc    calloc
#define mbedtls_free      free
#endif

#include "mbedtls/ssl_cache.h"

#include <string.h>

void mbedtls_ssl_cache_init( mbedtls_ssl_cache_context *cache )
{
    memset( cache, 0, sizeof( mbedtls_ssl_cache_context ) );

    cache->timeout = MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT;
    cache->max_entries = MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES;

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_init( &cache->mutex );
#endif
}

int mbedtls_ssl_cache_get( void *data, mbedtls_ssl_session *session )
{
    int ret = 1;
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t t = mbedtls_time( NULL );
#endif
    mbedtls_ssl_cache_context *cache = (mbedtls_ssl_cache_context *) data;
    mbedtls_ssl_cache_entry *cur, *entry;

#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_lock( &cache->mutex ) != 0 )
        return( 1 );
#endif

    cur = cache->chain;
    entry = NULL;

    while( cur != NULL )
    {
        entry = cur;
        cur = cur->next;

#if defined(MBEDTLS_HAVE_TIME)
        if( cache->timeout != 0 &&
            (int) ( t - entry->timestamp ) > cache->timeout )
            continue;
#endif

        if( session->ciphersuite != entry->session.ciphersuite ||
            session->compression != entry->session.compression ||
            session->id_len != entry->session.id_len )
            continue;

        if( memcmp( session->id, entry->session.id,
                    entry->session.id_len ) != 0 )
            continue;

        memcpy( session->master, entry->session.master, 48 );

        session->verify_result = entry->session.verify_result;

#if defined(MBEDTLS_X509_CRT_PARSE_C)
        /*
         * Restore peer certificate (without rest of the original chain)
         */
        if( entry->peer_cert.p != NULL )
        {
            if( ( session->peer_cert = mbedtls_calloc( 1,
                                 sizeof(mbedtls_x509_crt) ) ) == NULL )
            {
                ret = 1;
                goto exit;
            }

            mbedtls_x509_crt_init( session->peer_cert );
            if( mbedtls_x509_crt_parse( session->peer_cert, entry->peer_cert.p,
                                entry->peer_cert.len ) != 0 )
            {
                mbedtls_free( session->peer_cert );
                session->peer_cert = NULL;
                ret = 1;
                goto exit;
            }
        }
#endif /* MBEDTLS_X509_CRT_PARSE_C */

        ret = 0;
        goto exit;
    }

exit:
#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_unlock( &cache->mutex ) != 0 )
        ret = 1;
#endif

    return( ret );
}

int mbedtls_ssl_cache_set( void *data, const mbedtls_ssl_session *session )
{
    int ret = 1;
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t t = mbedtls_time( NULL ), oldest = 0;
    mbedtls_ssl_cache_entry *old = NULL;
#endif
    mbedtls_ssl_cache_context *cache = (mbedtls_ssl_cache_context *) data;
    mbedtls_ssl_cache_entry *cur, *prv;
    int count = 0;

#if defined(MBEDTLS_THREADING_C)
    if( ( ret = mbedtls_mutex_lock( &cache->mutex ) ) != 0 )
        return( ret );
#endif

    cur = cache->chain;
    prv = NULL;

    while( cur != NULL )
    {
        count++;

#if defined(MBEDTLS_HAVE_TIME)
        if( cache->timeout != 0 &&
            (int) ( t - cur->timestamp ) > cache->timeout )
        {
            cur->timestamp = t;
            break; /* expired, reuse this slot, update timestamp */
        }
#endif

        if( memcmp( session->id, cur->session.id, cur->session.id_len ) == 0 )
            break; /* client reconnected, keep timestamp for session id */

#if defined(MBEDTLS_HAVE_TIME)
        if( oldest == 0 || cur->timestamp < oldest )
        {
            oldest = cur->timestamp;
            old = cur;
        }
#endif

        prv = cur;
        cur = cur->next;
    }

    if( cur == NULL )
    {
#if defined(MBEDTLS_HAVE_TIME)
        /*
         * Reuse oldest entry if max_entries reached
         */
        if( count >= cache->max_entries )
        {
            if( old == NULL )
            {
                ret = 1;
                goto exit;
            }

            cur = old;
        }
#else /* MBEDTLS_HAVE_TIME */
        /*
         * Reuse first entry in chain if max_entries reached,
         * but move to last place
         */
        if( count >= cache->max_entries )
        {
            if( cache->chain == NULL )
            {
                ret = 1;
                goto exit;
            }

            cur = cache->chain;
            cache->chain = cur->next;
            cur->next = NULL;
            prv->next = cur;
        }
#endif /* MBEDTLS_HAVE_TIME */
        else
        {
            /*
             * max_entries not reached, create new entry
             */
            cur = mbedtls_calloc( 1, sizeof(mbedtls_ssl_cache_entry) );
            if( cur == NULL )
            {
                ret = 1;
                goto exit;
            }

            if( prv == NULL )
                cache->chain = cur;
            else
                prv->next = cur;
        }

#if defined(MBEDTLS_HAVE_TIME)
        cur->timestamp = t;
#endif
    }

    memcpy( &cur->session, session, sizeof( mbedtls_ssl_session ) );

#if defined(MBEDTLS_X509_CRT_PARSE_C)
    /*
     * If we're reusing an entry, free its certificate first
     */
    if( cur->peer_cert.p != NULL )
    {
        mbedtls_free( cur->peer_cert.p );
        memset( &cur->peer_cert, 0, sizeof(mbedtls_x509_buf) );
    }

    /*
     * Store peer certificate
     */
    if( session->peer_cert != NULL )
    {
        cur->peer_cert.p = mbedtls_calloc( 1, session->peer_cert->raw.len );
        if( cur->peer_cert.p == NULL )
        {
            ret = 1;
            goto exit;
        }

        memcpy( cur->peer_cert.p, session->peer_cert->raw.p,
                session->peer_cert->raw.len );
        cur->peer_cert.len = session->peer_cert->raw.len;

        cur->session.peer_cert = NULL;
    }
#endif /* MBEDTLS_X509_CRT_PARSE_C */

    ret = 0;

exit:
#if defined(MBEDTLS_THREADING_C)
    if( mbedtls_mutex_unlock( &cache->mutex ) != 0 )
        ret = 1;
#endif

    return( ret );
}

#if defined(MBEDTLS_HAVE_TIME)
void mbedtls_ssl_cache_set_timeout( mbedtls_ssl_cache_context *cache, int timeout )
{
    if( timeout < 0 ) timeout = 0;

    cache->timeout = timeout;
}
#endif /* MBEDTLS_HAVE_TIME */

void mbedtls_ssl_cache_set_max_entries( mbedtls_ssl_cache_context *cache, int max )
{
    if( max < 0 ) max = 0;

    cache->max_entries = max;
}

void mbedtls_ssl_cache_free( mbedtls_ssl_cache_context *cache )
{
    mbedtls_ssl_cache_entry *cur, *prv;

    cur = cache->chain;

    while( cur != NULL )
    {
        prv = cur;
        cur = cur->next;

        mbedtls_ssl_session_free( &prv->session );

#if defined(MBEDTLS_X509_CRT_PARSE_C)
        mbedtls_free( prv->peer_cert.p );
#endif /* MBEDTLS_X509_CRT_PARSE_C */

        mbedtls_free( prv );
    }

#if defined(MBEDTLS_THREADING_C)
    mbedtls_mutex_free( &cache->mutex );
#endif
    cache->chain = NULL;
}

#endif /* MBEDTLS_SSL_CACHE_C */
