/**
 * \file ssl_cache.h
 *
 * \brief SSL session cache implementation
 */
/*
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
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
 *
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */
#ifndef MBEDTLS_SSL_CACHE_H
#define MBEDTLS_SSL_CACHE_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "ssl.h"

#if defined(MBEDTLS_THREADING_C)
#include "threading.h"
#endif

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h or define them on the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT)
#define MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT       86400   /*!< 1 day  */
#endif

#if !defined(MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES)
#define MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES      50   /*!< Maximum entries in cache */
#endif

/* \} name SECTION: Module settings */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mbedtls_ssl_cache_context mbedtls_ssl_cache_context;
typedef struct mbedtls_ssl_cache_entry mbedtls_ssl_cache_entry;

/**
 * \brief   This structure is used for storing cache entries
 */
struct mbedtls_ssl_cache_entry
{
#if defined(MBEDTLS_HAVE_TIME)
    mbedtls_time_t timestamp;           /*!< entry timestamp    */
#endif
    mbedtls_ssl_session session;        /*!< entry session      */
#if defined(MBEDTLS_X509_CRT_PARSE_C)
    mbedtls_x509_buf peer_cert;         /*!< entry peer_cert    */
#endif
    mbedtls_ssl_cache_entry *next;      /*!< chain pointer      */
};

/**
 * \brief Cache context
 */
struct mbedtls_ssl_cache_context
{
    mbedtls_ssl_cache_entry *chain;     /*!< start of the chain     */
    int timeout;                /*!< cache entry timeout    */
    int max_entries;            /*!< maximum entries        */
#if defined(MBEDTLS_THREADING_C)
    mbedtls_threading_mutex_t mutex;    /*!< mutex                  */
#endif
};

/**
 * \brief          Initialize an SSL cache context
 *
 * \param cache    SSL cache context
 */
void mbedtls_ssl_cache_init( mbedtls_ssl_cache_context *cache );

/**
 * \brief          Cache get callback implementation
 *                 (Thread-safe if MBEDTLS_THREADING_C is enabled)
 *
 * \param data     SSL cache context
 * \param session  session to retrieve entry for
 */
int mbedtls_ssl_cache_get( void *data, mbedtls_ssl_session *session );

/**
 * \brief          Cache set callback implementation
 *                 (Thread-safe if MBEDTLS_THREADING_C is enabled)
 *
 * \param data     SSL cache context
 * \param session  session to store entry for
 */
int mbedtls_ssl_cache_set( void *data, const mbedtls_ssl_session *session );

#if defined(MBEDTLS_HAVE_TIME)
/**
 * \brief          Set the cache timeout
 *                 (Default: MBEDTLS_SSL_CACHE_DEFAULT_TIMEOUT (1 day))
 *
 *                 A timeout of 0 indicates no timeout.
 *
 * \param cache    SSL cache context
 * \param timeout  cache entry timeout in seconds
 */
void mbedtls_ssl_cache_set_timeout( mbedtls_ssl_cache_context *cache, int timeout );
#endif /* MBEDTLS_HAVE_TIME */

/**
 * \brief          Set the maximum number of cache entries
 *                 (Default: MBEDTLS_SSL_CACHE_DEFAULT_MAX_ENTRIES (50))
 *
 * \param cache    SSL cache context
 * \param max      cache entry maximum
 */
void mbedtls_ssl_cache_set_max_entries( mbedtls_ssl_cache_context *cache, int max );

/**
 * \brief          Free referenced items in a cache context and clear memory
 *
 * \param cache    SSL cache context
 */
void mbedtls_ssl_cache_free( mbedtls_ssl_cache_context *cache );

#ifdef __cplusplus
}
#endif

#endif /* ssl_cache.h */
