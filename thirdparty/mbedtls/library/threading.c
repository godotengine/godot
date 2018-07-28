/*
 *  Threading abstraction layer
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

#if defined(MBEDTLS_THREADING_C)

#include "mbedtls/threading.h"

#if defined(MBEDTLS_THREADING_PTHREAD)
static void threading_mutex_init_pthread( mbedtls_threading_mutex_t *mutex )
{
    if( mutex == NULL )
        return;

    mutex->is_valid = pthread_mutex_init( &mutex->mutex, NULL ) == 0;
}

static void threading_mutex_free_pthread( mbedtls_threading_mutex_t *mutex )
{
    if( mutex == NULL || !mutex->is_valid )
        return;

    (void) pthread_mutex_destroy( &mutex->mutex );
    mutex->is_valid = 0;
}

static int threading_mutex_lock_pthread( mbedtls_threading_mutex_t *mutex )
{
    if( mutex == NULL || ! mutex->is_valid )
        return( MBEDTLS_ERR_THREADING_BAD_INPUT_DATA );

    if( pthread_mutex_lock( &mutex->mutex ) != 0 )
        return( MBEDTLS_ERR_THREADING_MUTEX_ERROR );

    return( 0 );
}

static int threading_mutex_unlock_pthread( mbedtls_threading_mutex_t *mutex )
{
    if( mutex == NULL || ! mutex->is_valid )
        return( MBEDTLS_ERR_THREADING_BAD_INPUT_DATA );

    if( pthread_mutex_unlock( &mutex->mutex ) != 0 )
        return( MBEDTLS_ERR_THREADING_MUTEX_ERROR );

    return( 0 );
}

void (*mbedtls_mutex_init)( mbedtls_threading_mutex_t * ) = threading_mutex_init_pthread;
void (*mbedtls_mutex_free)( mbedtls_threading_mutex_t * ) = threading_mutex_free_pthread;
int (*mbedtls_mutex_lock)( mbedtls_threading_mutex_t * ) = threading_mutex_lock_pthread;
int (*mbedtls_mutex_unlock)( mbedtls_threading_mutex_t * ) = threading_mutex_unlock_pthread;

/*
 * With phtreads we can statically initialize mutexes
 */
#define MUTEX_INIT  = { PTHREAD_MUTEX_INITIALIZER, 1 }

#endif /* MBEDTLS_THREADING_PTHREAD */

#if defined(MBEDTLS_THREADING_ALT)
static int threading_mutex_fail( mbedtls_threading_mutex_t *mutex )
{
    ((void) mutex );
    return( MBEDTLS_ERR_THREADING_BAD_INPUT_DATA );
}
static void threading_mutex_dummy( mbedtls_threading_mutex_t *mutex )
{
    ((void) mutex );
    return;
}

void (*mbedtls_mutex_init)( mbedtls_threading_mutex_t * ) = threading_mutex_dummy;
void (*mbedtls_mutex_free)( mbedtls_threading_mutex_t * ) = threading_mutex_dummy;
int (*mbedtls_mutex_lock)( mbedtls_threading_mutex_t * ) = threading_mutex_fail;
int (*mbedtls_mutex_unlock)( mbedtls_threading_mutex_t * ) = threading_mutex_fail;

/*
 * Set functions pointers and initialize global mutexes
 */
void mbedtls_threading_set_alt( void (*mutex_init)( mbedtls_threading_mutex_t * ),
                       void (*mutex_free)( mbedtls_threading_mutex_t * ),
                       int (*mutex_lock)( mbedtls_threading_mutex_t * ),
                       int (*mutex_unlock)( mbedtls_threading_mutex_t * ) )
{
    mbedtls_mutex_init = mutex_init;
    mbedtls_mutex_free = mutex_free;
    mbedtls_mutex_lock = mutex_lock;
    mbedtls_mutex_unlock = mutex_unlock;

#if defined(MBEDTLS_FS_IO)
    mbedtls_mutex_init( &mbedtls_threading_readdir_mutex );
#endif
}

/*
 * Free global mutexes
 */
void mbedtls_threading_free_alt( void )
{
#if defined(MBEDTLS_FS_IO)
    mbedtls_mutex_free( &mbedtls_threading_readdir_mutex );
#endif
}
#endif /* MBEDTLS_THREADING_ALT */

/*
 * Define global mutexes
 */
#ifndef MUTEX_INIT
#define MUTEX_INIT
#endif
#if defined(MBEDTLS_FS_IO)
mbedtls_threading_mutex_t mbedtls_threading_readdir_mutex MUTEX_INIT;
#endif

#endif /* MBEDTLS_THREADING_C */
