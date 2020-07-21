/**
 * \file ssl_cookie.h
 *
 * \brief DTLS cookie callbacks implementation
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
#ifndef MBEDTLS_SSL_COOKIE_H
#define MBEDTLS_SSL_COOKIE_H

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
#ifndef MBEDTLS_SSL_COOKIE_TIMEOUT
#define MBEDTLS_SSL_COOKIE_TIMEOUT     60 /**< Default expiration delay of DTLS cookies, in seconds if HAVE_TIME, or in number of cookies issued */
#endif

/* \} name SECTION: Module settings */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Context for the default cookie functions.
 */
typedef struct mbedtls_ssl_cookie_ctx
{
    mbedtls_md_context_t    hmac_ctx;   /*!< context for the HMAC portion   */
#if !defined(MBEDTLS_HAVE_TIME)
    unsigned long   serial;     /*!< serial number for expiration   */
#endif
    unsigned long   timeout;    /*!< timeout delay, in seconds if HAVE_TIME,
                                     or in number of tickets issued */

#if defined(MBEDTLS_THREADING_C)
    mbedtls_threading_mutex_t mutex;
#endif
} mbedtls_ssl_cookie_ctx;

/**
 * \brief          Initialize cookie context
 */
void mbedtls_ssl_cookie_init( mbedtls_ssl_cookie_ctx *ctx );

/**
 * \brief          Setup cookie context (generate keys)
 */
int mbedtls_ssl_cookie_setup( mbedtls_ssl_cookie_ctx *ctx,
                      int (*f_rng)(void *, unsigned char *, size_t),
                      void *p_rng );

/**
 * \brief          Set expiration delay for cookies
 *                 (Default MBEDTLS_SSL_COOKIE_TIMEOUT)
 *
 * \param ctx      Cookie contex
 * \param delay    Delay, in seconds if HAVE_TIME, or in number of cookies
 *                 issued in the meantime.
 *                 0 to disable expiration (NOT recommended)
 */
void mbedtls_ssl_cookie_set_timeout( mbedtls_ssl_cookie_ctx *ctx, unsigned long delay );

/**
 * \brief          Free cookie context
 */
void mbedtls_ssl_cookie_free( mbedtls_ssl_cookie_ctx *ctx );

/**
 * \brief          Generate cookie, see \c mbedtls_ssl_cookie_write_t
 */
mbedtls_ssl_cookie_write_t mbedtls_ssl_cookie_write;

/**
 * \brief          Verify cookie, see \c mbedtls_ssl_cookie_write_t
 */
mbedtls_ssl_cookie_check_t mbedtls_ssl_cookie_check;

#ifdef __cplusplus
}
#endif

#endif /* ssl_cookie.h */
