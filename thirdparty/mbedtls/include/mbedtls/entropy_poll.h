/**
 * \file entropy_poll.h
 *
 * \brief Platform-specific and custom entropy polling functions
 */
/*
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
#ifndef MBEDTLS_ENTROPY_POLL_H
#define MBEDTLS_ENTROPY_POLL_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Default thresholds for built-in sources, in bytes
 */
#define MBEDTLS_ENTROPY_MIN_PLATFORM     32     /**< Minimum for platform source    */
#define MBEDTLS_ENTROPY_MIN_HAVEGE       32     /**< Minimum for HAVEGE             */
#define MBEDTLS_ENTROPY_MIN_HARDCLOCK     4     /**< Minimum for mbedtls_timing_hardclock()        */
#if !defined(MBEDTLS_ENTROPY_MIN_HARDWARE)
#define MBEDTLS_ENTROPY_MIN_HARDWARE     32     /**< Minimum for the hardware source */
#endif

/**
 * \brief           Entropy poll callback that provides 0 entropy.
 */
#if defined(MBEDTLS_TEST_NULL_ENTROPY)
    int mbedtls_null_entropy_poll( void *data,
                                unsigned char *output, size_t len, size_t *olen );
#endif

#if !defined(MBEDTLS_NO_PLATFORM_ENTROPY)
/**
 * \brief           Platform-specific entropy poll callback
 */
int mbedtls_platform_entropy_poll( void *data,
                           unsigned char *output, size_t len, size_t *olen );
#endif

#if defined(MBEDTLS_HAVEGE_C)
/**
 * \brief           HAVEGE based entropy poll callback
 *
 * Requires an HAVEGE state as its data pointer.
 */
int mbedtls_havege_poll( void *data,
                 unsigned char *output, size_t len, size_t *olen );
#endif

#if defined(MBEDTLS_TIMING_C)
/**
 * \brief           mbedtls_timing_hardclock-based entropy poll callback
 */
int mbedtls_hardclock_poll( void *data,
                    unsigned char *output, size_t len, size_t *olen );
#endif

#if defined(MBEDTLS_ENTROPY_HARDWARE_ALT)
/**
 * \brief           Entropy poll callback for a hardware source
 *
 * \warning         This is not provided by mbed TLS!
 *                  See \c MBEDTLS_ENTROPY_HARDWARE_ALT in config.h.
 *
 * \note            This must accept NULL as its first argument.
 */
int mbedtls_hardware_poll( void *data,
                           unsigned char *output, size_t len, size_t *olen );
#endif

#if defined(MBEDTLS_ENTROPY_NV_SEED)
/**
 * \brief           Entropy poll callback for a non-volatile seed file
 *
 * \note            This must accept NULL as its first argument.
 */
int mbedtls_nv_seed_poll( void *data,
                          unsigned char *output, size_t len, size_t *olen );
#endif

#ifdef __cplusplus
}
#endif

#endif /* entropy_poll.h */
