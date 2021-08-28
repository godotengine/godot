/**
 * \file havege.h
 *
 * \brief HAVEGE: HArdware Volatile Entropy Gathering and Expansion
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
#ifndef MBEDTLS_HAVEGE_H
#define MBEDTLS_HAVEGE_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>

#define MBEDTLS_HAVEGE_COLLECT_SIZE 1024

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          HAVEGE state structure
 */
typedef struct mbedtls_havege_state
{
    int PT1, PT2, offset[2];
    int pool[MBEDTLS_HAVEGE_COLLECT_SIZE];
    int WALK[8192];
}
mbedtls_havege_state;

/**
 * \brief          HAVEGE initialization
 *
 * \param hs       HAVEGE state to be initialized
 */
void mbedtls_havege_init( mbedtls_havege_state *hs );

/**
 * \brief          Clear HAVEGE state
 *
 * \param hs       HAVEGE state to be cleared
 */
void mbedtls_havege_free( mbedtls_havege_state *hs );

/**
 * \brief          HAVEGE rand function
 *
 * \param p_rng    A HAVEGE state
 * \param output   Buffer to fill
 * \param len      Length of buffer
 *
 * \return         0
 */
int mbedtls_havege_random( void *p_rng, unsigned char *output, size_t len );

#ifdef __cplusplus
}
#endif

#endif /* havege.h */
