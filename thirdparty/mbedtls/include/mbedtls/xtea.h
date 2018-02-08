/**
 * \file xtea.h
 *
 * \brief XTEA block cipher (32-bit)
 */
/*
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
#ifndef MBEDTLS_XTEA_H
#define MBEDTLS_XTEA_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>
#include <stdint.h>

#define MBEDTLS_XTEA_ENCRYPT     1
#define MBEDTLS_XTEA_DECRYPT     0

#define MBEDTLS_ERR_XTEA_INVALID_INPUT_LENGTH             -0x0028  /**< The data input has an invalid length. */
#define MBEDTLS_ERR_XTEA_HW_ACCEL_FAILED                  -0x0029  /**< XTEA hardware accelerator failed. */

#if !defined(MBEDTLS_XTEA_ALT)
// Regular implementation
//

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          XTEA context structure
 */
typedef struct
{
    uint32_t k[4];       /*!< key */
}
mbedtls_xtea_context;

/**
 * \brief          Initialize XTEA context
 *
 * \param ctx      XTEA context to be initialized
 */
void mbedtls_xtea_init( mbedtls_xtea_context *ctx );

/**
 * \brief          Clear XTEA context
 *
 * \param ctx      XTEA context to be cleared
 */
void mbedtls_xtea_free( mbedtls_xtea_context *ctx );

/**
 * \brief          XTEA key schedule
 *
 * \param ctx      XTEA context to be initialized
 * \param key      the secret key
 */
void mbedtls_xtea_setup( mbedtls_xtea_context *ctx, const unsigned char key[16] );

/**
 * \brief          XTEA cipher function
 *
 * \param ctx      XTEA context
 * \param mode     MBEDTLS_XTEA_ENCRYPT or MBEDTLS_XTEA_DECRYPT
 * \param input    8-byte input block
 * \param output   8-byte output block
 *
 * \return         0 if successful
 */
int mbedtls_xtea_crypt_ecb( mbedtls_xtea_context *ctx,
                    int mode,
                    const unsigned char input[8],
                    unsigned char output[8] );

#if defined(MBEDTLS_CIPHER_MODE_CBC)
/**
 * \brief          XTEA CBC cipher function
 *
 * \param ctx      XTEA context
 * \param mode     MBEDTLS_XTEA_ENCRYPT or MBEDTLS_XTEA_DECRYPT
 * \param length   the length of input, multiple of 8
 * \param iv       initialization vector for CBC mode
 * \param input    input block
 * \param output   output block
 *
 * \return         0 if successful,
 *                 MBEDTLS_ERR_XTEA_INVALID_INPUT_LENGTH if the length % 8 != 0
 */
int mbedtls_xtea_crypt_cbc( mbedtls_xtea_context *ctx,
                    int mode,
                    size_t length,
                    unsigned char iv[8],
                    const unsigned char *input,
                    unsigned char *output);
#endif /* MBEDTLS_CIPHER_MODE_CBC */

#ifdef __cplusplus
}
#endif

#else  /* MBEDTLS_XTEA_ALT */
#include "xtea_alt.h"
#endif /* MBEDTLS_XTEA_ALT */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
int mbedtls_xtea_self_test( int verbose );

#ifdef __cplusplus
}
#endif

#endif /* xtea.h */
