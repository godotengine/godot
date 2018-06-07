/**
 * \file sha512.h
 * \brief This file contains SHA-384 and SHA-512 definitions and functions.
 *
 * The Secure Hash Algorithms 384 and 512 (SHA-384 and SHA-512) cryptographic
 * hash functions are defined in <em>FIPS 180-4: Secure Hash Standard (SHS)</em>.
 */
/*
 *  Copyright (C) 2006-2018, Arm Limited (or its affiliates), All Rights Reserved
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
#ifndef MBEDTLS_SHA512_H
#define MBEDTLS_SHA512_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>
#include <stdint.h>

#define MBEDTLS_ERR_SHA512_HW_ACCEL_FAILED                -0x0039  /**< SHA-512 hardware accelerator failed */

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(MBEDTLS_SHA512_ALT)
// Regular implementation
//

/**
 * \brief          The SHA-512 context structure.
 *
 *                 The structure is used both for SHA-384 and for SHA-512
 *                 checksum calculations. The choice between these two is
 *                 made in the call to mbedtls_sha512_starts_ret().
 */
typedef struct
{
    uint64_t total[2];          /*!< The number of Bytes processed. */
    uint64_t state[8];          /*!< The intermediate digest state. */
    unsigned char buffer[128];  /*!< The data block being processed. */
    int is384;                  /*!< Determines which function to use:
                                     0: Use SHA-512, or 1: Use SHA-384. */
}
mbedtls_sha512_context;

#else  /* MBEDTLS_SHA512_ALT */
#include "sha512_alt.h"
#endif /* MBEDTLS_SHA512_ALT */

/**
 * \brief          This function initializes a SHA-512 context.
 *
 * \param ctx      The SHA-512 context to initialize.
 */
void mbedtls_sha512_init( mbedtls_sha512_context *ctx );

/**
 * \brief          This function clears a SHA-512 context.
 *
 * \param ctx      The SHA-512 context to clear.
 */
void mbedtls_sha512_free( mbedtls_sha512_context *ctx );

/**
 * \brief          This function clones the state of a SHA-512 context.
 *
 * \param dst      The destination context.
 * \param src      The context to clone.
 */
void mbedtls_sha512_clone( mbedtls_sha512_context *dst,
                           const mbedtls_sha512_context *src );

/**
 * \brief          This function starts a SHA-384 or SHA-512 checksum
 *                 calculation.
 *
 * \param ctx      The SHA-512 context to initialize.
 * \param is384    Determines which function to use:
 *                 0: Use SHA-512, or 1: Use SHA-384.
 *
 * \return         \c 0 on success.
 */
int mbedtls_sha512_starts_ret( mbedtls_sha512_context *ctx, int is384 );

/**
 * \brief          This function feeds an input buffer into an ongoing
 *                 SHA-512 checksum calculation.
 *
 * \param ctx      The SHA-512 context.
 * \param input    The buffer holding the input data.
 * \param ilen     The length of the input data.
 *
 * \return         \c 0 on success.
 */
int mbedtls_sha512_update_ret( mbedtls_sha512_context *ctx,
                    const unsigned char *input,
                    size_t ilen );

/**
 * \brief          This function finishes the SHA-512 operation, and writes
 *                 the result to the output buffer. This function is for
 *                 internal use only.
 *
 * \param ctx      The SHA-512 context.
 * \param output   The SHA-384 or SHA-512 checksum result.
 *
 * \return         \c 0 on success.
 */
int mbedtls_sha512_finish_ret( mbedtls_sha512_context *ctx,
                               unsigned char output[64] );

/**
 * \brief          This function processes a single data block within
 *                 the ongoing SHA-512 computation.
 *
 * \param ctx      The SHA-512 context.
 * \param data     The buffer holding one block of data.
 *
 * \return         \c 0 on success.
 */
int mbedtls_internal_sha512_process( mbedtls_sha512_context *ctx,
                                     const unsigned char data[128] );
#if !defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED      __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief          This function starts a SHA-384 or SHA-512 checksum
 *                 calculation.
 *
 * \deprecated     Superseded by mbedtls_sha512_starts_ret() in 2.7.0
 *
 * \param ctx      The SHA-512 context to initialize.
 * \param is384    Determines which function to use:
 *                 0: Use SHA-512, or 1: Use SHA-384.
 */
MBEDTLS_DEPRECATED void mbedtls_sha512_starts( mbedtls_sha512_context *ctx,
                                               int is384 );

/**
 * \brief          This function feeds an input buffer into an ongoing
 *                 SHA-512 checksum calculation.
 *
 * \deprecated     Superseded by mbedtls_sha512_update_ret() in 2.7.0.
 *
 * \param ctx      The SHA-512 context.
 * \param input    The buffer holding the data.
 * \param ilen     The length of the input data.
 */
MBEDTLS_DEPRECATED void mbedtls_sha512_update( mbedtls_sha512_context *ctx,
                                               const unsigned char *input,
                                               size_t ilen );

/**
 * \brief          This function finishes the SHA-512 operation, and writes
 *                 the result to the output buffer.
 *
 * \deprecated     Superseded by mbedtls_sha512_finish_ret() in 2.7.0.
 *
 * \param ctx      The SHA-512 context.
 * \param output   The SHA-384 or SHA-512 checksum result.
 */
MBEDTLS_DEPRECATED void mbedtls_sha512_finish( mbedtls_sha512_context *ctx,
                                               unsigned char output[64] );

/**
 * \brief          This function processes a single data block within
 *                 the ongoing SHA-512 computation. This function is for
 *                 internal use only.
 *
 * \deprecated     Superseded by mbedtls_internal_sha512_process() in 2.7.0.
 *
 * \param ctx      The SHA-512 context.
 * \param data     The buffer holding one block of data.
 */
MBEDTLS_DEPRECATED void mbedtls_sha512_process(
                                            mbedtls_sha512_context *ctx,
                                            const unsigned char data[128] );

#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */

/**
 * \brief          This function calculates the SHA-512 or SHA-384
 *                 checksum of a buffer.
 *
 *                 The function allocates the context, performs the
 *                 calculation, and frees the context.
 *
 *                 The SHA-512 result is calculated as
 *                 output = SHA-512(input buffer).
 *
 * \param input    The buffer holding the input data.
 * \param ilen     The length of the input data.
 * \param output   The SHA-384 or SHA-512 checksum result.
 * \param is384    Determines which function to use:
 *                 0: Use SHA-512, or 1: Use SHA-384.
 *
 * \return         \c 0 on success.
 */
int mbedtls_sha512_ret( const unsigned char *input,
                        size_t ilen,
                        unsigned char output[64],
                        int is384 );

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED      __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief          This function calculates the SHA-512 or SHA-384
 *                 checksum of a buffer.
 *
 *                 The function allocates the context, performs the
 *                 calculation, and frees the context.
 *
 *                 The SHA-512 result is calculated as
 *                 output = SHA-512(input buffer).
 *
 * \deprecated     Superseded by mbedtls_sha512_ret() in 2.7.0
 *
 * \param input    The buffer holding the data.
 * \param ilen     The length of the input data.
 * \param output   The SHA-384 or SHA-512 checksum result.
 * \param is384    Determines which function to use:
 *                 0: Use SHA-512, or 1: Use SHA-384.
 */
MBEDTLS_DEPRECATED void mbedtls_sha512( const unsigned char *input,
                                        size_t ilen,
                                        unsigned char output[64],
                                        int is384 );

#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */
 /**
 * \brief          The SHA-384 or SHA-512 checkup routine.
 *
 * \return         \c 0 on success.
 * \return         \c 1 on failure.
 */
int mbedtls_sha512_self_test( int verbose );

#ifdef __cplusplus
}
#endif

#endif /* mbedtls_sha512.h */
