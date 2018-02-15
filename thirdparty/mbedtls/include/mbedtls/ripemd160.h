/**
 * \file ripemd160.h
 *
 * \brief RIPE MD-160 message digest
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
#ifndef MBEDTLS_RIPEMD160_H
#define MBEDTLS_RIPEMD160_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stddef.h>
#include <stdint.h>

#define MBEDTLS_ERR_RIPEMD160_HW_ACCEL_FAILED             -0x0031  /**< RIPEMD160 hardware accelerator failed */

#if ( defined(__ARMCC_VERSION) || defined(_MSC_VER) ) && \
    !defined(inline) && !defined(__cplusplus)
#define inline __inline
#endif

#if !defined(MBEDTLS_RIPEMD160_ALT)
// Regular implementation
//

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          RIPEMD-160 context structure
 */
typedef struct
{
    uint32_t total[2];          /*!< number of bytes processed  */
    uint32_t state[5];          /*!< intermediate digest state  */
    unsigned char buffer[64];   /*!< data block being processed */
}
mbedtls_ripemd160_context;

/**
 * \brief          Initialize RIPEMD-160 context
 *
 * \param ctx      RIPEMD-160 context to be initialized
 */
void mbedtls_ripemd160_init( mbedtls_ripemd160_context *ctx );

/**
 * \brief          Clear RIPEMD-160 context
 *
 * \param ctx      RIPEMD-160 context to be cleared
 */
void mbedtls_ripemd160_free( mbedtls_ripemd160_context *ctx );

/**
 * \brief          Clone (the state of) an RIPEMD-160 context
 *
 * \param dst      The destination context
 * \param src      The context to be cloned
 */
void mbedtls_ripemd160_clone( mbedtls_ripemd160_context *dst,
                        const mbedtls_ripemd160_context *src );

/**
 * \brief          RIPEMD-160 context setup
 *
 * \param ctx      context to be initialized
 *
 * \return         0 if successful
 */
int mbedtls_ripemd160_starts_ret( mbedtls_ripemd160_context *ctx );

/**
 * \brief          RIPEMD-160 process buffer
 *
 * \param ctx      RIPEMD-160 context
 * \param input    buffer holding the data
 * \param ilen     length of the input data
 *
 * \return         0 if successful
 */
int mbedtls_ripemd160_update_ret( mbedtls_ripemd160_context *ctx,
                                  const unsigned char *input,
                                  size_t ilen );

/**
 * \brief          RIPEMD-160 final digest
 *
 * \param ctx      RIPEMD-160 context
 * \param output   RIPEMD-160 checksum result
 *
 * \return         0 if successful
 */
int mbedtls_ripemd160_finish_ret( mbedtls_ripemd160_context *ctx,
                                  unsigned char output[20] );

/**
 * \brief          RIPEMD-160 process data block (internal use only)
 *
 * \param ctx      RIPEMD-160 context
 * \param data     buffer holding one block of data
 *
 * \return         0 if successful
 */
int mbedtls_internal_ripemd160_process( mbedtls_ripemd160_context *ctx,
                                        const unsigned char data[64] );

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED      __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief          RIPEMD-160 context setup
 *
 * \deprecated     Superseded by mbedtls_ripemd160_starts_ret() in 2.7.0
 *
 * \param ctx      context to be initialized
 */
MBEDTLS_DEPRECATED static inline void mbedtls_ripemd160_starts(
                                            mbedtls_ripemd160_context *ctx )
{
    mbedtls_ripemd160_starts_ret( ctx );
}

/**
 * \brief          RIPEMD-160 process buffer
 *
 * \deprecated     Superseded by mbedtls_ripemd160_update_ret() in 2.7.0
 *
 * \param ctx      RIPEMD-160 context
 * \param input    buffer holding the data
 * \param ilen     length of the input data
 */
MBEDTLS_DEPRECATED static inline void mbedtls_ripemd160_update(
                                                mbedtls_ripemd160_context *ctx,
                                                const unsigned char *input,
                                                size_t ilen )
{
    mbedtls_ripemd160_update_ret( ctx, input, ilen );
}

/**
 * \brief          RIPEMD-160 final digest
 *
 * \deprecated     Superseded by mbedtls_ripemd160_finish_ret() in 2.7.0
 *
 * \param ctx      RIPEMD-160 context
 * \param output   RIPEMD-160 checksum result
 */
MBEDTLS_DEPRECATED static inline void mbedtls_ripemd160_finish(
                                                mbedtls_ripemd160_context *ctx,
                                                unsigned char output[20] )
{
    mbedtls_ripemd160_finish_ret( ctx, output );
}

/**
 * \brief          RIPEMD-160 process data block (internal use only)
 *
 * \deprecated     Superseded by mbedtls_internal_ripemd160_process() in 2.7.0
 *
 * \param ctx      RIPEMD-160 context
 * \param data     buffer holding one block of data
 */
MBEDTLS_DEPRECATED static inline void mbedtls_ripemd160_process(
                                            mbedtls_ripemd160_context *ctx,
                                            const unsigned char data[64] )
{
    mbedtls_internal_ripemd160_process( ctx, data );
}

#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */

#ifdef __cplusplus
}
#endif

#else  /* MBEDTLS_RIPEMD160_ALT */
#include "ripemd160_alt.h"
#endif /* MBEDTLS_RIPEMD160_ALT */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          Output = RIPEMD-160( input buffer )
 *
 * \param input    buffer holding the data
 * \param ilen     length of the input data
 * \param output   RIPEMD-160 checksum result
 *
 * \return         0 if successful
 */
int mbedtls_ripemd160_ret( const unsigned char *input,
                           size_t ilen,
                           unsigned char output[20] );

#if !defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED      __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief          Output = RIPEMD-160( input buffer )
 *
 * \deprecated     Superseded by mbedtls_ripemd160_ret() in 2.7.0
 *
 * \param input    buffer holding the data
 * \param ilen     length of the input data
 * \param output   RIPEMD-160 checksum result
 */
MBEDTLS_DEPRECATED static inline void mbedtls_ripemd160(
                                                    const unsigned char *input,
                                                    size_t ilen,
                                                    unsigned char output[20] )
{
    mbedtls_ripemd160_ret( input, ilen, output );
}

#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */

/**
 * \brief          Checkup routine
 *
 * \return         0 if successful, or 1 if the test failed
 */
int mbedtls_ripemd160_self_test( int verbose );

#ifdef __cplusplus
}
#endif

#endif /* mbedtls_ripemd160.h */
