/**
 * \file ctr_drbg.h
 *
 * \brief    This file contains definitions and functions for the
 *           CTR_DRBG pseudorandom generator.
 *
 * CTR_DRBG is a standardized way of building a PRNG from a block-cipher
 * in counter mode operation, as defined in <em>NIST SP 800-90A:
 * Recommendation for Random Number Generation Using Deterministic Random
 * Bit Generators</em>.
 *
 * The Mbed TLS implementation of CTR_DRBG uses AES-256 (default) or AES-128
 * (if \c MBEDTLS_CTR_DRBG_USE_128_BIT_KEY is enabled at compile time)
 * as the underlying block cipher, with a derivation function.
 * The initial seeding grabs #MBEDTLS_CTR_DRBG_ENTROPY_LEN bytes of entropy.
 * See the documentation of mbedtls_ctr_drbg_seed() for more details.
 *
 * Based on NIST SP 800-90A §10.2.1 table 3 and NIST SP 800-57 part 1 table 2,
 * here are the security strengths achieved in typical configuration:
 * - 256 bits under the default configuration of the library, with AES-256
 *   and with #MBEDTLS_CTR_DRBG_ENTROPY_LEN set to 48 or more.
 * - 256 bits if AES-256 is used, #MBEDTLS_CTR_DRBG_ENTROPY_LEN is set
 *   to 32 or more, and the DRBG is initialized with an explicit
 *   nonce in the \c custom parameter to mbedtls_ctr_drbg_seed().
 * - 128 bits if AES-256 is used but #MBEDTLS_CTR_DRBG_ENTROPY_LEN is
 *   between 24 and 47 and the DRBG is not initialized with an explicit
 *   nonce (see mbedtls_ctr_drbg_seed()).
 * - 128 bits if AES-128 is used (\c MBEDTLS_CTR_DRBG_USE_128_BIT_KEY enabled)
 *   and #MBEDTLS_CTR_DRBG_ENTROPY_LEN is set to 24 or more (which is
 *   always the case unless it is explicitly set to a different value
 *   in config.h).
 *
 * Note that the value of #MBEDTLS_CTR_DRBG_ENTROPY_LEN defaults to:
 * - \c 48 if the module \c MBEDTLS_SHA512_C is enabled and the symbol
 *   \c MBEDTLS_ENTROPY_FORCE_SHA256 is disabled at compile time.
 *   This is the default configuration of the library.
 * - \c 32 if the module \c MBEDTLS_SHA512_C is disabled at compile time.
 * - \c 32 if \c MBEDTLS_ENTROPY_FORCE_SHA256 is enabled at compile time.
 */
/*
 *  Copyright (C) 2006-2019, Arm Limited (or its affiliates), All Rights Reserved
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
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */

#ifndef MBEDTLS_CTR_DRBG_H
#define MBEDTLS_CTR_DRBG_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "aes.h"

#if defined(MBEDTLS_THREADING_C)
#include "threading.h"
#endif

#define MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED        -0x0034  /**< The entropy source failed. */
#define MBEDTLS_ERR_CTR_DRBG_REQUEST_TOO_BIG              -0x0036  /**< The requested random buffer length is too big. */
#define MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG                -0x0038  /**< The input (entropy + additional data) is too large. */
#define MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR                -0x003A  /**< Read or write error in file. */

#define MBEDTLS_CTR_DRBG_BLOCKSIZE          16 /**< The block size used by the cipher. */

#if defined(MBEDTLS_CTR_DRBG_USE_128_BIT_KEY)
#define MBEDTLS_CTR_DRBG_KEYSIZE            16
/**< The key size in bytes used by the cipher.
 *
 * Compile-time choice: 16 bytes (128 bits)
 * because #MBEDTLS_CTR_DRBG_USE_128_BIT_KEY is enabled.
 */
#else
#define MBEDTLS_CTR_DRBG_KEYSIZE            32
/**< The key size in bytes used by the cipher.
 *
 * Compile-time choice: 32 bytes (256 bits)
 * because \c MBEDTLS_CTR_DRBG_USE_128_BIT_KEY is disabled.
 */
#endif

#define MBEDTLS_CTR_DRBG_KEYBITS            ( MBEDTLS_CTR_DRBG_KEYSIZE * 8 ) /**< The key size for the DRBG operation, in bits. */
#define MBEDTLS_CTR_DRBG_SEEDLEN            ( MBEDTLS_CTR_DRBG_KEYSIZE + MBEDTLS_CTR_DRBG_BLOCKSIZE ) /**< The seed length, calculated as (counter + AES key). */

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h or define them using the compiler command
 * line.
 * \{
 */

/** \def MBEDTLS_CTR_DRBG_ENTROPY_LEN
 *
 * \brief The amount of entropy used per seed by default, in bytes.
 */
#if !defined(MBEDTLS_CTR_DRBG_ENTROPY_LEN)
#if defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_ENTROPY_FORCE_SHA256)
/** This is 48 bytes because the entropy module uses SHA-512
 * (\c MBEDTLS_ENTROPY_FORCE_SHA256 is disabled).
 */
#define MBEDTLS_CTR_DRBG_ENTROPY_LEN        48

#else /* defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_ENTROPY_FORCE_SHA256) */

/** This is 32 bytes because the entropy module uses SHA-256
 * (the SHA512 module is disabled or
 * \c MBEDTLS_ENTROPY_FORCE_SHA256 is enabled).
 */
#if !defined(MBEDTLS_CTR_DRBG_USE_128_BIT_KEY)
/** \warning To achieve a 256-bit security strength, you must pass a nonce
 *           to mbedtls_ctr_drbg_seed().
 */
#endif /* !defined(MBEDTLS_CTR_DRBG_USE_128_BIT_KEY) */
#define MBEDTLS_CTR_DRBG_ENTROPY_LEN        32
#endif /* defined(MBEDTLS_SHA512_C) && !defined(MBEDTLS_ENTROPY_FORCE_SHA256) */
#endif /* !defined(MBEDTLS_CTR_DRBG_ENTROPY_LEN) */

#if !defined(MBEDTLS_CTR_DRBG_RESEED_INTERVAL)
#define MBEDTLS_CTR_DRBG_RESEED_INTERVAL    10000
/**< The interval before reseed is performed by default. */
#endif

#if !defined(MBEDTLS_CTR_DRBG_MAX_INPUT)
#define MBEDTLS_CTR_DRBG_MAX_INPUT          256
/**< The maximum number of additional input Bytes. */
#endif

#if !defined(MBEDTLS_CTR_DRBG_MAX_REQUEST)
#define MBEDTLS_CTR_DRBG_MAX_REQUEST        1024
/**< The maximum number of requested Bytes per call. */
#endif

#if !defined(MBEDTLS_CTR_DRBG_MAX_SEED_INPUT)
#define MBEDTLS_CTR_DRBG_MAX_SEED_INPUT     384
/**< The maximum size of seed or reseed buffer in bytes. */
#endif

/* \} name SECTION: Module settings */

#define MBEDTLS_CTR_DRBG_PR_OFF             0
/**< Prediction resistance is disabled. */
#define MBEDTLS_CTR_DRBG_PR_ON              1
/**< Prediction resistance is enabled. */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief          The CTR_DRBG context structure.
 */
typedef struct mbedtls_ctr_drbg_context
{
    unsigned char counter[16];  /*!< The counter (V). */
    int reseed_counter;         /*!< The reseed counter. */
    int prediction_resistance;  /*!< This determines whether prediction
                                     resistance is enabled, that is
                                     whether to systematically reseed before
                                     each random generation. */
    size_t entropy_len;         /*!< The amount of entropy grabbed on each
                                     seed or reseed operation. */
    int reseed_interval;        /*!< The reseed interval. */

    mbedtls_aes_context aes_ctx;        /*!< The AES context. */

    /*
     * Callbacks (Entropy)
     */
    int (*f_entropy)(void *, unsigned char *, size_t);
                                /*!< The entropy callback function. */

    void *p_entropy;            /*!< The context for the entropy function. */

#if defined(MBEDTLS_THREADING_C)
    mbedtls_threading_mutex_t mutex;
#endif
}
mbedtls_ctr_drbg_context;

/**
 * \brief               This function initializes the CTR_DRBG context,
 *                      and prepares it for mbedtls_ctr_drbg_seed()
 *                      or mbedtls_ctr_drbg_free().
 *
 * \param ctx           The CTR_DRBG context to initialize.
 */
void mbedtls_ctr_drbg_init( mbedtls_ctr_drbg_context *ctx );

/**
 * \brief               This function seeds and sets up the CTR_DRBG
 *                      entropy source for future reseeds.
 *
 * A typical choice for the \p f_entropy and \p p_entropy parameters is
 * to use the entropy module:
 * - \p f_entropy is mbedtls_entropy_func();
 * - \p p_entropy is an instance of ::mbedtls_entropy_context initialized
 *   with mbedtls_entropy_init() (which registers the platform's default
 *   entropy sources).
 *
 * The entropy length is #MBEDTLS_CTR_DRBG_ENTROPY_LEN by default.
 * You can override it by calling mbedtls_ctr_drbg_set_entropy_len().
 *
 * You can provide a personalization string in addition to the
 * entropy source, to make this instantiation as unique as possible.
 *
 * \note                The _seed_material_ value passed to the derivation
 *                      function in the CTR_DRBG Instantiate Process
 *                      described in NIST SP 800-90A §10.2.1.3.2
 *                      is the concatenation of the string obtained from
 *                      calling \p f_entropy and the \p custom string.
 *                      The origin of the nonce depends on the value of
 *                      the entropy length relative to the security strength.
 *                      - If the entropy length is at least 1.5 times the
 *                        security strength then the nonce is taken from the
 *                        string obtained with \p f_entropy.
 *                      - If the entropy length is less than the security
 *                        strength, then the nonce is taken from \p custom.
 *                        In this case, for compliance with SP 800-90A,
 *                        you must pass a unique value of \p custom at
 *                        each invocation. See SP 800-90A §8.6.7 for more
 *                        details.
 */
#if MBEDTLS_CTR_DRBG_ENTROPY_LEN < MBEDTLS_CTR_DRBG_KEYSIZE * 3 / 2
/** \warning            When #MBEDTLS_CTR_DRBG_ENTROPY_LEN is less than
 *                      #MBEDTLS_CTR_DRBG_KEYSIZE * 3 / 2, to achieve the
 *                      maximum security strength permitted by CTR_DRBG,
 *                      you must pass a value of \p custom that is a nonce:
 *                      this value must never be repeated in subsequent
 *                      runs of the same application or on a different
 *                      device.
 */
#endif
/**
 * \param ctx           The CTR_DRBG context to seed.
 *                      It must have been initialized with
 *                      mbedtls_ctr_drbg_init().
 *                      After a successful call to mbedtls_ctr_drbg_seed(),
 *                      you may not call mbedtls_ctr_drbg_seed() again on
 *                      the same context unless you call
 *                      mbedtls_ctr_drbg_free() and mbedtls_ctr_drbg_init()
 *                      again first.
 * \param f_entropy     The entropy callback, taking as arguments the
 *                      \p p_entropy context, the buffer to fill, and the
 *                      length of the buffer.
 *                      \p f_entropy is always called with a buffer size
 *                      equal to the entropy length.
 * \param p_entropy     The entropy context to pass to \p f_entropy.
 * \param custom        The personalization string.
 *                      This can be \c NULL, in which case the personalization
 *                      string is empty regardless of the value of \p len.
 * \param len           The length of the personalization string.
 *                      This must be at most
 *                      #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT
 *                      - #MBEDTLS_CTR_DRBG_ENTROPY_LEN.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED on failure.
 */
int mbedtls_ctr_drbg_seed( mbedtls_ctr_drbg_context *ctx,
                   int (*f_entropy)(void *, unsigned char *, size_t),
                   void *p_entropy,
                   const unsigned char *custom,
                   size_t len );

/**
 * \brief               This function clears CTR_CRBG context data.
 *
 * \param ctx           The CTR_DRBG context to clear.
 */
void mbedtls_ctr_drbg_free( mbedtls_ctr_drbg_context *ctx );

/**
 * \brief               This function turns prediction resistance on or off.
 *                      The default value is off.
 *
 * \note                If enabled, entropy is gathered at the beginning of
 *                      every call to mbedtls_ctr_drbg_random_with_add()
 *                      or mbedtls_ctr_drbg_random().
 *                      Only use this if your entropy source has sufficient
 *                      throughput.
 *
 * \param ctx           The CTR_DRBG context.
 * \param resistance    #MBEDTLS_CTR_DRBG_PR_ON or #MBEDTLS_CTR_DRBG_PR_OFF.
 */
void mbedtls_ctr_drbg_set_prediction_resistance( mbedtls_ctr_drbg_context *ctx,
                                         int resistance );

/**
 * \brief               This function sets the amount of entropy grabbed on each
 *                      seed or reseed.
 *
 * The default value is #MBEDTLS_CTR_DRBG_ENTROPY_LEN.
 *
 * \note                The security strength of CTR_DRBG is bounded by the
 *                      entropy length. Thus:
 *                      - When using AES-256
 *                        (\c MBEDTLS_CTR_DRBG_USE_128_BIT_KEY is disabled,
 *                        which is the default),
 *                        \p len must be at least 32 (in bytes)
 *                        to achieve a 256-bit strength.
 *                      - When using AES-128
 *                        (\c MBEDTLS_CTR_DRBG_USE_128_BIT_KEY is enabled)
 *                        \p len must be at least 16 (in bytes)
 *                        to achieve a 128-bit strength.
 *
 * \param ctx           The CTR_DRBG context.
 * \param len           The amount of entropy to grab, in bytes.
 *                      This must be at most #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT.
 */
void mbedtls_ctr_drbg_set_entropy_len( mbedtls_ctr_drbg_context *ctx,
                               size_t len );

/**
 * \brief               This function sets the reseed interval.
 *
 * The reseed interval is the number of calls to mbedtls_ctr_drbg_random()
 * or mbedtls_ctr_drbg_random_with_add() after which the entropy function
 * is called again.
 *
 * The default value is #MBEDTLS_CTR_DRBG_RESEED_INTERVAL.
 *
 * \param ctx           The CTR_DRBG context.
 * \param interval      The reseed interval.
 */
void mbedtls_ctr_drbg_set_reseed_interval( mbedtls_ctr_drbg_context *ctx,
                                   int interval );

/**
 * \brief               This function reseeds the CTR_DRBG context, that is
 *                      extracts data from the entropy source.
 *
 * \param ctx           The CTR_DRBG context.
 * \param additional    Additional data to add to the state. Can be \c NULL.
 * \param len           The length of the additional data.
 *                      This must be less than
 *                      #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT - \c entropy_len
 *                      where \c entropy_len is the entropy length
 *                      configured for the context.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED on failure.
 */
int mbedtls_ctr_drbg_reseed( mbedtls_ctr_drbg_context *ctx,
                     const unsigned char *additional, size_t len );

/**
 * \brief              This function updates the state of the CTR_DRBG context.
 *
 * \param ctx          The CTR_DRBG context.
 * \param additional   The data to update the state with. This must not be
 *                     \c NULL unless \p add_len is \c 0.
 * \param add_len      Length of \p additional in bytes. This must be at
 *                     most #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT.
 *
 * \return             \c 0 on success.
 * \return             #MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG if
 *                     \p add_len is more than
 *                     #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT.
 * \return             An error from the underlying AES cipher on failure.
 */
int mbedtls_ctr_drbg_update_ret( mbedtls_ctr_drbg_context *ctx,
                                 const unsigned char *additional,
                                 size_t add_len );

/**
 * \brief   This function updates a CTR_DRBG instance with additional
 *          data and uses it to generate random data.
 *
 * This function automatically reseeds if the reseed counter is exceeded
 * or prediction resistance is enabled.
 *
 * \param p_rng         The CTR_DRBG context. This must be a pointer to a
 *                      #mbedtls_ctr_drbg_context structure.
 * \param output        The buffer to fill.
 * \param output_len    The length of the buffer in bytes.
 * \param additional    Additional data to update. Can be \c NULL, in which
 *                      case the additional data is empty regardless of
 *                      the value of \p add_len.
 * \param add_len       The length of the additional data
 *                      if \p additional is not \c NULL.
 *                      This must be less than #MBEDTLS_CTR_DRBG_MAX_INPUT
 *                      and less than
 *                      #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT - \c entropy_len
 *                      where \c entropy_len is the entropy length
 *                      configured for the context.
 *
 * \return    \c 0 on success.
 * \return    #MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED or
 *            #MBEDTLS_ERR_CTR_DRBG_REQUEST_TOO_BIG on failure.
 */
int mbedtls_ctr_drbg_random_with_add( void *p_rng,
                              unsigned char *output, size_t output_len,
                              const unsigned char *additional, size_t add_len );

/**
 * \brief   This function uses CTR_DRBG to generate random data.
 *
 * This function automatically reseeds if the reseed counter is exceeded
 * or prediction resistance is enabled.
 *
 *
 * \param p_rng         The CTR_DRBG context. This must be a pointer to a
 *                      #mbedtls_ctr_drbg_context structure.
 * \param output        The buffer to fill.
 * \param output_len    The length of the buffer in bytes.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED or
 *                      #MBEDTLS_ERR_CTR_DRBG_REQUEST_TOO_BIG on failure.
 */
int mbedtls_ctr_drbg_random( void *p_rng,
                     unsigned char *output, size_t output_len );


#if ! defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED    __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief              This function updates the state of the CTR_DRBG context.
 *
 * \deprecated         Superseded by mbedtls_ctr_drbg_update_ret()
 *                     in 2.16.0.
 *
 * \note               If \p add_len is greater than
 *                     #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT, only the first
 *                     #MBEDTLS_CTR_DRBG_MAX_SEED_INPUT Bytes are used.
 *                     The remaining Bytes are silently discarded.
 *
 * \param ctx          The CTR_DRBG context.
 * \param additional   The data to update the state with.
 * \param add_len      Length of \p additional data.
 */
MBEDTLS_DEPRECATED void mbedtls_ctr_drbg_update(
    mbedtls_ctr_drbg_context *ctx,
    const unsigned char *additional,
    size_t add_len );
#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */

#if defined(MBEDTLS_FS_IO)
/**
 * \brief               This function writes a seed file.
 *
 * \param ctx           The CTR_DRBG context.
 * \param path          The name of the file.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR on file error.
 * \return              #MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED on reseed
 *                      failure.
 */
int mbedtls_ctr_drbg_write_seed_file( mbedtls_ctr_drbg_context *ctx, const char *path );

/**
 * \brief               This function reads and updates a seed file. The seed
 *                      is added to this instance.
 *
 * \param ctx           The CTR_DRBG context.
 * \param path          The name of the file.
 *
 * \return              \c 0 on success.
 * \return              #MBEDTLS_ERR_CTR_DRBG_FILE_IO_ERROR on file error.
 * \return              #MBEDTLS_ERR_CTR_DRBG_ENTROPY_SOURCE_FAILED on
 *                      reseed failure.
 * \return              #MBEDTLS_ERR_CTR_DRBG_INPUT_TOO_BIG if the existing
 *                      seed file is too large.
 */
int mbedtls_ctr_drbg_update_seed_file( mbedtls_ctr_drbg_context *ctx, const char *path );
#endif /* MBEDTLS_FS_IO */

#if defined(MBEDTLS_SELF_TEST)

/**
 * \brief               The CTR_DRBG checkup routine.
 *
 * \return              \c 0 on success.
 * \return              \c 1 on failure.
 */
int mbedtls_ctr_drbg_self_test( int verbose );

#endif /* MBEDTLS_SELF_TEST */

/* Internal functions (do not call directly) */
int mbedtls_ctr_drbg_seed_entropy_len( mbedtls_ctr_drbg_context *,
                               int (*)(void *, unsigned char *, size_t), void *,
                               const unsigned char *, size_t, size_t );

#ifdef __cplusplus
}
#endif

#endif /* ctr_drbg.h */
