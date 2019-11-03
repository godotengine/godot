/**
 * \file hmac_drbg.h
 *
 * \brief HMAC_DRBG (NIST SP 800-90A)
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
#ifndef MBEDTLS_HMAC_DRBG_H
#define MBEDTLS_HMAC_DRBG_H

#if !defined(MBEDTLS_CONFIG_FILE)
#include "config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include "md.h"

#if defined(MBEDTLS_THREADING_C)
#include "threading.h"
#endif

/*
 * Error codes
 */
#define MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG              -0x0003  /**< Too many random requested in single call. */
#define MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG                -0x0005  /**< Input too large (Entropy + additional). */
#define MBEDTLS_ERR_HMAC_DRBG_FILE_IO_ERROR                -0x0007  /**< Read/write error in file. */
#define MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED        -0x0009  /**< The entropy source failed. */

/**
 * \name SECTION: Module settings
 *
 * The configuration options you can set for this module are in this section.
 * Either change them in config.h or define them on the compiler command line.
 * \{
 */

#if !defined(MBEDTLS_HMAC_DRBG_RESEED_INTERVAL)
#define MBEDTLS_HMAC_DRBG_RESEED_INTERVAL   10000   /**< Interval before reseed is performed by default */
#endif

#if !defined(MBEDTLS_HMAC_DRBG_MAX_INPUT)
#define MBEDTLS_HMAC_DRBG_MAX_INPUT         256     /**< Maximum number of additional input bytes */
#endif

#if !defined(MBEDTLS_HMAC_DRBG_MAX_REQUEST)
#define MBEDTLS_HMAC_DRBG_MAX_REQUEST       1024    /**< Maximum number of requested bytes per call */
#endif

#if !defined(MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT)
#define MBEDTLS_HMAC_DRBG_MAX_SEED_INPUT    384     /**< Maximum size of (re)seed buffer */
#endif

/* \} name SECTION: Module settings */

#define MBEDTLS_HMAC_DRBG_PR_OFF   0   /**< No prediction resistance       */
#define MBEDTLS_HMAC_DRBG_PR_ON    1   /**< Prediction resistance enabled  */

#ifdef __cplusplus
extern "C" {
#endif

/**
 * HMAC_DRBG context.
 */
typedef struct mbedtls_hmac_drbg_context
{
    /* Working state: the key K is not stored explicitely,
     * but is implied by the HMAC context */
    mbedtls_md_context_t md_ctx;                    /*!< HMAC context (inc. K)  */
    unsigned char V[MBEDTLS_MD_MAX_SIZE];  /*!< V in the spec          */
    int reseed_counter;                     /*!< reseed counter         */

    /* Administrative state */
    size_t entropy_len;         /*!< entropy bytes grabbed on each (re)seed */
    int prediction_resistance;  /*!< enable prediction resistance (Automatic
                                     reseed before every random generation) */
    int reseed_interval;        /*!< reseed interval   */

    /* Callbacks */
    int (*f_entropy)(void *, unsigned char *, size_t); /*!< entropy function */
    void *p_entropy;            /*!< context for the entropy function        */

#if defined(MBEDTLS_THREADING_C)
    mbedtls_threading_mutex_t mutex;
#endif
} mbedtls_hmac_drbg_context;

/**
 * \brief               HMAC_DRBG context initialization
 *                      Makes the context ready for mbedtls_hmac_drbg_seed(),
 *                      mbedtls_hmac_drbg_seed_buf() or
 *                      mbedtls_hmac_drbg_free().
 *
 * \param ctx           HMAC_DRBG context to be initialized
 */
void mbedtls_hmac_drbg_init( mbedtls_hmac_drbg_context *ctx );

/**
 * \brief               HMAC_DRBG initial seeding
 *                      Seed and setup entropy source for future reseeds.
 *
 * \param ctx           HMAC_DRBG context to be seeded
 * \param md_info       MD algorithm to use for HMAC_DRBG
 * \param f_entropy     Entropy callback (p_entropy, buffer to fill, buffer
 *                      length)
 * \param p_entropy     Entropy context
 * \param custom        Personalization data (Device specific identifiers)
 *                      (Can be NULL)
 * \param len           Length of personalization data
 *
 * \note                The "security strength" as defined by NIST is set to:
 *                      128 bits if md_alg is SHA-1,
 *                      192 bits if md_alg is SHA-224,
 *                      256 bits if md_alg is SHA-256 or higher.
 *                      Note that SHA-256 is just as efficient as SHA-224.
 *
 * \return              0 if successful, or
 *                      MBEDTLS_ERR_MD_BAD_INPUT_DATA, or
 *                      MBEDTLS_ERR_MD_ALLOC_FAILED, or
 *                      MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED.
 */
int mbedtls_hmac_drbg_seed( mbedtls_hmac_drbg_context *ctx,
                    const mbedtls_md_info_t * md_info,
                    int (*f_entropy)(void *, unsigned char *, size_t),
                    void *p_entropy,
                    const unsigned char *custom,
                    size_t len );

/**
 * \brief               Initilisation of simpified HMAC_DRBG (never reseeds).
 *                      (For use with deterministic ECDSA.)
 *
 * \param ctx           HMAC_DRBG context to be initialised
 * \param md_info       MD algorithm to use for HMAC_DRBG
 * \param data          Concatenation of entropy string and additional data
 * \param data_len      Length of data in bytes
 *
 * \return              0 if successful, or
 *                      MBEDTLS_ERR_MD_BAD_INPUT_DATA, or
 *                      MBEDTLS_ERR_MD_ALLOC_FAILED.
 */
int mbedtls_hmac_drbg_seed_buf( mbedtls_hmac_drbg_context *ctx,
                        const mbedtls_md_info_t * md_info,
                        const unsigned char *data, size_t data_len );

/**
 * \brief               Enable / disable prediction resistance (Default: Off)
 *
 * Note: If enabled, entropy is used for ctx->entropy_len before each call!
 *       Only use this if you have ample supply of good entropy!
 *
 * \param ctx           HMAC_DRBG context
 * \param resistance    MBEDTLS_HMAC_DRBG_PR_ON or MBEDTLS_HMAC_DRBG_PR_OFF
 */
void mbedtls_hmac_drbg_set_prediction_resistance( mbedtls_hmac_drbg_context *ctx,
                                          int resistance );

/**
 * \brief               Set the amount of entropy grabbed on each reseed
 *                      (Default: given by the security strength, which
 *                      depends on the hash used, see \c mbedtls_hmac_drbg_init() )
 *
 * \param ctx           HMAC_DRBG context
 * \param len           Amount of entropy to grab, in bytes
 */
void mbedtls_hmac_drbg_set_entropy_len( mbedtls_hmac_drbg_context *ctx,
                                size_t len );

/**
 * \brief               Set the reseed interval
 *                      (Default: MBEDTLS_HMAC_DRBG_RESEED_INTERVAL)
 *
 * \param ctx           HMAC_DRBG context
 * \param interval      Reseed interval
 */
void mbedtls_hmac_drbg_set_reseed_interval( mbedtls_hmac_drbg_context *ctx,
                                    int interval );

/**
 * \brief               HMAC_DRBG update state
 *
 * \param ctx           HMAC_DRBG context
 * \param additional    Additional data to update state with, or NULL
 * \param add_len       Length of additional data, or 0
 *
 * \return              \c 0 on success, or an error from the underlying
 *                      hash calculation.
 *
 * \note                Additional data is optional, pass NULL and 0 as second
 *                      third argument if no additional data is being used.
 */
int mbedtls_hmac_drbg_update_ret( mbedtls_hmac_drbg_context *ctx,
                       const unsigned char *additional, size_t add_len );

/**
 * \brief               HMAC_DRBG reseeding (extracts data from entropy source)
 *
 * \param ctx           HMAC_DRBG context
 * \param additional    Additional data to add to state (Can be NULL)
 * \param len           Length of additional data
 *
 * \return              0 if successful, or
 *                      MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED
 */
int mbedtls_hmac_drbg_reseed( mbedtls_hmac_drbg_context *ctx,
                      const unsigned char *additional, size_t len );

/**
 * \brief               HMAC_DRBG generate random with additional update input
 *
 * Note: Automatically reseeds if reseed_counter is reached or PR is enabled.
 *
 * \param p_rng         HMAC_DRBG context
 * \param output        Buffer to fill
 * \param output_len    Length of the buffer
 * \param additional    Additional data to update with (can be NULL)
 * \param add_len       Length of additional data (can be 0)
 *
 * \return              0 if successful, or
 *                      MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED, or
 *                      MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG, or
 *                      MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG.
 */
int mbedtls_hmac_drbg_random_with_add( void *p_rng,
                               unsigned char *output, size_t output_len,
                               const unsigned char *additional,
                               size_t add_len );

/**
 * \brief               HMAC_DRBG generate random
 *
 * Note: Automatically reseeds if reseed_counter is reached or PR is enabled.
 *
 * \param p_rng         HMAC_DRBG context
 * \param output        Buffer to fill
 * \param out_len       Length of the buffer
 *
 * \return              0 if successful, or
 *                      MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED, or
 *                      MBEDTLS_ERR_HMAC_DRBG_REQUEST_TOO_BIG
 */
int mbedtls_hmac_drbg_random( void *p_rng, unsigned char *output, size_t out_len );

/**
 * \brief               Free an HMAC_DRBG context
 *
 * \param ctx           HMAC_DRBG context to free.
 */
void mbedtls_hmac_drbg_free( mbedtls_hmac_drbg_context *ctx );

#if ! defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED    __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief               HMAC_DRBG update state
 *
 * \deprecated          Superseded by mbedtls_hmac_drbg_update_ret()
 *                      in 2.16.0.
 *
 * \param ctx           HMAC_DRBG context
 * \param additional    Additional data to update state with, or NULL
 * \param add_len       Length of additional data, or 0
 *
 * \note                Additional data is optional, pass NULL and 0 as second
 *                      third argument if no additional data is being used.
 */
MBEDTLS_DEPRECATED void mbedtls_hmac_drbg_update(
    mbedtls_hmac_drbg_context *ctx,
    const unsigned char *additional, size_t add_len );
#undef MBEDTLS_DEPRECATED
#endif /* !MBEDTLS_DEPRECATED_REMOVED */

#if defined(MBEDTLS_FS_IO)
/**
 * \brief               Write a seed file
 *
 * \param ctx           HMAC_DRBG context
 * \param path          Name of the file
 *
 * \return              0 if successful, 1 on file error, or
 *                      MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED
 */
int mbedtls_hmac_drbg_write_seed_file( mbedtls_hmac_drbg_context *ctx, const char *path );

/**
 * \brief               Read and update a seed file. Seed is added to this
 *                      instance
 *
 * \param ctx           HMAC_DRBG context
 * \param path          Name of the file
 *
 * \return              0 if successful, 1 on file error,
 *                      MBEDTLS_ERR_HMAC_DRBG_ENTROPY_SOURCE_FAILED or
 *                      MBEDTLS_ERR_HMAC_DRBG_INPUT_TOO_BIG
 */
int mbedtls_hmac_drbg_update_seed_file( mbedtls_hmac_drbg_context *ctx, const char *path );
#endif /* MBEDTLS_FS_IO */


#if defined(MBEDTLS_SELF_TEST)
/**
 * \brief               Checkup routine
 *
 * \return              0 if successful, or 1 if the test failed
 */
int mbedtls_hmac_drbg_self_test( int verbose );
#endif

#ifdef __cplusplus
}
#endif

#endif /* hmac_drbg.h */
