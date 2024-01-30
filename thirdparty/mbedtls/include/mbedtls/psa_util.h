/**
 * \file psa_util.h
 *
 * \brief Utility functions for the use of the PSA Crypto library.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_PSA_UTIL_H
#define MBEDTLS_PSA_UTIL_H
#include "mbedtls/private_access.h"

#include "mbedtls/build_info.h"

#if defined(MBEDTLS_PSA_CRYPTO_C)

/* Expose whatever RNG the PSA subsystem uses to applications using the
 * mbedtls_xxx API. The declarations and definitions here need to be
 * consistent with the implementation in library/psa_crypto_random_impl.h.
 * See that file for implementation documentation. */


/* The type of a `f_rng` random generator function that many library functions
 * take.
 *
 * This type name is not part of the Mbed TLS stable API. It may be renamed
 * or moved without warning.
 */
typedef int mbedtls_f_rng_t(void *p_rng, unsigned char *output, size_t output_size);

#if defined(MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG)

/** The random generator function for the PSA subsystem.
 *
 * This function is suitable as the `f_rng` random generator function
 * parameter of many `mbedtls_xxx` functions. Use #MBEDTLS_PSA_RANDOM_STATE
 * to obtain the \p p_rng parameter.
 *
 * The implementation of this function depends on the configuration of the
 * library.
 *
 * \note Depending on the configuration, this may be a function or
 *       a pointer to a function.
 *
 * \note This function may only be used if the PSA crypto subsystem is active.
 *       This means that you must call psa_crypto_init() before any call to
 *       this function, and you must not call this function after calling
 *       mbedtls_psa_crypto_free().
 *
 * \param p_rng         The random generator context. This must be
 *                      #MBEDTLS_PSA_RANDOM_STATE. No other state is
 *                      supported.
 * \param output        The buffer to fill. It must have room for
 *                      \c output_size bytes.
 * \param output_size   The number of bytes to write to \p output.
 *                      This function may fail if \p output_size is too
 *                      large. It is guaranteed to accept any output size
 *                      requested by Mbed TLS library functions. The
 *                      maximum request size depends on the library
 *                      configuration.
 *
 * \return              \c 0 on success.
 * \return              An `MBEDTLS_ERR_ENTROPY_xxx`,
 *                      `MBEDTLS_ERR_PLATFORM_xxx,
 *                      `MBEDTLS_ERR_CTR_DRBG_xxx` or
 *                      `MBEDTLS_ERR_HMAC_DRBG_xxx` on error.
 */
int mbedtls_psa_get_random(void *p_rng,
                           unsigned char *output,
                           size_t output_size);

/** The random generator state for the PSA subsystem.
 *
 * This macro expands to an expression which is suitable as the `p_rng`
 * random generator state parameter of many `mbedtls_xxx` functions.
 * It must be used in combination with the random generator function
 * mbedtls_psa_get_random().
 *
 * The implementation of this macro depends on the configuration of the
 * library. Do not make any assumption on its nature.
 */
#define MBEDTLS_PSA_RANDOM_STATE NULL

#else /* !defined(MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG) */

#if defined(MBEDTLS_CTR_DRBG_C)
#include "mbedtls/ctr_drbg.h"
typedef mbedtls_ctr_drbg_context mbedtls_psa_drbg_context_t;
static mbedtls_f_rng_t *const mbedtls_psa_get_random = mbedtls_ctr_drbg_random;
#elif defined(MBEDTLS_HMAC_DRBG_C)
#include "mbedtls/hmac_drbg.h"
typedef mbedtls_hmac_drbg_context mbedtls_psa_drbg_context_t;
static mbedtls_f_rng_t *const mbedtls_psa_get_random = mbedtls_hmac_drbg_random;
#endif
extern mbedtls_psa_drbg_context_t *const mbedtls_psa_random_state;

#define MBEDTLS_PSA_RANDOM_STATE mbedtls_psa_random_state

#endif /* !defined(MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG) */

#endif /* MBEDTLS_PSA_CRYPTO_C */
#endif /* MBEDTLS_PSA_UTIL_H */
