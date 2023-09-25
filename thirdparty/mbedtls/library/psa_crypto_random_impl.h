/** \file psa_crypto_random_impl.h
 *
 * \brief PSA crypto random generator implementation abstraction.
 *
 * The definitions here need to be consistent with the declarations
 * in include/psa_util_internal.h. This file contains some redundant
 * declarations to increase the chance that a compiler will detect
 * inconsistencies if one file is changed without updating the other,
 * but not all potential inconsistencies can be enforced, so make sure
 * to check the public declarations and contracts in
 * include/psa_util_internal.h if you modify this file.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef PSA_CRYPTO_RANDOM_IMPL_H
#define PSA_CRYPTO_RANDOM_IMPL_H

#include "psa_util_internal.h"

#if defined(MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG)

#include <string.h>
#include <mbedtls/entropy.h> // only for error codes
#include <psa/crypto.h>

typedef mbedtls_psa_external_random_context_t mbedtls_psa_random_context_t;

/* Trivial wrapper around psa_generate_random(). */
int mbedtls_psa_get_random(void *p_rng,
                           unsigned char *output,
                           size_t output_size);

/* The PSA RNG API doesn't need any externally maintained state. */
#define MBEDTLS_PSA_RANDOM_STATE NULL

#else /* MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG */

/* Choose a DRBG based on configuration and availability */
#if defined(MBEDTLS_PSA_HMAC_DRBG_MD_TYPE)

#include "mbedtls/hmac_drbg.h"

#elif defined(MBEDTLS_CTR_DRBG_C)

#include "mbedtls/ctr_drbg.h"

#elif defined(MBEDTLS_HMAC_DRBG_C)

#include "mbedtls/hmac_drbg.h"
#if defined(MBEDTLS_MD_CAN_SHA512) && defined(MBEDTLS_MD_CAN_SHA256)
#include <limits.h>
#if SIZE_MAX > 0xffffffff
/* Looks like a 64-bit system, so prefer SHA-512. */
#define MBEDTLS_PSA_HMAC_DRBG_MD_TYPE MBEDTLS_MD_SHA512
#else
/* Looks like a 32-bit system, so prefer SHA-256. */
#define MBEDTLS_PSA_HMAC_DRBG_MD_TYPE MBEDTLS_MD_SHA256
#endif
#elif defined(MBEDTLS_MD_CAN_SHA512)
#define MBEDTLS_PSA_HMAC_DRBG_MD_TYPE MBEDTLS_MD_SHA512
#elif defined(MBEDTLS_MD_CAN_SHA256)
#define MBEDTLS_PSA_HMAC_DRBG_MD_TYPE MBEDTLS_MD_SHA256
#else
#error "No hash algorithm available for HMAC_DBRG."
#endif

#else
#error "No DRBG module available for the psa_crypto module."
#endif

#include "mbedtls/entropy.h"

/** Initialize the PSA DRBG.
 *
 * \param p_rng        Pointer to the Mbed TLS DRBG state.
 */
static inline void mbedtls_psa_drbg_init(mbedtls_psa_drbg_context_t *p_rng)
{
#if defined(MBEDTLS_CTR_DRBG_C)
    mbedtls_ctr_drbg_init(p_rng);
#elif defined(MBEDTLS_HMAC_DRBG_C)
    mbedtls_hmac_drbg_init(p_rng);
#endif
}

/** Deinitialize the PSA DRBG.
 *
 * \param p_rng        Pointer to the Mbed TLS DRBG state.
 */
static inline void mbedtls_psa_drbg_free(mbedtls_psa_drbg_context_t *p_rng)
{
#if defined(MBEDTLS_CTR_DRBG_C)
    mbedtls_ctr_drbg_free(p_rng);
#elif defined(MBEDTLS_HMAC_DRBG_C)
    mbedtls_hmac_drbg_free(p_rng);
#endif
}

/** The type of the PSA random generator context.
 *
 * The random generator context is composed of an entropy context and
 * a DRBG context.
 */
typedef struct {
    void (* entropy_init)(mbedtls_entropy_context *ctx);
    void (* entropy_free)(mbedtls_entropy_context *ctx);
    mbedtls_entropy_context entropy;
    mbedtls_psa_drbg_context_t drbg;
} mbedtls_psa_random_context_t;

/* Defined in include/psa_util_internal.h so that it's visible to
 * application code. The declaration here is redundant, but included
 * as a safety net to make it more likely that a future change that
 * accidentally causes the implementation to diverge from the interface
 * will be noticed. */
/* Do not include the declaration under MSVC because it doesn't accept it
 * ("error C2370: 'mbedtls_psa_get_random' : redefinition; different storage class").
 * Observed with Visual Studio 2013. A known bug apparently:
 * https://stackoverflow.com/questions/8146541/duplicate-external-static-declarations-not-allowed-in-visual-studio
 */
#if !defined(_MSC_VER)
static mbedtls_f_rng_t *const mbedtls_psa_get_random;
#endif

/** The maximum number of bytes that mbedtls_psa_get_random() is expected to
 * return.
 */
#if defined(MBEDTLS_CTR_DRBG_C)
#define MBEDTLS_PSA_RANDOM_MAX_REQUEST MBEDTLS_CTR_DRBG_MAX_REQUEST
#elif defined(MBEDTLS_HMAC_DRBG_C)
#define MBEDTLS_PSA_RANDOM_MAX_REQUEST MBEDTLS_HMAC_DRBG_MAX_REQUEST
#endif

/** A pointer to the PSA DRBG state.
 *
 * This variable is only intended to be used through the macro
 * #MBEDTLS_PSA_RANDOM_STATE.
 */
/* psa_crypto.c sets this variable to a pointer to the DRBG state in the
 * global PSA crypto state. */
/* The type `mbedtls_psa_drbg_context_t` is defined in
 * include/psa_util_internal.h so that `mbedtls_psa_random_state` can be
 * declared there and be visible to application code. */
extern mbedtls_psa_drbg_context_t *const mbedtls_psa_random_state;

/** A pointer to the PSA DRBG state.
 *
 * This macro expands to an expression that is suitable as the \c p_rng
 * parameter to pass to mbedtls_psa_get_random().
 *
 * This macro exists in all configurations where the psa_crypto module is
 * enabled. Its expansion depends on the configuration.
 */
#define MBEDTLS_PSA_RANDOM_STATE mbedtls_psa_random_state

/** Seed the PSA DRBG.
 *
 * \param entropy       An entropy context to read the seed from.
 * \param custom        The personalization string.
 *                      This can be \c NULL, in which case the personalization
 *                      string is empty regardless of the value of \p len.
 * \param len           The length of the personalization string.
 *
 * \return              \c 0 on success.
 * \return              An Mbed TLS error code (\c MBEDTLS_ERR_xxx) on failure.
 */
static inline int mbedtls_psa_drbg_seed(
    mbedtls_entropy_context *entropy,
    const unsigned char *custom, size_t len)
{
#if defined(MBEDTLS_CTR_DRBG_C)
    return mbedtls_ctr_drbg_seed(MBEDTLS_PSA_RANDOM_STATE,
                                 mbedtls_entropy_func,
                                 entropy,
                                 custom, len);
#elif defined(MBEDTLS_HMAC_DRBG_C)
    const mbedtls_md_info_t *md_info =
        mbedtls_md_info_from_type(MBEDTLS_PSA_HMAC_DRBG_MD_TYPE);
    return mbedtls_hmac_drbg_seed(MBEDTLS_PSA_RANDOM_STATE,
                                  md_info,
                                  mbedtls_entropy_func,
                                  entropy,
                                  custom, len);
#endif
}

#endif /* MBEDTLS_PSA_CRYPTO_EXTERNAL_RNG */

#endif /* PSA_CRYPTO_RANDOM_IMPL_H */
