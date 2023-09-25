/**
 * Translation between MD and PSA identifiers (algorithms, errors).
 *
 *  Note: this internal module will go away when everything becomes based on
 *  PSA Crypto; it is a helper for the transition period.
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef MBEDTLS_MD_PSA_H
#define MBEDTLS_MD_PSA_H

#include "common.h"

#include "mbedtls/md.h"
#include "psa/crypto.h"

/**
 * \brief           This function returns the PSA algorithm identifier
 *                  associated with the given digest type.
 *
 * \param md_type   The type of digest to search for. Must not be NONE.
 *
 * \warning         If \p md_type is \c MBEDTLS_MD_NONE, this function will
 *                  not return \c PSA_ALG_NONE, but an invalid algorithm.
 *
 * \warning         This function does not check if the algorithm is
 *                  supported, it always returns the corresponding identifier.
 *
 * \return          The PSA algorithm identifier associated with \p md_type,
 *                  regardless of whether it is supported or not.
 */
static inline psa_algorithm_t mbedtls_md_psa_alg_from_type(mbedtls_md_type_t md_type)
{
    return PSA_ALG_CATEGORY_HASH | (psa_algorithm_t) md_type;
}

/**
 * \brief           This function returns the given digest type
 *                  associated with the PSA algorithm identifier.
 *
 * \param psa_alg   The PSA algorithm identifier to search for.
 *
 * \warning         This function does not check if the algorithm is
 *                  supported, it always returns the corresponding identifier.
 *
 * \return          The MD type associated with \p psa_alg,
 *                  regardless of whether it is supported or not.
 */
static inline mbedtls_md_type_t mbedtls_md_type_from_psa_alg(psa_algorithm_t psa_alg)
{
    return (mbedtls_md_type_t) (psa_alg & PSA_ALG_HASH_MASK);
}

/** Convert PSA status to MD error code.
 *
 * \param status    PSA status.
 *
 * \return          The corresponding MD error code,
 */
int mbedtls_md_error_from_psa(psa_status_t status);

#endif /* MBEDTLS_MD_PSA_H */
