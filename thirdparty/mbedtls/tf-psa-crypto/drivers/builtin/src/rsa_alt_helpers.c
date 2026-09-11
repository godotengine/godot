/*
 *  Helper functions for the RSA module
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 *
 */

#include "tf_psa_crypto_common.h"

#if defined(MBEDTLS_RSA_C)

#include "mbedtls/private/rsa.h"
#include "mbedtls/private/bignum.h"
#include "bignum_internal.h"
#include "rsa_alt_helpers.h"

/*
 * Given P, Q and the public exponent E, deduce D.
 * This is essentially a modular inversion.
 */
int mbedtls_rsa_deduce_private_exponent(mbedtls_mpi const *P,
                                        mbedtls_mpi const *Q,
                                        mbedtls_mpi const *E,
                                        mbedtls_mpi *D)
{
    int ret = 0;
    mbedtls_mpi K, L;

    if (D == NULL) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    if (mbedtls_mpi_cmp_int(P, 1) <= 0 ||
        mbedtls_mpi_cmp_int(Q, 1) <= 0 ||
        mbedtls_mpi_cmp_int(E, 0) == 0) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    if (mbedtls_mpi_get_bit(E, 0) != 1) {
        return MBEDTLS_ERR_MPI_NOT_ACCEPTABLE;
    }

    mbedtls_mpi_init(&K);
    mbedtls_mpi_init(&L);

    /* Temporarily put K := P-1 and L := Q-1 */
    MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, P, 1));
    MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&L, Q, 1));

    /* Temporarily put D := gcd(P-1, Q-1) */
    MBEDTLS_MPI_CHK(mbedtls_mpi_gcd(D, &K, &L));

    /* K := LCM(P-1, Q-1) */
    MBEDTLS_MPI_CHK(mbedtls_mpi_mul_mpi(&K, &K, &L));
    MBEDTLS_MPI_CHK(mbedtls_mpi_div_mpi(&K, NULL, &K, D));

    /* Compute modular inverse of E mod LCM(P-1, Q-1)
     * This is FIPS 186-4 §B.3.1 criterion 3(b).
     * This will return MBEDTLS_ERR_MPI_NOT_ACCEPTABLE if E is not coprime to
     * (P-1)(Q-1), also validating FIPS 186-4 §B.3.1 criterion 2(a). */
    MBEDTLS_MPI_CHK(mbedtls_mpi_inv_mod_even_in_range(D, E, &K));

cleanup:

    mbedtls_mpi_free(&K);
    mbedtls_mpi_free(&L);

    return ret;
}

int mbedtls_rsa_deduce_crt(const mbedtls_mpi *P, const mbedtls_mpi *Q,
                           const mbedtls_mpi *D, mbedtls_mpi *DP,
                           mbedtls_mpi *DQ, mbedtls_mpi *QP)
{
    int ret = 0;
    mbedtls_mpi K;
    mbedtls_mpi_init(&K);

    /* DP = D mod P-1 */
    if (DP != NULL) {
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, P, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(DP, D, &K));
    }

    /* DQ = D mod Q-1 */
    if (DQ != NULL) {
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, Q, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(DQ, D, &K));
    }

    /* QP = Q^{-1} mod P */
    if (QP != NULL) {
        MBEDTLS_MPI_CHK(mbedtls_mpi_inv_mod_odd(QP, Q, P));
    }

cleanup:
    mbedtls_mpi_free(&K);

    return ret;
}

/*
 * Check that core RSA parameters are sane.
 */
int mbedtls_rsa_validate_params(const mbedtls_mpi *N, const mbedtls_mpi *P,
                                const mbedtls_mpi *Q, const mbedtls_mpi *D,
                                const mbedtls_mpi *E,
                                int (*f_rng)(void *, unsigned char *, size_t),
                                void *p_rng)
{
    int ret = 0;
    mbedtls_mpi K, L;

    mbedtls_mpi_init(&K);
    mbedtls_mpi_init(&L);

    /*
     * Step 1: If PRNG provided, check that P and Q are prime
     */

#if defined(MBEDTLS_GENPRIME)
    /*
     * When generating keys, the strongest security we support aims for an error
     * rate of at most 2^-100 and we are aiming for the same certainty here as
     * well.
     */
    if (f_rng != NULL && P != NULL &&
        (ret = mbedtls_mpi_is_prime_ext(P, 50, f_rng, p_rng)) != 0) {
        ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
        goto cleanup;
    }

    if (f_rng != NULL && Q != NULL &&
        (ret = mbedtls_mpi_is_prime_ext(Q, 50, f_rng, p_rng)) != 0) {
        ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
        goto cleanup;
    }
#else
    ((void) f_rng);
    ((void) p_rng);
#endif /* MBEDTLS_GENPRIME */

    /*
     * Step 2: Check that 1 < N = P * Q
     */

    if (P != NULL && Q != NULL && N != NULL) {
        MBEDTLS_MPI_CHK(mbedtls_mpi_mul_mpi(&K, P, Q));
        if (mbedtls_mpi_cmp_int(N, 1)  <= 0 ||
            mbedtls_mpi_cmp_mpi(&K, N) != 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }
    }

    /*
     * Step 3: Check and 1 < D, E < N if present.
     */

    if (N != NULL && D != NULL && E != NULL) {
        if (mbedtls_mpi_cmp_int(D, 1) <= 0 ||
            mbedtls_mpi_cmp_int(E, 1) <= 0 ||
            mbedtls_mpi_cmp_mpi(D, N) >= 0 ||
            mbedtls_mpi_cmp_mpi(E, N) >= 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }
    }

    /*
     * Step 4: Check that D, E are inverse modulo P-1 and Q-1
     */

    if (P != NULL && Q != NULL && D != NULL && E != NULL) {
        if (mbedtls_mpi_cmp_int(P, 1) <= 0 ||
            mbedtls_mpi_cmp_int(Q, 1) <= 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }

        /* Compute DE-1 mod P-1 */
        MBEDTLS_MPI_CHK(mbedtls_mpi_mul_mpi(&K, D, E));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, &K, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&L, P, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(&K, &K, &L));
        if (mbedtls_mpi_cmp_int(&K, 0) != 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }

        /* Compute DE-1 mod Q-1 */
        MBEDTLS_MPI_CHK(mbedtls_mpi_mul_mpi(&K, D, E));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, &K, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&L, Q, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(&K, &K, &L));
        if (mbedtls_mpi_cmp_int(&K, 0) != 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }
    }

cleanup:

    mbedtls_mpi_free(&K);
    mbedtls_mpi_free(&L);

    /* Wrap MPI error codes by RSA check failure error code */
    if (ret != 0 && ret != MBEDTLS_ERR_RSA_KEY_CHECK_FAILED) {
        ret += MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
    }

    return ret;
}

/*
 * Check that RSA CRT parameters are in accordance with core parameters.
 */
int mbedtls_rsa_validate_crt(const mbedtls_mpi *P,  const mbedtls_mpi *Q,
                             const mbedtls_mpi *D,  const mbedtls_mpi *DP,
                             const mbedtls_mpi *DQ, const mbedtls_mpi *QP)
{
    int ret = 0;

    mbedtls_mpi K, L;
    mbedtls_mpi_init(&K);
    mbedtls_mpi_init(&L);

    /* Check that DP - D == 0 mod P - 1 */
    if (DP != NULL) {
        if (P == NULL) {
            ret = MBEDTLS_ERR_RSA_BAD_INPUT_DATA;
            goto cleanup;
        }

        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, P, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_mpi(&L, DP, D));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(&L, &L, &K));

        if (mbedtls_mpi_cmp_int(&L, 0) != 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }
    }

    /* Check that DQ - D == 0 mod Q - 1 */
    if (DQ != NULL) {
        if (Q == NULL) {
            ret = MBEDTLS_ERR_RSA_BAD_INPUT_DATA;
            goto cleanup;
        }

        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, Q, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_mpi(&L, DQ, D));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(&L, &L, &K));

        if (mbedtls_mpi_cmp_int(&L, 0) != 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }
    }

    /* Check that QP * Q - 1 == 0 mod P */
    if (QP != NULL) {
        if (P == NULL || Q == NULL) {
            ret = MBEDTLS_ERR_RSA_BAD_INPUT_DATA;
            goto cleanup;
        }

        MBEDTLS_MPI_CHK(mbedtls_mpi_mul_mpi(&K, QP, Q));
        MBEDTLS_MPI_CHK(mbedtls_mpi_sub_int(&K, &K, 1));
        MBEDTLS_MPI_CHK(mbedtls_mpi_mod_mpi(&K, &K, P));
        if (mbedtls_mpi_cmp_int(&K, 0) != 0) {
            ret = MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
            goto cleanup;
        }
    }

cleanup:

    /* Wrap MPI error codes by RSA check failure error code */
    if (ret != 0 &&
        ret != MBEDTLS_ERR_RSA_KEY_CHECK_FAILED &&
        ret != MBEDTLS_ERR_RSA_BAD_INPUT_DATA) {
        ret += MBEDTLS_ERR_RSA_KEY_CHECK_FAILED;
    }

    mbedtls_mpi_free(&K);
    mbedtls_mpi_free(&L);

    return ret;
}

#endif /* MBEDTLS_RSA_C */
