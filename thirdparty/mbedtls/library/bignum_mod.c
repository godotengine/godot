/**
 *  Modular bignum functions
 *
 *  Copyright The Mbed TLS Contributors
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
 */

#include "common.h"

#if defined(MBEDTLS_BIGNUM_C)

#include <string.h>

#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"
#include "mbedtls/bignum.h"

#include "mbedtls/platform.h"

#include "bignum_core.h"
#include "bignum_mod.h"
#include "bignum_mod_raw.h"
#include "constant_time_internal.h"

int mbedtls_mpi_mod_residue_setup(mbedtls_mpi_mod_residue *r,
                                  const mbedtls_mpi_mod_modulus *N,
                                  mbedtls_mpi_uint *p,
                                  size_t p_limbs)
{
    if (p_limbs != N->limbs || !mbedtls_mpi_core_lt_ct(p, N->p, N->limbs)) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    r->limbs = N->limbs;
    r->p = p;

    return 0;
}

void mbedtls_mpi_mod_residue_release(mbedtls_mpi_mod_residue *r)
{
    if (r == NULL) {
        return;
    }

    r->limbs = 0;
    r->p = NULL;
}

void mbedtls_mpi_mod_modulus_init(mbedtls_mpi_mod_modulus *N)
{
    if (N == NULL) {
        return;
    }

    N->p = NULL;
    N->limbs = 0;
    N->bits = 0;
    N->int_rep = MBEDTLS_MPI_MOD_REP_INVALID;
}

void mbedtls_mpi_mod_modulus_free(mbedtls_mpi_mod_modulus *N)
{
    if (N == NULL) {
        return;
    }

    switch (N->int_rep) {
        case MBEDTLS_MPI_MOD_REP_MONTGOMERY:
            if (N->rep.mont.rr != NULL) {
                mbedtls_platform_zeroize((mbedtls_mpi_uint *) N->rep.mont.rr,
                                         N->limbs * sizeof(mbedtls_mpi_uint));
                mbedtls_free((mbedtls_mpi_uint *) N->rep.mont.rr);
                N->rep.mont.rr = NULL;
            }
            N->rep.mont.mm = 0;
            break;
        case MBEDTLS_MPI_MOD_REP_OPT_RED:
            mbedtls_free(N->rep.ored);
            break;
        case MBEDTLS_MPI_MOD_REP_INVALID:
            break;
    }

    N->p = NULL;
    N->limbs = 0;
    N->bits = 0;
    N->int_rep = MBEDTLS_MPI_MOD_REP_INVALID;
}

static int set_mont_const_square(const mbedtls_mpi_uint **X,
                                 const mbedtls_mpi_uint *A,
                                 size_t limbs)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    mbedtls_mpi N;
    mbedtls_mpi RR;
    *X = NULL;

    mbedtls_mpi_init(&N);
    mbedtls_mpi_init(&RR);

    if (A == NULL || limbs == 0 || limbs >= (MBEDTLS_MPI_MAX_LIMBS / 2) - 2) {
        goto cleanup;
    }

    if (mbedtls_mpi_grow(&N, limbs)) {
        goto cleanup;
    }

    memcpy(N.p, A, sizeof(mbedtls_mpi_uint) * limbs);

    ret = mbedtls_mpi_core_get_mont_r2_unsafe(&RR, &N);

    if (ret == 0) {
        *X = RR.p;
        RR.p = NULL;
    }

cleanup:
    mbedtls_mpi_free(&N);
    mbedtls_mpi_free(&RR);
    ret = (ret != 0) ? MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED : 0;
    return ret;
}

int mbedtls_mpi_mod_modulus_setup(mbedtls_mpi_mod_modulus *N,
                                  const mbedtls_mpi_uint *p,
                                  size_t p_limbs,
                                  mbedtls_mpi_mod_rep_selector int_rep)
{
    int ret = 0;

    N->p = p;
    N->limbs = p_limbs;
    N->bits = mbedtls_mpi_core_bitlen(p, p_limbs);

    switch (int_rep) {
        case MBEDTLS_MPI_MOD_REP_MONTGOMERY:
            N->int_rep = int_rep;
            N->rep.mont.mm = mbedtls_mpi_core_montmul_init(N->p);
            ret = set_mont_const_square(&N->rep.mont.rr, N->p, N->limbs);
            break;
        case MBEDTLS_MPI_MOD_REP_OPT_RED:
            N->int_rep = int_rep;
            N->rep.ored = NULL;
            break;
        default:
            ret = MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
            goto exit;
    }

exit:

    if (ret != 0) {
        mbedtls_mpi_mod_modulus_free(N);
    }

    return ret;
}

/* BEGIN MERGE SLOT 1 */

/* END MERGE SLOT 1 */

/* BEGIN MERGE SLOT 2 */

int mbedtls_mpi_mod_mul(mbedtls_mpi_mod_residue *X,
                        const mbedtls_mpi_mod_residue *A,
                        const mbedtls_mpi_mod_residue *B,
                        const mbedtls_mpi_mod_modulus *N)
{
    if (N->limbs == 0) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    if (X->limbs != N->limbs || A->limbs != N->limbs || B->limbs != N->limbs) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    mbedtls_mpi_uint *T = mbedtls_calloc(N->limbs * 2 + 1, ciL);
    if (T == NULL) {
        return MBEDTLS_ERR_MPI_ALLOC_FAILED;
    }

    mbedtls_mpi_mod_raw_mul(X->p, A->p, B->p, N, T);

    mbedtls_free(T);

    return 0;
}

/* END MERGE SLOT 2 */

/* BEGIN MERGE SLOT 3 */
int mbedtls_mpi_mod_sub(mbedtls_mpi_mod_residue *X,
                        const mbedtls_mpi_mod_residue *A,
                        const mbedtls_mpi_mod_residue *B,
                        const mbedtls_mpi_mod_modulus *N)
{
    if (X->limbs != N->limbs || A->limbs != N->limbs || B->limbs != N->limbs) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    mbedtls_mpi_mod_raw_sub(X->p, A->p, B->p, N);

    return 0;
}

static int mbedtls_mpi_mod_inv_mont(mbedtls_mpi_mod_residue *X,
                                    const mbedtls_mpi_mod_residue *A,
                                    const mbedtls_mpi_mod_modulus *N,
                                    mbedtls_mpi_uint *working_memory)
{
    /* Input already in Montgomery form, so there's little to do */
    mbedtls_mpi_mod_raw_inv_prime(X->p, A->p,
                                  N->p, N->limbs,
                                  N->rep.mont.rr,
                                  working_memory);
    return 0;
}

static int mbedtls_mpi_mod_inv_non_mont(mbedtls_mpi_mod_residue *X,
                                        const mbedtls_mpi_mod_residue *A,
                                        const mbedtls_mpi_mod_modulus *N,
                                        mbedtls_mpi_uint *working_memory)
{
    /* Need to convert input into Montgomery form */

    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    mbedtls_mpi_mod_modulus Nmont;
    mbedtls_mpi_mod_modulus_init(&Nmont);

    MBEDTLS_MPI_CHK(mbedtls_mpi_mod_modulus_setup(&Nmont, N->p, N->limbs,
                                                  MBEDTLS_MPI_MOD_REP_MONTGOMERY));

    /* We'll use X->p to hold the Montgomery form of the input A->p */
    mbedtls_mpi_core_to_mont_rep(X->p, A->p, Nmont.p, Nmont.limbs,
                                 Nmont.rep.mont.mm, Nmont.rep.mont.rr,
                                 working_memory);

    mbedtls_mpi_mod_raw_inv_prime(X->p, X->p,
                                  Nmont.p, Nmont.limbs,
                                  Nmont.rep.mont.rr,
                                  working_memory);

    /* And convert back from Montgomery form */

    mbedtls_mpi_core_from_mont_rep(X->p, X->p, Nmont.p, Nmont.limbs,
                                   Nmont.rep.mont.mm, working_memory);

cleanup:
    mbedtls_mpi_mod_modulus_free(&Nmont);
    return ret;
}

int mbedtls_mpi_mod_inv(mbedtls_mpi_mod_residue *X,
                        const mbedtls_mpi_mod_residue *A,
                        const mbedtls_mpi_mod_modulus *N)
{
    if (X->limbs != N->limbs || A->limbs != N->limbs) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    /* Zero has the same value regardless of Montgomery form or not */
    if (mbedtls_mpi_core_check_zero_ct(A->p, A->limbs) == 0) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    size_t working_limbs =
        mbedtls_mpi_mod_raw_inv_prime_working_limbs(N->limbs);

    mbedtls_mpi_uint *working_memory = mbedtls_calloc(working_limbs,
                                                      sizeof(mbedtls_mpi_uint));
    if (working_memory == NULL) {
        return MBEDTLS_ERR_MPI_ALLOC_FAILED;
    }

    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    switch (N->int_rep) {
        case MBEDTLS_MPI_MOD_REP_MONTGOMERY:
            ret = mbedtls_mpi_mod_inv_mont(X, A, N, working_memory);
            break;
        case MBEDTLS_MPI_MOD_REP_OPT_RED:
            ret = mbedtls_mpi_mod_inv_non_mont(X, A, N, working_memory);
            break;
        default:
            ret = MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
            break;
    }

    mbedtls_platform_zeroize(working_memory,
                             working_limbs * sizeof(mbedtls_mpi_uint));
    mbedtls_free(working_memory);

    return ret;
}
/* END MERGE SLOT 3 */

/* BEGIN MERGE SLOT 4 */

/* END MERGE SLOT 4 */

/* BEGIN MERGE SLOT 5 */
int mbedtls_mpi_mod_add(mbedtls_mpi_mod_residue *X,
                        const mbedtls_mpi_mod_residue *A,
                        const mbedtls_mpi_mod_residue *B,
                        const mbedtls_mpi_mod_modulus *N)
{
    if (X->limbs != N->limbs || A->limbs != N->limbs || B->limbs != N->limbs) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    mbedtls_mpi_mod_raw_add(X->p, A->p, B->p, N);

    return 0;
}
/* END MERGE SLOT 5 */

/* BEGIN MERGE SLOT 6 */

int mbedtls_mpi_mod_random(mbedtls_mpi_mod_residue *X,
                           mbedtls_mpi_uint min,
                           const mbedtls_mpi_mod_modulus *N,
                           int (*f_rng)(void *, unsigned char *, size_t),
                           void *p_rng)
{
    if (X->limbs != N->limbs) {
        return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }
    return mbedtls_mpi_mod_raw_random(X->p, min, N, f_rng, p_rng);
}

/* END MERGE SLOT 6 */

/* BEGIN MERGE SLOT 7 */
int mbedtls_mpi_mod_read(mbedtls_mpi_mod_residue *r,
                         const mbedtls_mpi_mod_modulus *N,
                         const unsigned char *buf,
                         size_t buflen,
                         mbedtls_mpi_mod_ext_rep ext_rep)
{
    int ret = MBEDTLS_ERR_MPI_BAD_INPUT_DATA;

    /* Do our best to check if r and m have been set up */
    if (r->limbs == 0 || N->limbs == 0) {
        goto cleanup;
    }
    if (r->limbs != N->limbs) {
        goto cleanup;
    }

    ret = mbedtls_mpi_mod_raw_read(r->p, N, buf, buflen, ext_rep);
    if (ret != 0) {
        goto cleanup;
    }

    r->limbs = N->limbs;

    ret = mbedtls_mpi_mod_raw_canonical_to_modulus_rep(r->p, N);

cleanup:
    return ret;
}

int mbedtls_mpi_mod_write(const mbedtls_mpi_mod_residue *r,
                          const mbedtls_mpi_mod_modulus *N,
                          unsigned char *buf,
                          size_t buflen,
                          mbedtls_mpi_mod_ext_rep ext_rep)
{
    int ret = MBEDTLS_ERR_MPI_BAD_INPUT_DATA;

    /* Do our best to check if r and m have been set up */
    if (r->limbs == 0 || N->limbs == 0) {
        goto cleanup;
    }
    if (r->limbs != N->limbs) {
        goto cleanup;
    }

    if (N->int_rep == MBEDTLS_MPI_MOD_REP_MONTGOMERY) {
        ret = mbedtls_mpi_mod_raw_from_mont_rep(r->p, N);
        if (ret != 0) {
            goto cleanup;
        }
    }

    ret = mbedtls_mpi_mod_raw_write(r->p, N, buf, buflen, ext_rep);

    if (N->int_rep == MBEDTLS_MPI_MOD_REP_MONTGOMERY) {
        /* If this fails, the value of r is corrupted and we want to return
         * this error (as opposed to the error code from the write above) to
         * let the caller know. If it succeeds, we want to return the error
         * code from write above. */
        int conv_ret = mbedtls_mpi_mod_raw_to_mont_rep(r->p, N);
        if (ret == 0) {
            ret = conv_ret;
        }
    }

cleanup:

    return ret;
}
/* END MERGE SLOT 7 */

/* BEGIN MERGE SLOT 8 */

/* END MERGE SLOT 8 */

/* BEGIN MERGE SLOT 9 */

/* END MERGE SLOT 9 */

/* BEGIN MERGE SLOT 10 */

/* END MERGE SLOT 10 */

#endif /* MBEDTLS_BIGNUM_C */
