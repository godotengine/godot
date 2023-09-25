/*
 *  Low-level modular bignum functions
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

#include "mbedtls/error.h"
#include "mbedtls/platform_util.h"

#include "mbedtls/platform.h"

#include "bignum_core.h"
#include "bignum_mod_raw.h"
#include "bignum_mod.h"
#include "constant_time_internal.h"

#include "bignum_mod_raw_invasive.h"

void mbedtls_mpi_mod_raw_cond_assign(mbedtls_mpi_uint *X,
                                     const mbedtls_mpi_uint *A,
                                     const mbedtls_mpi_mod_modulus *N,
                                     unsigned char assign)
{
    mbedtls_mpi_core_cond_assign(X, A, N->limbs, assign);
}

void mbedtls_mpi_mod_raw_cond_swap(mbedtls_mpi_uint *X,
                                   mbedtls_mpi_uint *Y,
                                   const mbedtls_mpi_mod_modulus *N,
                                   unsigned char swap)
{
    mbedtls_mpi_core_cond_swap(X, Y, N->limbs, swap);
}

int mbedtls_mpi_mod_raw_read(mbedtls_mpi_uint *X,
                             const mbedtls_mpi_mod_modulus *N,
                             const unsigned char *input,
                             size_t input_length,
                             mbedtls_mpi_mod_ext_rep ext_rep)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    switch (ext_rep) {
        case MBEDTLS_MPI_MOD_EXT_REP_LE:
            ret = mbedtls_mpi_core_read_le(X, N->limbs,
                                           input, input_length);
            break;
        case MBEDTLS_MPI_MOD_EXT_REP_BE:
            ret = mbedtls_mpi_core_read_be(X, N->limbs,
                                           input, input_length);
            break;
        default:
            return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }

    if (ret != 0) {
        goto cleanup;
    }

    if (!mbedtls_mpi_core_lt_ct(X, N->p, N->limbs)) {
        ret = MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
        goto cleanup;
    }

cleanup:

    return ret;
}

int mbedtls_mpi_mod_raw_write(const mbedtls_mpi_uint *A,
                              const mbedtls_mpi_mod_modulus *N,
                              unsigned char *output,
                              size_t output_length,
                              mbedtls_mpi_mod_ext_rep ext_rep)
{
    switch (ext_rep) {
        case MBEDTLS_MPI_MOD_EXT_REP_LE:
            return mbedtls_mpi_core_write_le(A, N->limbs,
                                             output, output_length);
        case MBEDTLS_MPI_MOD_EXT_REP_BE:
            return mbedtls_mpi_core_write_be(A, N->limbs,
                                             output, output_length);
        default:
            return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }
}

/* BEGIN MERGE SLOT 1 */

/* END MERGE SLOT 1 */

/* BEGIN MERGE SLOT 2 */

void mbedtls_mpi_mod_raw_sub(mbedtls_mpi_uint *X,
                             const mbedtls_mpi_uint *A,
                             const mbedtls_mpi_uint *B,
                             const mbedtls_mpi_mod_modulus *N)
{
    mbedtls_mpi_uint c = mbedtls_mpi_core_sub(X, A, B, N->limbs);

    (void) mbedtls_mpi_core_add_if(X, N->p, N->limbs, (unsigned) c);
}

#if defined(MBEDTLS_TEST_HOOKS)

MBEDTLS_STATIC_TESTABLE
void mbedtls_mpi_mod_raw_fix_quasi_reduction(mbedtls_mpi_uint *X,
                                             const mbedtls_mpi_mod_modulus *N)
{
    mbedtls_mpi_uint c = mbedtls_mpi_core_sub(X, X, N->p, N->limbs);

    (void) mbedtls_mpi_core_add_if(X, N->p, N->limbs, (unsigned) c);
}

#endif /* MBEDTLS_TEST_HOOKS */

void mbedtls_mpi_mod_raw_mul(mbedtls_mpi_uint *X,
                             const mbedtls_mpi_uint *A,
                             const mbedtls_mpi_uint *B,
                             const mbedtls_mpi_mod_modulus *N,
                             mbedtls_mpi_uint *T)
{
    mbedtls_mpi_core_montmul(X, A, B, N->limbs, N->p, N->limbs,
                             N->rep.mont.mm, T);
}

/* END MERGE SLOT 2 */

/* BEGIN MERGE SLOT 3 */

size_t mbedtls_mpi_mod_raw_inv_prime_working_limbs(size_t AN_limbs)
{
    /* mbedtls_mpi_mod_raw_inv_prime() needs a temporary for the exponent,
     * which will be the same size as the modulus and input (AN_limbs),
     * and additional space to pass to mbedtls_mpi_core_exp_mod(). */
    return AN_limbs +
           mbedtls_mpi_core_exp_mod_working_limbs(AN_limbs, AN_limbs);
}

void mbedtls_mpi_mod_raw_inv_prime(mbedtls_mpi_uint *X,
                                   const mbedtls_mpi_uint *A,
                                   const mbedtls_mpi_uint *N,
                                   size_t AN_limbs,
                                   const mbedtls_mpi_uint *RR,
                                   mbedtls_mpi_uint *T)
{
    /* Inversion by power: g^|G| = 1 => g^(-1) = g^(|G|-1), and
     *                       |G| = N - 1, so we want
     *                 g^(|G|-1) = g^(N - 2)
     */

    /* Use the first AN_limbs of T to hold N - 2 */
    mbedtls_mpi_uint *Nminus2 = T;
    (void) mbedtls_mpi_core_sub_int(Nminus2, N, 2, AN_limbs);

    /* Rest of T is given to exp_mod for its working space */
    mbedtls_mpi_core_exp_mod(X,
                             A, N, AN_limbs, Nminus2, AN_limbs,
                             RR, T + AN_limbs);
}

/* END MERGE SLOT 3 */

/* BEGIN MERGE SLOT 4 */

/* END MERGE SLOT 4 */

/* BEGIN MERGE SLOT 5 */
void mbedtls_mpi_mod_raw_add(mbedtls_mpi_uint *X,
                             const mbedtls_mpi_uint *A,
                             const mbedtls_mpi_uint *B,
                             const mbedtls_mpi_mod_modulus *N)
{
    mbedtls_mpi_uint carry, borrow;
    carry  = mbedtls_mpi_core_add(X, A, B, N->limbs);
    borrow = mbedtls_mpi_core_sub(X, X, N->p, N->limbs);
    (void) mbedtls_mpi_core_add_if(X, N->p, N->limbs, (unsigned) (carry ^ borrow));
}
/* END MERGE SLOT 5 */

/* BEGIN MERGE SLOT 6 */

int mbedtls_mpi_mod_raw_canonical_to_modulus_rep(
    mbedtls_mpi_uint *X,
    const mbedtls_mpi_mod_modulus *N)
{
    switch (N->int_rep) {
        case MBEDTLS_MPI_MOD_REP_MONTGOMERY:
            return mbedtls_mpi_mod_raw_to_mont_rep(X, N);
        case MBEDTLS_MPI_MOD_REP_OPT_RED:
            return 0;
        default:
            return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }
}

int mbedtls_mpi_mod_raw_modulus_to_canonical_rep(
    mbedtls_mpi_uint *X,
    const mbedtls_mpi_mod_modulus *N)
{
    switch (N->int_rep) {
        case MBEDTLS_MPI_MOD_REP_MONTGOMERY:
            return mbedtls_mpi_mod_raw_from_mont_rep(X, N);
        case MBEDTLS_MPI_MOD_REP_OPT_RED:
            return 0;
        default:
            return MBEDTLS_ERR_MPI_BAD_INPUT_DATA;
    }
}

int mbedtls_mpi_mod_raw_random(mbedtls_mpi_uint *X,
                               mbedtls_mpi_uint min,
                               const mbedtls_mpi_mod_modulus *N,
                               int (*f_rng)(void *, unsigned char *, size_t),
                               void *p_rng)
{
    int ret = mbedtls_mpi_core_random(X, min, N->p, N->limbs, f_rng, p_rng);
    if (ret != 0) {
        return ret;
    }
    return mbedtls_mpi_mod_raw_canonical_to_modulus_rep(X, N);
}

/* END MERGE SLOT 6 */

/* BEGIN MERGE SLOT 7 */
int mbedtls_mpi_mod_raw_to_mont_rep(mbedtls_mpi_uint *X,
                                    const mbedtls_mpi_mod_modulus *N)
{
    mbedtls_mpi_uint *T;
    const size_t t_limbs = mbedtls_mpi_core_montmul_working_limbs(N->limbs);

    if ((T = (mbedtls_mpi_uint *) mbedtls_calloc(t_limbs, ciL)) == NULL) {
        return MBEDTLS_ERR_MPI_ALLOC_FAILED;
    }

    mbedtls_mpi_core_to_mont_rep(X, X, N->p, N->limbs,
                                 N->rep.mont.mm, N->rep.mont.rr, T);

    mbedtls_platform_zeroize(T, t_limbs * ciL);
    mbedtls_free(T);
    return 0;
}

int mbedtls_mpi_mod_raw_from_mont_rep(mbedtls_mpi_uint *X,
                                      const mbedtls_mpi_mod_modulus *N)
{
    const size_t t_limbs = mbedtls_mpi_core_montmul_working_limbs(N->limbs);
    mbedtls_mpi_uint *T;

    if ((T = (mbedtls_mpi_uint *) mbedtls_calloc(t_limbs, ciL)) == NULL) {
        return MBEDTLS_ERR_MPI_ALLOC_FAILED;
    }

    mbedtls_mpi_core_from_mont_rep(X, X, N->p, N->limbs, N->rep.mont.mm, T);

    mbedtls_platform_zeroize(T, t_limbs * ciL);
    mbedtls_free(T);
    return 0;
}

void mbedtls_mpi_mod_raw_neg(mbedtls_mpi_uint *X,
                             const mbedtls_mpi_uint *A,
                             const mbedtls_mpi_mod_modulus *N)
{
    mbedtls_mpi_core_sub(X, N->p, A, N->limbs);

    /* If A=0 initially, then X=N now. Detect this by
     * subtracting N and catching the carry. */
    mbedtls_mpi_uint borrow = mbedtls_mpi_core_sub(X, X, N->p, N->limbs);
    (void) mbedtls_mpi_core_add_if(X, N->p, N->limbs, (unsigned) borrow);
}
/* END MERGE SLOT 7 */

/* BEGIN MERGE SLOT 8 */

/* END MERGE SLOT 8 */

/* BEGIN MERGE SLOT 9 */

/* END MERGE SLOT 9 */

/* BEGIN MERGE SLOT 10 */

/* END MERGE SLOT 10 */

#endif /* MBEDTLS_BIGNUM_C */
