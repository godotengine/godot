/**
 *  Low level bignum functions
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef MBEDTLS_BIGNUM_INTERNAL_H
#define MBEDTLS_BIGNUM_INTERNAL_H

#include "mbedtls/bignum.h"

/**
 * \brief Calculate the square of the Montgomery constant. (Needed
 *        for conversion and operations in Montgomery form.)
 *
 * \param[out] X  A pointer to the result of the calculation of
 *                the square of the Montgomery constant:
 *                2^{2*n*biL} mod N.
 * \param[in]  N  Little-endian presentation of the modulus, which must be odd.
 *
 * \return        0 if successful.
 * \return        #MBEDTLS_ERR_MPI_ALLOC_FAILED if there is not enough space
 *                to store the value of Montgomery constant squared.
 * \return        #MBEDTLS_ERR_MPI_DIVISION_BY_ZERO if \p N modulus is zero.
 * \return        #MBEDTLS_ERR_MPI_NEGATIVE_VALUE if \p N modulus is negative.
 */
int mbedtls_mpi_get_mont_r2_unsafe(mbedtls_mpi *X,
                                   const mbedtls_mpi *N);

/**
 * \brief Calculate initialisation value for fast Montgomery modular
 *        multiplication
 *
 * \param[in] N  Little-endian presentation of the modulus. This must have
 *               at least one limb.
 *
 * \return       The initialisation value for fast Montgomery modular multiplication
 */
mbedtls_mpi_uint mbedtls_mpi_montmul_init(const mbedtls_mpi_uint *N);

/** Montgomery multiplication: A = A * B * R^-1 mod N  (HAC 14.36)
 *
 * \param[in,out]   A   One of the numbers to multiply.
 *                      It must have at least as many limbs as N
 *                      (A->n >= N->n), and any limbs beyond n are ignored.
 *                      On successful completion, A contains the result of
 *                      the multiplication A * B * R^-1 mod N where
 *                      R = (2^ciL)^n.
 * \param[in]       B   One of the numbers to multiply.
 *                      It must be nonzero and must not have more limbs than N
 *                      (B->n <= N->n).
 * \param[in]       N   The modulo. N must be odd.
 * \param           mm  The value calculated by
 *                      `mbedtls_mpi_montg_init(&mm, N)`.
 *                      This is -N^-1 mod 2^ciL.
 * \param[in,out]   T   A bignum for temporary storage.
 *                      It must be at least twice the limb size of N plus 2
 *                      (T->n >= 2 * (N->n + 1)).
 *                      Its initial content is unused and
 *                      its final content is indeterminate.
 *                      Note that unlike the usual convention in the library
 *                      for `const mbedtls_mpi*`, the content of T can change.
 */
void mbedtls_mpi_montmul(mbedtls_mpi *A,
                         const mbedtls_mpi *B,
                         const mbedtls_mpi *N,
                         mbedtls_mpi_uint mm,
                         const mbedtls_mpi *T);

#endif /* MBEDTLS_BIGNUM_INTERNAL_H */
