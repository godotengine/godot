/**
 * \file ecp_invasive.h
 *
 * \brief ECP module: interfaces for invasive testing only.
 *
 * The interfaces in this file are intended for testing purposes only.
 * They SHOULD NOT be made available in library integrations except when
 * building the library for testing.
 */
/*
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
#ifndef MBEDTLS_ECP_INVASIVE_H
#define MBEDTLS_ECP_INVASIVE_H

#include "common.h"
#include "mbedtls/bignum.h"
#include "mbedtls/ecp.h"

#if defined(MBEDTLS_TEST_HOOKS) && defined(MBEDTLS_ECP_C)

#if defined(MBEDTLS_ECP_DP_SECP224R1_ENABLED) ||   \
    defined(MBEDTLS_ECP_DP_SECP256R1_ENABLED) ||   \
    defined(MBEDTLS_ECP_DP_SECP384R1_ENABLED)
/* Preconditions:
 *   - bits is a multiple of 64 or is 224
 *   - c is -1 or -2
 *   - 0 <= N < 2^bits
 *   - N has room for bits plus one limb
 *
 * Behavior:
 * Set N to c * 2^bits + old_value_of_N.
 */
void mbedtls_ecp_fix_negative( mbedtls_mpi *N, signed char c, size_t bits );
#endif

#if defined(MBEDTLS_ECP_MONTGOMERY_ENABLED)
/** Generate a private key on a Montgomery curve (Curve25519 or Curve448).
 *
 * This function implements key generation for the set of secret keys
 * specified in [Curve25519] p. 5 and in [Curve448]. The resulting value
 * has the lower bits masked but is not necessarily canonical.
 *
 * \note            - [Curve25519] http://cr.yp.to/ecdh/curve25519-20060209.pdf
 *                  - [RFC7748] https://tools.ietf.org/html/rfc7748
 *
 * \p high_bit      The position of the high-order bit of the key to generate.
 *                  This is the bit-size of the key minus 1:
 *                  254 for Curve25519 or 447 for Curve448.
 * \param d         The randomly generated key. This is a number of size
 *                  exactly \p n_bits + 1 bits, with the least significant bits
 *                  masked as specified in [Curve25519] and in [RFC7748] §5.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context to be passed to \p f_rng.
 *
 * \return          \c 0 on success.
 * \return          \c MBEDTLS_ERR_ECP_xxx or MBEDTLS_ERR_MPI_xxx on failure.
 */
int mbedtls_ecp_gen_privkey_mx( size_t n_bits,
                                mbedtls_mpi *d,
                                int (*f_rng)(void *, unsigned char *, size_t),
                                void *p_rng );

#endif /* MBEDTLS_ECP_MONTGOMERY_ENABLED */

#endif /* MBEDTLS_TEST_HOOKS && MBEDTLS_ECP_C */

#endif /* MBEDTLS_ECP_INVASIVE_H */
