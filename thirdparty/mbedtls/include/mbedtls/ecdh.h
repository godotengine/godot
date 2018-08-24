/**
 * \file ecdh.h
 *
 * \brief This file contains ECDH definitions and functions.
 *
 * The Elliptic Curve Diffie-Hellman (ECDH) protocol is an anonymous
 * key agreement protocol allowing two parties to establish a shared
 * secret over an insecure channel. Each party must have an
 * elliptic-curve public–private key pair.
 *
 * For more information, see <em>NIST SP 800-56A Rev. 2: Recommendation for
 * Pair-Wise Key Establishment Schemes Using Discrete Logarithm
 * Cryptography</em>.
 */
/*
 *  Copyright (C) 2006-2018, Arm Limited (or its affiliates), All Rights Reserved
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
 *  This file is part of Mbed TLS (https://tls.mbed.org)
 */

#ifndef MBEDTLS_ECDH_H
#define MBEDTLS_ECDH_H

#include "ecp.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Defines the source of the imported EC key.
 */
typedef enum
{
    MBEDTLS_ECDH_OURS,   /**< Our key. */
    MBEDTLS_ECDH_THEIRS, /**< The key of the peer. */
} mbedtls_ecdh_side;

/**
 * \brief           The ECDH context structure.
 */
typedef struct
{
    mbedtls_ecp_group grp;   /*!< The elliptic curve used. */
    mbedtls_mpi d;           /*!< The private key. */
    mbedtls_ecp_point Q;     /*!< The public key. */
    mbedtls_ecp_point Qp;    /*!< The value of the public key of the peer. */
    mbedtls_mpi z;           /*!< The shared secret. */
    int point_format;        /*!< The format of point export in TLS messages. */
    mbedtls_ecp_point Vi;    /*!< The blinding value. */
    mbedtls_ecp_point Vf;    /*!< The unblinding value. */
    mbedtls_mpi _d;          /*!< The previous \p d. */
}
mbedtls_ecdh_context;

/**
 * \brief           This function generates an ECDH keypair on an elliptic
 *                  curve.
 *
 *                  This function performs the first of two core computations
 *                  implemented during the ECDH key exchange. The second core
 *                  computation is performed by mbedtls_ecdh_compute_shared().
 *
 * \see             ecp.h
 *
 * \param grp       The ECP group.
 * \param d         The destination MPI (private key).
 * \param Q         The destination point (public key).
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or
 *                  \c MBEDTLS_MPI_XXX error code on failure.
 *
 */
int mbedtls_ecdh_gen_public( mbedtls_ecp_group *grp, mbedtls_mpi *d, mbedtls_ecp_point *Q,
                     int (*f_rng)(void *, unsigned char *, size_t),
                     void *p_rng );

/**
 * \brief           This function computes the shared secret.
 *
 *                  This function performs the second of two core computations
 *                  implemented during the ECDH key exchange. The first core
 *                  computation is performed by mbedtls_ecdh_gen_public().
 *
 * \see             ecp.h
 *
 * \note            If \p f_rng is not NULL, it is used to implement
 *                  countermeasures against side-channel attacks.
 *                  For more information, see mbedtls_ecp_mul().
 *
 * \param grp       The ECP group.
 * \param z         The destination MPI (shared secret).
 * \param Q         The public key from another party.
 * \param d         Our secret exponent (private key).
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or
 *                  \c MBEDTLS_MPI_XXX error code on failure.
 */
int mbedtls_ecdh_compute_shared( mbedtls_ecp_group *grp, mbedtls_mpi *z,
                         const mbedtls_ecp_point *Q, const mbedtls_mpi *d,
                         int (*f_rng)(void *, unsigned char *, size_t),
                         void *p_rng );

/**
 * \brief           This function initializes an ECDH context.
 *
 * \param ctx       The ECDH context to initialize.
 */
void mbedtls_ecdh_init( mbedtls_ecdh_context *ctx );

/**
 * \brief           This function frees a context.
 *
 * \param ctx       The context to free.
 */
void mbedtls_ecdh_free( mbedtls_ecdh_context *ctx );

/**
 * \brief           This function generates a public key and a TLS
 *                  ServerKeyExchange payload.
 *
 *                  This is the first function used by a TLS server for ECDHE
 *                  ciphersuites.
 *
 * \note            This function assumes that the ECP group (grp) of the
 *                  \p ctx context has already been properly set,
 *                  for example, using mbedtls_ecp_group_load().
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param olen      The number of characters written.
 * \param buf       The destination buffer.
 * \param blen      The length of the destination buffer.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_ecdh_make_params( mbedtls_ecdh_context *ctx, size_t *olen,
                      unsigned char *buf, size_t blen,
                      int (*f_rng)(void *, unsigned char *, size_t),
                      void *p_rng );

/**
 * \brief           This function parses and processes a TLS ServerKeyExhange
 *                  payload.
 *
 *                  This is the first function used by a TLS client for ECDHE
 *                  ciphersuites.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param buf       The pointer to the start of the input buffer.
 * \param end       The address for one Byte past the end of the buffer.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 *
 */
int mbedtls_ecdh_read_params( mbedtls_ecdh_context *ctx,
                      const unsigned char **buf, const unsigned char *end );

/**
 * \brief           This function sets up an ECDH context from an EC key.
 *
 *                  It is used by clients and servers in place of the
 *                  ServerKeyEchange for static ECDH, and imports ECDH
 *                  parameters from the EC key information of a certificate.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context to set up.
 * \param key       The EC key to use.
 * \param side      Defines the source of the key: 1: Our key, or
 *                  0: The key of the peer.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 *
 */
int mbedtls_ecdh_get_params( mbedtls_ecdh_context *ctx, const mbedtls_ecp_keypair *key,
                     mbedtls_ecdh_side side );

/**
 * \brief           This function generates a public key and a TLS
 *                  ClientKeyExchange payload.
 *
 *                  This is the second function used by a TLS client for ECDH(E)
 *                  ciphersuites.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param olen      The number of Bytes written.
 * \param buf       The destination buffer.
 * \param blen      The size of the destination buffer.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_ecdh_make_public( mbedtls_ecdh_context *ctx, size_t *olen,
                      unsigned char *buf, size_t blen,
                      int (*f_rng)(void *, unsigned char *, size_t),
                      void *p_rng );

/**
 * \brief       This function parses and processes a TLS ClientKeyExchange
 *              payload.
 *
 *              This is the second function used by a TLS server for ECDH(E)
 *              ciphersuites.
 *
 * \see         ecp.h
 *
 * \param ctx   The ECDH context.
 * \param buf   The start of the input buffer.
 * \param blen  The length of the input buffer.
 *
 * \return      \c 0 on success.
 * \return      An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_ecdh_read_public( mbedtls_ecdh_context *ctx,
                      const unsigned char *buf, size_t blen );

/**
 * \brief           This function derives and exports the shared secret.
 *
 *                  This is the last function used by both TLS client
 *                  and servers.
 *
 * \note            If \p f_rng is not NULL, it is used to implement
 *                  countermeasures against side-channel attacks.
 *                  For more information, see mbedtls_ecp_mul().
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDH context.
 * \param olen      The number of Bytes written.
 * \param buf       The destination buffer.
 * \param blen      The length of the destination buffer.
 * \param f_rng     The RNG function.
 * \param p_rng     The RNG context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX error code on failure.
 */
int mbedtls_ecdh_calc_secret( mbedtls_ecdh_context *ctx, size_t *olen,
                      unsigned char *buf, size_t blen,
                      int (*f_rng)(void *, unsigned char *, size_t),
                      void *p_rng );

#ifdef __cplusplus
}
#endif

#endif /* ecdh.h */
