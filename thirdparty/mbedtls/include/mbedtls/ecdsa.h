/**
 * \file ecdsa.h
 *
 * \brief This file contains ECDSA definitions and functions.
 *
 * The Elliptic Curve Digital Signature Algorithm (ECDSA) is defined in
 * <em>Standards for Efficient Cryptography Group (SECG):
 * SEC1 Elliptic Curve Cryptography</em>.
 * The use of ECDSA for TLS is defined in <em>RFC-4492: Elliptic Curve
 * Cryptography (ECC) Cipher Suites for Transport Layer Security (TLS)</em>.
 *
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

#ifndef MBEDTLS_ECDSA_H
#define MBEDTLS_ECDSA_H

#include "ecp.h"
#include "md.h"

/*
 * RFC-4492 page 20:
 *
 *     Ecdsa-Sig-Value ::= SEQUENCE {
 *         r       INTEGER,
 *         s       INTEGER
 *     }
 *
 * Size is at most
 *    1 (tag) + 1 (len) + 1 (initial 0) + ECP_MAX_BYTES for each of r and s,
 *    twice that + 1 (tag) + 2 (len) for the sequence
 * (assuming ECP_MAX_BYTES is less than 126 for r and s,
 * and less than 124 (total len <= 255) for the sequence)
 */
#if MBEDTLS_ECP_MAX_BYTES > 124
#error "MBEDTLS_ECP_MAX_BYTES bigger than expected, please fix MBEDTLS_ECDSA_MAX_LEN"
#endif
/** The maximal size of an ECDSA signature in Bytes. */
#define MBEDTLS_ECDSA_MAX_LEN  ( 3 + 2 * ( 3 + MBEDTLS_ECP_MAX_BYTES ) )

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief           The ECDSA context structure.
 *
 * \warning         Performing multiple operations concurrently on the same
 *                  ECDSA context is not supported; objects of this type
 *                  should not be shared between multiple threads.
 */
typedef mbedtls_ecp_keypair mbedtls_ecdsa_context;

#if defined(MBEDTLS_ECP_RESTARTABLE)

/**
 * \brief           Internal restart context for ecdsa_verify()
 *
 * \note            Opaque struct, defined in ecdsa.c
 */
typedef struct mbedtls_ecdsa_restart_ver mbedtls_ecdsa_restart_ver_ctx;

/**
 * \brief           Internal restart context for ecdsa_sign()
 *
 * \note            Opaque struct, defined in ecdsa.c
 */
typedef struct mbedtls_ecdsa_restart_sig mbedtls_ecdsa_restart_sig_ctx;

#if defined(MBEDTLS_ECDSA_DETERMINISTIC)
/**
 * \brief           Internal restart context for ecdsa_sign_det()
 *
 * \note            Opaque struct, defined in ecdsa.c
 */
typedef struct mbedtls_ecdsa_restart_det mbedtls_ecdsa_restart_det_ctx;
#endif

/**
 * \brief           General context for resuming ECDSA operations
 */
typedef struct
{
    mbedtls_ecp_restart_ctx ecp;        /*!<  base context for ECP restart and
                                              shared administrative info    */
    mbedtls_ecdsa_restart_ver_ctx *ver; /*!<  ecdsa_verify() sub-context    */
    mbedtls_ecdsa_restart_sig_ctx *sig; /*!<  ecdsa_sign() sub-context      */
#if defined(MBEDTLS_ECDSA_DETERMINISTIC)
    mbedtls_ecdsa_restart_det_ctx *det; /*!<  ecdsa_sign_det() sub-context  */
#endif
} mbedtls_ecdsa_restart_ctx;

#else /* MBEDTLS_ECP_RESTARTABLE */

/* Now we can declare functions that take a pointer to that */
typedef void mbedtls_ecdsa_restart_ctx;

#endif /* MBEDTLS_ECP_RESTARTABLE */

/**
 * \brief           This function computes the ECDSA signature of a
 *                  previously-hashed message.
 *
 * \note            The deterministic version implemented in
 *                  mbedtls_ecdsa_sign_det() is usually preferred.
 *
 * \note            If the bitlength of the message hash is larger than the
 *                  bitlength of the group order, then the hash is truncated
 *                  as defined in <em>Standards for Efficient Cryptography Group
 *                  (SECG): SEC1 Elliptic Curve Cryptography</em>, section
 *                  4.1.3, step 5.
 *
 * \see             ecp.h
 *
 * \param grp       The context for the elliptic curve to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param r         The MPI context in which to store the first part
 *                  the signature. This must be initialized.
 * \param s         The MPI context in which to store the second part
 *                  the signature. This must be initialized.
 * \param d         The private signing key. This must be initialized.
 * \param buf       The content to be signed. This is usually the hash of
 *                  the original data to be signed. This must be a readable
 *                  buffer of length \p blen Bytes. It may be \c NULL if
 *                  \p blen is zero.
 * \param blen      The length of \p buf in Bytes.
 * \param f_rng     The RNG function. This must not be \c NULL.
 * \param p_rng     The RNG context to be passed to \p f_rng. This may be
 *                  \c NULL if \p f_rng doesn't need a context parameter.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX
 *                  or \c MBEDTLS_MPI_XXX error code on failure.
 */
int mbedtls_ecdsa_sign( mbedtls_ecp_group *grp, mbedtls_mpi *r, mbedtls_mpi *s,
                const mbedtls_mpi *d, const unsigned char *buf, size_t blen,
                int (*f_rng)(void *, unsigned char *, size_t), void *p_rng );

#if defined(MBEDTLS_ECDSA_DETERMINISTIC)
/**
 * \brief           This function computes the ECDSA signature of a
 *                  previously-hashed message, deterministic version.
 *
 *                  For more information, see <em>RFC-6979: Deterministic
 *                  Usage of the Digital Signature Algorithm (DSA) and Elliptic
 *                  Curve Digital Signature Algorithm (ECDSA)</em>.
 *
 * \note            If the bitlength of the message hash is larger than the
 *                  bitlength of the group order, then the hash is truncated as
 *                  defined in <em>Standards for Efficient Cryptography Group
 *                  (SECG): SEC1 Elliptic Curve Cryptography</em>, section
 *                  4.1.3, step 5.
 *
 * \see             ecp.h
 *
 * \param grp       The context for the elliptic curve to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param r         The MPI context in which to store the first part
 *                  the signature. This must be initialized.
 * \param s         The MPI context in which to store the second part
 *                  the signature. This must be initialized.
 * \param d         The private signing key. This must be initialized
 *                  and setup, for example through mbedtls_ecp_gen_privkey().
 * \param buf       The hashed content to be signed. This must be a readable
 *                  buffer of length \p blen Bytes. It may be \c NULL if
 *                  \p blen is zero.
 * \param blen      The length of \p buf in Bytes.
 * \param md_alg    The hash algorithm used to hash the original data.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_MPI_XXX
 *                  error code on failure.
 */
int mbedtls_ecdsa_sign_det( mbedtls_ecp_group *grp, mbedtls_mpi *r,
                            mbedtls_mpi *s, const mbedtls_mpi *d,
                            const unsigned char *buf, size_t blen,
                            mbedtls_md_type_t md_alg );
#endif /* MBEDTLS_ECDSA_DETERMINISTIC */

/**
 * \brief           This function verifies the ECDSA signature of a
 *                  previously-hashed message.
 *
 * \note            If the bitlength of the message hash is larger than the
 *                  bitlength of the group order, then the hash is truncated as
 *                  defined in <em>Standards for Efficient Cryptography Group
 *                  (SECG): SEC1 Elliptic Curve Cryptography</em>, section
 *                  4.1.4, step 3.
 *
 * \see             ecp.h
 *
 * \param grp       The ECP group to use.
 *                  This must be initialized and have group parameters
 *                  set, for example through mbedtls_ecp_group_load().
 * \param buf       The hashed content that was signed. This must be a readable
 *                  buffer of length \p blen Bytes. It may be \c NULL if
 *                  \p blen is zero.
 * \param blen      The length of \p buf in Bytes.
 * \param Q         The public key to use for verification. This must be
 *                  initialized and setup.
 * \param r         The first integer of the signature.
 *                  This must be initialized.
 * \param s         The second integer of the signature.
 *                  This must be initialized.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if the signature
 *                  is invalid.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_MPI_XXX
 *                  error code on failure for any other reason.
 */
int mbedtls_ecdsa_verify( mbedtls_ecp_group *grp,
                          const unsigned char *buf, size_t blen,
                          const mbedtls_ecp_point *Q, const mbedtls_mpi *r,
                          const mbedtls_mpi *s);

/**
 * \brief           This function computes the ECDSA signature and writes it
 *                  to a buffer, serialized as defined in <em>RFC-4492:
 *                  Elliptic Curve Cryptography (ECC) Cipher Suites for
 *                  Transport Layer Security (TLS)</em>.
 *
 * \warning         It is not thread-safe to use the same context in
 *                  multiple threads.
 *
 * \note            The deterministic version is used if
 *                  #MBEDTLS_ECDSA_DETERMINISTIC is defined. For more
 *                  information, see <em>RFC-6979: Deterministic Usage
 *                  of the Digital Signature Algorithm (DSA) and Elliptic
 *                  Curve Digital Signature Algorithm (ECDSA)</em>.
 *
 * \note            If the bitlength of the message hash is larger than the
 *                  bitlength of the group order, then the hash is truncated as
 *                  defined in <em>Standards for Efficient Cryptography Group
 *                  (SECG): SEC1 Elliptic Curve Cryptography</em>, section
 *                  4.1.3, step 5.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDSA context to use. This must be initialized
 *                  and have a group and private key bound to it, for example
 *                  via mbedtls_ecdsa_genkey() or mbedtls_ecdsa_from_keypair().
 * \param md_alg    The message digest that was used to hash the message.
 * \param hash      The message hash to be signed. This must be a readable
 *                  buffer of length \p blen Bytes.
 * \param hlen      The length of the hash \p hash in Bytes.
 * \param sig       The buffer to which to write the signature. This must be a
 *                  writable buffer of length at least twice as large as the
 *                  size of the curve used, plus 9. For example, 73 Bytes if
 *                  a 256-bit curve is used. A buffer length of
 *                  #MBEDTLS_ECDSA_MAX_LEN is always safe.
 * \param slen      The address at which to store the actual length of
 *                  the signature written. Must not be \c NULL.
 * \param f_rng     The RNG function. This must not be \c NULL if
 *                  #MBEDTLS_ECDSA_DETERMINISTIC is unset. Otherwise,
 *                  it is unused and may be set to \c NULL.
 * \param p_rng     The RNG context to be passed to \p f_rng. This may be
 *                  \c NULL if \p f_rng is \c NULL or doesn't use a context.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX, \c MBEDTLS_ERR_MPI_XXX or
 *                  \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_ecdsa_write_signature( mbedtls_ecdsa_context *ctx,
                                   mbedtls_md_type_t md_alg,
                           const unsigned char *hash, size_t hlen,
                           unsigned char *sig, size_t *slen,
                           int (*f_rng)(void *, unsigned char *, size_t),
                           void *p_rng );

/**
 * \brief           This function computes the ECDSA signature and writes it
 *                  to a buffer, in a restartable way.
 *
 * \see             \c mbedtls_ecdsa_write_signature()
 *
 * \note            This function is like \c mbedtls_ecdsa_write_signature()
 *                  but it can return early and restart according to the limit
 *                  set with \c mbedtls_ecp_set_max_ops() to reduce blocking.
 *
 * \param ctx       The ECDSA context to use. This must be initialized
 *                  and have a group and private key bound to it, for example
 *                  via mbedtls_ecdsa_genkey() or mbedtls_ecdsa_from_keypair().
 * \param md_alg    The message digest that was used to hash the message.
 * \param hash      The message hash to be signed. This must be a readable
 *                  buffer of length \p blen Bytes.
 * \param hlen      The length of the hash \p hash in Bytes.
 * \param sig       The buffer to which to write the signature. This must be a
 *                  writable buffer of length at least twice as large as the
 *                  size of the curve used, plus 9. For example, 73 Bytes if
 *                  a 256-bit curve is used. A buffer length of
 *                  #MBEDTLS_ECDSA_MAX_LEN is always safe.
 * \param slen      The address at which to store the actual length of
 *                  the signature written. Must not be \c NULL.
 * \param f_rng     The RNG function. This must not be \c NULL if
 *                  #MBEDTLS_ECDSA_DETERMINISTIC is unset. Otherwise,
 *                  it is unused and may be set to \c NULL.
 * \param p_rng     The RNG context to be passed to \p f_rng. This may be
 *                  \c NULL if \p f_rng is \c NULL or doesn't use a context.
 * \param rs_ctx    The restart context to use. This may be \c NULL to disable
 *                  restarting. If it is not \c NULL, it must point to an
 *                  initialized restart context.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                  operations was reached: see \c mbedtls_ecp_set_max_ops().
 * \return          Another \c MBEDTLS_ERR_ECP_XXX, \c MBEDTLS_ERR_MPI_XXX or
 *                  \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_ecdsa_write_signature_restartable( mbedtls_ecdsa_context *ctx,
                           mbedtls_md_type_t md_alg,
                           const unsigned char *hash, size_t hlen,
                           unsigned char *sig, size_t *slen,
                           int (*f_rng)(void *, unsigned char *, size_t),
                           void *p_rng,
                           mbedtls_ecdsa_restart_ctx *rs_ctx );

#if defined(MBEDTLS_ECDSA_DETERMINISTIC)
#if ! defined(MBEDTLS_DEPRECATED_REMOVED)
#if defined(MBEDTLS_DEPRECATED_WARNING)
#define MBEDTLS_DEPRECATED    __attribute__((deprecated))
#else
#define MBEDTLS_DEPRECATED
#endif
/**
 * \brief           This function computes an ECDSA signature and writes
 *                  it to a buffer, serialized as defined in <em>RFC-4492:
 *                  Elliptic Curve Cryptography (ECC) Cipher Suites for
 *                  Transport Layer Security (TLS)</em>.
 *
 *                  The deterministic version is defined in <em>RFC-6979:
 *                  Deterministic Usage of the Digital Signature Algorithm (DSA)
 *                  and Elliptic Curve Digital Signature Algorithm (ECDSA)</em>.
 *
 * \warning         It is not thread-safe to use the same context in
 *                  multiple threads.
 *
 * \note            If the bitlength of the message hash is larger than the
 *                  bitlength of the group order, then the hash is truncated as
 *                  defined in <em>Standards for Efficient Cryptography Group
 *                  (SECG): SEC1 Elliptic Curve Cryptography</em>, section
 *                  4.1.3, step 5.
 *
 * \see             ecp.h
 *
 * \deprecated      Superseded by mbedtls_ecdsa_write_signature() in
 *                  Mbed TLS version 2.0 and later.
 *
 * \param ctx       The ECDSA context to use. This must be initialized
 *                  and have a group and private key bound to it, for example
 *                  via mbedtls_ecdsa_genkey() or mbedtls_ecdsa_from_keypair().
 * \param hash      The message hash to be signed. This must be a readable
 *                  buffer of length \p blen Bytes.
 * \param hlen      The length of the hash \p hash in Bytes.
 * \param sig       The buffer to which to write the signature. This must be a
 *                  writable buffer of length at least twice as large as the
 *                  size of the curve used, plus 9. For example, 73 Bytes if
 *                  a 256-bit curve is used. A buffer length of
 *                  #MBEDTLS_ECDSA_MAX_LEN is always safe.
 * \param slen      The address at which to store the actual length of
 *                  the signature written. Must not be \c NULL.
 * \param md_alg    The message digest that was used to hash the message.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX, \c MBEDTLS_ERR_MPI_XXX or
 *                  \c MBEDTLS_ERR_ASN1_XXX error code on failure.
 */
int mbedtls_ecdsa_write_signature_det( mbedtls_ecdsa_context *ctx,
                               const unsigned char *hash, size_t hlen,
                               unsigned char *sig, size_t *slen,
                               mbedtls_md_type_t md_alg ) MBEDTLS_DEPRECATED;
#undef MBEDTLS_DEPRECATED
#endif /* MBEDTLS_DEPRECATED_REMOVED */
#endif /* MBEDTLS_ECDSA_DETERMINISTIC */

/**
 * \brief           This function reads and verifies an ECDSA signature.
 *
 * \note            If the bitlength of the message hash is larger than the
 *                  bitlength of the group order, then the hash is truncated as
 *                  defined in <em>Standards for Efficient Cryptography Group
 *                  (SECG): SEC1 Elliptic Curve Cryptography</em>, section
 *                  4.1.4, step 3.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDSA context to use. This must be initialized
 *                  and have a group and public key bound to it.
 * \param hash      The message hash that was signed. This must be a readable
 *                  buffer of length \p size Bytes.
 * \param hlen      The size of the hash \p hash.
 * \param sig       The signature to read and verify. This must be a readable
 *                  buffer of length \p slen Bytes.
 * \param slen      The size of \p sig in Bytes.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if signature is invalid.
 * \return          #MBEDTLS_ERR_ECP_SIG_LEN_MISMATCH if there is a valid
 *                  signature in \p sig, but its length is less than \p siglen.
 * \return          An \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_ERR_MPI_XXX
 *                  error code on failure for any other reason.
 */
int mbedtls_ecdsa_read_signature( mbedtls_ecdsa_context *ctx,
                          const unsigned char *hash, size_t hlen,
                          const unsigned char *sig, size_t slen );

/**
 * \brief           This function reads and verifies an ECDSA signature,
 *                  in a restartable way.
 *
 * \see             \c mbedtls_ecdsa_read_signature()
 *
 * \note            This function is like \c mbedtls_ecdsa_read_signature()
 *                  but it can return early and restart according to the limit
 *                  set with \c mbedtls_ecp_set_max_ops() to reduce blocking.
 *
 * \param ctx       The ECDSA context to use. This must be initialized
 *                  and have a group and public key bound to it.
 * \param hash      The message hash that was signed. This must be a readable
 *                  buffer of length \p size Bytes.
 * \param hlen      The size of the hash \p hash.
 * \param sig       The signature to read and verify. This must be a readable
 *                  buffer of length \p slen Bytes.
 * \param slen      The size of \p sig in Bytes.
 * \param rs_ctx    The restart context to use. This may be \c NULL to disable
 *                  restarting. If it is not \c NULL, it must point to an
 *                  initialized restart context.
 *
 * \return          \c 0 on success.
 * \return          #MBEDTLS_ERR_ECP_BAD_INPUT_DATA if signature is invalid.
 * \return          #MBEDTLS_ERR_ECP_SIG_LEN_MISMATCH if there is a valid
 *                  signature in \p sig, but its length is less than \p siglen.
 * \return          #MBEDTLS_ERR_ECP_IN_PROGRESS if maximum number of
 *                  operations was reached: see \c mbedtls_ecp_set_max_ops().
 * \return          Another \c MBEDTLS_ERR_ECP_XXX or \c MBEDTLS_ERR_MPI_XXX
 *                  error code on failure for any other reason.
 */
int mbedtls_ecdsa_read_signature_restartable( mbedtls_ecdsa_context *ctx,
                          const unsigned char *hash, size_t hlen,
                          const unsigned char *sig, size_t slen,
                          mbedtls_ecdsa_restart_ctx *rs_ctx );

/**
 * \brief          This function generates an ECDSA keypair on the given curve.
 *
 * \see            ecp.h
 *
 * \param ctx      The ECDSA context to store the keypair in.
 *                 This must be initialized.
 * \param gid      The elliptic curve to use. One of the various
 *                 \c MBEDTLS_ECP_DP_XXX macros depending on configuration.
 * \param f_rng    The RNG function to use. This must not be \c NULL.
 * \param p_rng    The RNG context to be passed to \p f_rng. This may be
 *                 \c NULL if \p f_rng doesn't need a context argument.
 *
 * \return         \c 0 on success.
 * \return         An \c MBEDTLS_ERR_ECP_XXX code on failure.
 */
int mbedtls_ecdsa_genkey( mbedtls_ecdsa_context *ctx, mbedtls_ecp_group_id gid,
                  int (*f_rng)(void *, unsigned char *, size_t), void *p_rng );

/**
 * \brief           This function sets up an ECDSA context from an EC key pair.
 *
 * \see             ecp.h
 *
 * \param ctx       The ECDSA context to setup. This must be initialized.
 * \param key       The EC key to use. This must be initialized and hold
 *                  a private-public key pair or a public key. In the former
 *                  case, the ECDSA context may be used for signature creation
 *                  and verification after this call. In the latter case, it
 *                  may be used for signature verification.
 *
 * \return          \c 0 on success.
 * \return          An \c MBEDTLS_ERR_ECP_XXX code on failure.
 */
int mbedtls_ecdsa_from_keypair( mbedtls_ecdsa_context *ctx,
                                const mbedtls_ecp_keypair *key );

/**
 * \brief           This function initializes an ECDSA context.
 *
 * \param ctx       The ECDSA context to initialize.
 *                  This must not be \c NULL.
 */
void mbedtls_ecdsa_init( mbedtls_ecdsa_context *ctx );

/**
 * \brief           This function frees an ECDSA context.
 *
 * \param ctx       The ECDSA context to free. This may be \c NULL,
 *                  in which case this function does nothing. If it
 *                  is not \c NULL, it must be initialized.
 */
void mbedtls_ecdsa_free( mbedtls_ecdsa_context *ctx );

#if defined(MBEDTLS_ECP_RESTARTABLE)
/**
 * \brief           Initialize a restart context.
 *
 * \param ctx       The restart context to initialize.
 *                  This must not be \c NULL.
 */
void mbedtls_ecdsa_restart_init( mbedtls_ecdsa_restart_ctx *ctx );

/**
 * \brief           Free the components of a restart context.
 *
 * \param ctx       The restart context to free. This may be \c NULL,
 *                  in which case this function does nothing. If it
 *                  is not \c NULL, it must be initialized.
 */
void mbedtls_ecdsa_restart_free( mbedtls_ecdsa_restart_ctx *ctx );
#endif /* MBEDTLS_ECP_RESTARTABLE */

#ifdef __cplusplus
}
#endif

#endif /* ecdsa.h */
